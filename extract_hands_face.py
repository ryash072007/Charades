"""Per-frame finger angles and face blendshapes for one clip.

Inputs : RGB clip (mp4) + DWPose keypoint JSON (same frames, same order).
Output : npz with
    finger_angles  (T, 2, 5, 4)  radians, hand index 0=left 1=right (person's side),
                   fingers thumb..pinky, [curl1, curl2, curl3, spread]
    hand_valid     (T, 2)        bool, raw detection (before gap filling)
    hand_source    (T, 2)        0=none 1=mediapipe 2=dwpose-2d
    face           (T, 7)        jawOpen, mouthSmile, mouthFrown, browInnerUp, browDown, eyeBlink, eyeWide
    face_valid     (T,)          bool

Hands: each wrist region (located from DWPose) is cropped, upscaled and passed to
MediaPipe HandLandmarker; its 3D world landmarks give the angles. Where MediaPipe
misses, DWPose's 2D hand keypoints are used if confident. Remaining gaps are
interpolated and everything is smoothed with a One Euro filter.

Run with ComfyUI's embedded Python.
"""
import json
import os

import av
import cv2
import numpy as np

FACE_CHANNELS = ["jawOpen", "mouthSmile", "mouthFrown", "browInnerUp", "browDown", "eyeBlink", "eyeWide"]
# Which MediaPipe blendshapes are averaged into each channel.
FACE_SOURCES = {
    "jawOpen": ["jawOpen"],
    "mouthSmile": ["mouthSmileLeft", "mouthSmileRight"],
    "mouthFrown": ["mouthFrownLeft", "mouthFrownRight"],
    "browInnerUp": ["browInnerUp"],
    "browDown": ["browDownLeft", "browDownRight"],
    "eyeBlink": ["eyeBlinkLeft", "eyeBlinkRight"],
    "eyeWide": ["eyeWideLeft", "eyeWideRight"],
}

# 21-point hand layout shared by MediaPipe and DWPose/OpenPose.
FINGER_CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
# OpenPose-18 body indices
BODY_RWRIST, BODY_LWRIST, BODY_RELBOW, BODY_LELBOW = 4, 7, 3, 6


# ----------------------------------------------------------------------------- filters
class OneEuro:
    """One Euro filter (Casiez et al. 2012) over arrays of any shape."""

    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * self.freq)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x.copy(), np.zeros_like(x)
            return x.copy()
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


def one_euro_sequence(seq, fps, min_cutoff, beta, d_cutoff=1.0, bidirectional=True):
    """Filter (T, ...) array. Bidirectional pass removes the filter's lag (offline use)."""
    def run(s):
        f = OneEuro(fps, min_cutoff, beta, d_cutoff)
        return np.stack([f(x) for x in s])
    out = run(seq)
    if bidirectional:
        out = 0.5 * (out + run(out[::-1])[::-1])
    return out


def fill_gaps(seq, valid):
    """Linear interpolation over invalid frames; edges hold the nearest valid value.
    seq (T, ...), valid (T,). Returns filled copy; zeros if nothing is valid."""
    seq = seq.copy()
    T = len(seq)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return np.zeros_like(seq)
    flat = seq.reshape(T, -1)
    t = np.arange(T)
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(t, idx, flat[idx, c])
    return flat.reshape(seq.shape)


# ----------------------------------------------------------------------------- geometry
def _angle(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def finger_angles(pts):
    """pts (21, 2|3) -> (5, 4) [curl1, curl2, curl3, spread] in radians.

    Curls are unsigned bend angles between consecutive bone segments (fingers
    rarely hyperextend), so they need no handedness or palm-normal sign.
    Spread is the in-palm angle of each finger's first segment relative to the
    middle finger, positive toward the pinky side.
    For the thumb, curl1 is the metacarpal's bend away from the index metacarpal
    (opposition) and spread is unused.
    """
    p = np.asarray(pts, dtype=np.float64)
    if p.shape[1] == 2:
        p = np.concatenate([p, np.zeros((21, 1))], 1)
    w = p[0]
    u = p[9] - w                      # palm "up" (toward middle finger)
    v = p[5] - p[17]                  # across palm, pinky -> index
    u /= np.linalg.norm(u) + 1e-9
    v -= u * np.dot(v, u)
    v /= np.linalg.norm(v) + 1e-9
    out = np.zeros((5, 4))
    for f, ch in enumerate(FINGER_CHAINS):
        base = w if f else p[1]
        segs = [p[ch[0]] - base] + [p[ch[i + 1]] - p[ch[i]] for i in range(3)]
        if f == 0:
            out[f, 0] = _angle(p[2] - p[1], p[5] - w)            # thumb opposition
            out[f, 1] = _angle(segs[1], segs[2])
            out[f, 2] = _angle(segs[2], segs[3])
        else:
            out[f, 0] = _angle(segs[0], segs[1])
            out[f, 1] = _angle(segs[1], segs[2])
            out[f, 2] = _angle(segs[2], segs[3])
            d = segs[1] - np.dot(segs[1], np.cross(u, v)) * np.cross(u, v)  # project onto palm plane
            # positive toward pinky = negative v
            out[f, 3] = float(np.arctan2(-np.dot(d, v), np.dot(d, u)))
    # spread relative to the middle finger so a flat open hand reads ~0
    out[1:, 3] -= out[2, 3]
    return out


# ----------------------------------------------------------------------------- io
def read_video(path):
    c = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    return frames


def _kp(arr, n):
    if not arr:
        return np.zeros((n, 3))
    return np.asarray(arr, dtype=np.float64).reshape(n, 3)


def load_dwpose(path, W, H):
    """Return list of dicts with body (18,3), lhand/rhand (21,3), face (70,3) in pixels of the video."""
    data = json.load(open(path))
    out = []
    for fr in data:
        sx, sy = W / fr["canvas_width"], H / fr["canvas_height"]
        ppl = fr.get("people") or []
        if not ppl:
            out.append(None)
            continue
        # keep the biggest person
        def area(p):
            b = _kp(p.get("pose_keypoints_2d"), 18)
            b = b[b[:, 2] > 0.1]
            return 0 if len(b) < 2 else np.ptp(b[:, 0]) * np.ptp(b[:, 1])
        p = max(ppl, key=area)
        d = {"body": _kp(p.get("pose_keypoints_2d"), 18),
             "lhand": _kp(p.get("hand_left_keypoints_2d"), 21),
             "rhand": _kp(p.get("hand_right_keypoints_2d"), 21),
             "face": _kp(p.get("face_keypoints_2d"), 70)}
        for k in d:
            # Normalised coordinates (<=1) are scaled to the canvas first.
            xy = d[k][:, :2]
            if xy.size and xy.max() <= 1.0 + 1e-6:
                xy *= [fr["canvas_width"], fr["canvas_height"]]
            xy *= [sx, sy]
        out.append(d)
    return out


# ----------------------------------------------------------------------------- crops
def square_crop(img, cx, cy, size, out_size):
    """Crop a square (padding with edge colour) and resize. Returns crop and pixel->crop scale."""
    H, W = img.shape[:2]
    half = size / 2
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    s = int(round(size))
    pad = max(0, -x0, -y0, x0 + s - W, y0 + s - H)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0 += pad
        y0 += pad
    crop = img[y0:y0 + s, x0:x0 + s]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC)


def hand_region(dw, side):
    """(cx, cy, size) for the hand on `side` ('l'/'r') from DWPose, or None."""
    body = dw["body"]
    wi, ei = (BODY_LWRIST, BODY_LELBOW) if side == "l" else (BODY_RWRIST, BODY_RELBOW)
    wrist = body[wi]
    # choose whichever DWPose hand array sits closest to this body wrist
    cands = []
    for key in ("lhand", "rhand"):
        h = dw[key]
        good = h[:, 2] > 0.3
        if good.sum() >= 5:
            cands.append((np.linalg.norm(h[good, :2].mean(0) - wrist[:2]) if wrist[2] > 0.1 else 0, h, good))
    hand = None
    if cands:
        dist, h, good = min(cands, key=lambda c: c[0])
        if wrist[2] <= 0.1 or dist < 120:
            hand = (h, good)
    if hand is not None:
        h, good = hand
        pts = h[good, :2]
        c = 0.5 * (pts.min(0) + pts.max(0))
        ext = np.ptp(pts, axis=0).max()
        forearm = np.linalg.norm(body[wi, :2] - body[ei, :2]) if body[ei, 2] > 0.1 and body[wi, 2] > 0.1 else ext
        return c[0], c[1], max(2.0 * ext, 0.9 * forearm, 48), h, good
    if wrist[2] > 0.1 and body[ei, 2] > 0.1:
        d = wrist[:2] - body[ei, :2]
        c = wrist[:2] + 0.45 * d
        return c[0], c[1], max(1.1 * np.linalg.norm(d), 48), None, None
    return None


def face_region(dw):
    f = dw["face"]
    good = f[:, 2] > 0.3
    if good.sum() >= 10:
        pts = f[good, :2]
        c = 0.5 * (pts.min(0) + pts.max(0))
        return c[0], c[1], max(1.9 * np.ptp(pts, axis=0).max(), 64)
    b = dw["body"]
    if b[0, 2] > 0.1 and b[1, 2] > 0.1:          # nose / neck fallback
        s = 1.6 * np.linalg.norm(b[0, :2] - b[1, :2])
        return b[0, 0], b[0, 1], max(s, 64)
    return None


# ----------------------------------------------------------------------------- main
def extract(video_path, kps_path, out_path, models_dir, fps=16.0, hand_crop=256, face_crop=384,
            hand_min_cutoff=1.2, hand_beta=0.6, face_min_cutoff=1.8, face_beta=0.4, verbose=True):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    frames = read_video(video_path)
    T = len(frames)
    H, W = frames[0].shape[:2]
    dws = load_dwpose(kps_path, W, H)
    if len(dws) != T:
        print(f"[extract] WARNING: {T} video frames vs {len(dws)} keypoint frames; using min")
        T = min(T, len(dws))

    hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models_dir, "hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
    face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(models_dir, "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, output_face_blendshapes=True, num_faces=1,
        min_face_detection_confidence=0.3))

    angles = np.zeros((T, 2, 5, 4))
    source = np.zeros((T, 2), np.int8)
    face = np.zeros((T, len(FACE_CHANNELS)))
    face_ok = np.zeros(T, bool)

    for t in range(T):
        img, dw = frames[t], dws[t]
        if dw is None:
            continue
        for hi, side in enumerate("lr"):
            reg = hand_region(dw, side)
            if reg is None:
                continue
            cx, cy, size, dwh, dwgood = reg
            crop = square_crop(img, cx, cy, size, hand_crop)
            res = hand_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
            best = None
            for lm2d, lmw in zip(res.hand_landmarks, res.hand_world_landmarks):
                wx, wy = lm2d[0].x - 0.5, lm2d[0].y - 0.5
                c = np.array([[l.x, l.y] for l in lm2d]).mean(0) - 0.5
                d = np.hypot(*c)                       # prefer the hand nearest the crop centre
                if best is None or d < best[0]:
                    best = (d, lmw)
            if best is not None and best[0] < 0.35:
                angles[t, hi] = finger_angles(np.array([[l.x, l.y, l.z] for l in best[1]]))
                source[t, hi] = 1
            elif dwh is not None and dwgood.sum() >= 18:
                angles[t, hi] = finger_angles(dwh[:, :2])
                source[t, hi] = 2

        reg = face_region(dw)
        if reg is not None:
            crop = square_crop(img, reg[0], reg[1], reg[2], face_crop)
            res = face_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
            if res.face_blendshapes:
                bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
                face[t] = [np.mean([bs.get(s, 0.0) for s in FACE_SOURCES[ch]]) for ch in FACE_CHANNELS]
                face_ok[t] = True

    hand_lm.close()
    face_lm.close()

    valid = source > 0
    for hi in range(2):
        angles[:, hi] = fill_gaps(angles[:, hi], valid[:, hi])
        if valid[:, hi].any():
            angles[:, hi] = one_euro_sequence(angles[:, hi], fps, hand_min_cutoff, hand_beta)
    face = fill_gaps(face, face_ok)
    if face_ok.any():
        face = one_euro_sequence(face, fps, face_min_cutoff, face_beta)

    np.savez(out_path, finger_angles=angles.astype(np.float32), hand_valid=valid, hand_source=source,
             face=face.astype(np.float32), face_valid=face_ok, face_channels=np.array(FACE_CHANNELS), fps=fps)
    if verbose:
        for hi, name in enumerate(("left", "right")):
            print(f"[extract] {name} hand: mediapipe {np.sum(source[:, hi] == 1)}/{T}, "
                  f"dwpose {np.sum(source[:, hi] == 2)}/{T}")
        print(f"[extract] face: {face_ok.sum()}/{T} frames")
        print(f"[extract] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--kps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-dir", required=True, help="folder with hand_landmarker.task / face_landmarker.task")
    ap.add_argument("--fps", type=float, default=16.0)
    a = ap.parse_args()
    extract(a.video, a.kps, a.out, a.models_dir, a.fps)
