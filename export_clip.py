"""motion.npz (prepare_motion.py) -> ai-dumb-charades clip files.

<id>.bin      int16 millimetres, (frames, 24, 3), 30 fps, y-up, faces +z at frame 0, root at origin
              (HumanML3D 22 joints + 22/23 hand tips) - the site's existing clip format.
<id>.w2m.bin  int16 (frames, 59) sidecar, values x1000, same frames:
                [0:40]  finger angles, hand (L, R) x finger (thumb..pinky) x [curl1, curl2, curl3, spread], radians
                [40:47] face: jawOpen, mouthSmile, mouthFrown, browInnerUp, browDown, eyeBlink, eyeWide (0..1)
                [47:59] hand frames, unit vectors in clip space: L dir, L thumb-side, R dir, R thumb-side
"""
import numpy as np

from prepare_motion import fk

FPS_OUT = 30
SIDECAR_COLS = 59
HAND_LEN = 0.085  # wrist -> hand tip (site joints 22/23)


def site_normalize(P, extra_vecs=()):
    """Same steps as ai-dumb-charades mocap/build_library.py normalize(); also rotates direction vectors."""
    leg = np.linalg.norm(P[:, 1] - P[:, 4], axis=1) + np.linalg.norm(P[:, 4] - P[:, 7], axis=1)
    s = 0.86 / max(np.median(leg), 1e-6)
    P = P * s
    feet = P[:, [7, 8, 10, 11], 1]
    P[:, :, 1] -= np.percentile(feet.min(axis=1), 3)
    across = (P[0, 1] - P[0, 2]) + (P[0, 16] - P[0, 17])
    across[1] = 0
    across /= np.linalg.norm(across) + 1e-9
    fwd = np.cross(across, [0, 1, 0])
    ang = np.arctan2(fwd[0], fwd[2])
    c, sn = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
    P = P @ R.T
    P[:, :, [0, 2]] -= P[0, 0, [0, 2]]
    return P, [v @ R.T for v in extra_vecs]


def resample(x, fps_in, fps_out):
    T = len(x)
    t_out = np.arange(0, (T - 1) / fps_in + 1e-9, 1 / fps_out)
    f = t_out * fps_in
    i = np.minimum(np.floor(f).astype(int), T - 1)
    j = np.minimum(i + 1, T - 1)
    k = (f - i).reshape(-1, *([1] * (x.ndim - 1)))
    return x[i] * (1 - k) + x[j] * k


def export(motion_npz, out_bin, out_sidecar):
    d = np.load(motion_npz)
    J, local, root = d["rest_joints"], d["local_rot"].astype(np.float64), d["root_pos"].astype(np.float64)
    fps = float(d["fps"])
    G, P = fk(J, local, root)

    # hand direction / thumb side in the SMPL rest pose (T-pose, palms down, thumbs forward)
    vecs = []
    tips = []
    for wi, ei in ((20, 18), (21, 19)):
        rest_dir = J[wi] - J[ei]
        rest_dir[1] = 0
        rest_dir /= np.linalg.norm(rest_dir)
        hand_dir = np.einsum("tij,j->ti", G[:, wi], rest_dir)
        thumb = np.einsum("tij,j->ti", G[:, wi], [0.0, 0.0, 1.0])
        tips.append(P[:, wi] + hand_dir * HAND_LEN)
        vecs += [hand_dir, thumb]
    P24 = np.concatenate([P, np.stack(tips, 1)], 1)
    P24, vecs = site_normalize(P24, vecs)

    side = np.concatenate([d["finger_angles"].reshape(len(P), 40), d["face"].reshape(len(P), 7),
                           np.concatenate(vecs, 1)], 1)
    P30 = resample(P24, fps, FPS_OUT)
    side30 = resample(side, fps, FPS_OUT)
    for k in range(4):  # re-unit the interpolated direction vectors
        v = side30[:, 47 + 3 * k:50 + 3 * k]
        side30[:, 47 + 3 * k:50 + 3 * k] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    q = np.clip(np.round(P30 * 1000), -32768, 32767).astype("<i2")
    open(out_bin, "wb").write(q.tobytes())
    sq = np.clip(np.round(side30 * 1000), -32768, 32767).astype("<i2")
    open(out_sidecar, "wb").write(sq.tobytes())
    return {"frames": int(len(P30)), "seconds": round((len(P30) - 1) / FPS_OUT, 2)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("out_bin")
    ap.add_argument("out_sidecar")
    a = ap.parse_args()
    print(export(a.motion, a.out_bin, a.out_sidecar))
