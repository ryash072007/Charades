"""Pull GVHMR's arms toward where the video's 2D keypoints put the hands.

GVHMR's 3D arms drift a few centimetres from the video, which matters in charades: a hand that was at the
mouth must stay at the mouth. For every frame we take the DWPose wrist position relative to the face (when the
hand is near the face) or to the shoulder (otherwise), map it into the character's front view at the same body
scale, and move the wrist there with two-bone IK. Depth comes from GVHMR, the elbow keeps its bend direction and
the hand keeps its orientation. Corrections are confidence weighted, smoothed over time and capped.

Works on motion.npz from prepare_motion.py (character faces +Z, Y up) in place.
"""
import json

import numpy as np

from body_dims import betas_from_joints, measure
from prepare_motion import PARENTS, fk

BODY = {"nose": 0, "rsh": 2, "lsh": 5, "rwr": 4, "lwr": 7}
ARMS = ((7, 16, 18, 20), (4, 17, 19, 21))  # (DWPose wrist, SMPL shoulder, elbow, wrist): left, right


def _rot_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.9999:  # opposite: any perpendicular axis
        p = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _smooth(x, w, sigma=1.6):
    """Confidence-weighted Gaussian smoothing over time (x: (T, k), w: (T,))."""
    T = len(x)
    t = np.arange(T)
    K = np.exp(-0.5 * ((t[:, None] - t[None, :]) / sigma) ** 2) * w[None, :]
    s = K.sum(1, keepdims=True)
    return np.where(s > 1e-6, (K @ x) / np.maximum(s, 1e-6), 0.0)


def refine(motion_npz, kps_json, smpl_model, max_shift=0.6, verbose=True):
    d = dict(np.load(motion_npz))
    J = d["rest_joints"].astype(np.float64)
    local = d["local_rot"].astype(np.float64)
    root = d["root_pos"].astype(np.float64)
    T = len(local)
    betas = d["betas"] if "betas" in d else betas_from_joints(smpl_model, J)
    head = measure(smpl_model, betas)["head"]
    G, P = fk(J, local, root)
    mouth3 = P[:, 15] + np.einsum("tij,j->ti", G[:, 15], [0, head["mouth_y"], head["nose"][2] - 0.01])
    nose3 = P[:, 15] + np.einsum("tij,j->ti", G[:, 15], head["nose"])
    sh3 = (P[:, 16] + P[:, 17]) / 2

    kps = json.load(open(kps_json))
    deltas = np.zeros((2, T, 2))
    conf = np.zeros((2, T))
    for f in range(min(T, len(kps))):
        people = kps[f].get("people") or []
        if not people:
            continue
        p = people[0]
        b = np.asarray(p["pose_keypoints_2d"], dtype=np.float64).reshape(-1, 3)
        fc = np.asarray(p.get("face_keypoints_2d") or [0] * 210, dtype=np.float64).reshape(-1, 3)
        if min(b[0, 2], b[2, 2], b[5, 2]) < 0.3:
            continue
        sh2 = (b[2, :2] + b[5, :2]) / 2
        sc2 = np.linalg.norm(b[0, :2] - sh2)
        sc3 = np.linalg.norm((nose3[f] - sh3[f])[:2])
        if sc2 < 1 or sc3 < 1e-3:
            continue
        k = sc3 / sc2
        mouth_ok = fc[48:68, 2].mean() > 0.3
        m2 = fc[48:68, :2].mean(0) if mouth_ok else b[0, :2] + (sh2 - b[0, :2]) * 0.25
        for h, (wk, si, ei, wi) in enumerate(ARMS):
            if b[wk, 2] < 0.3:
                continue
            s2 = b[2 if si == 17 else 5, :2]  # same-side shoulder (DWPose 2 = right, 5 = left)
            to_face = np.linalg.norm(b[wk, :2] - m2) / sc2
            a = 1 - np.clip((to_face - 0.9) / 1.1, 0, 1)  # 1 near the face, 0 far from it
            via_face = mouth3[f, :2] + (b[wk, :2] - m2) * [1, -1] * k
            via_sh = P[f, si, :2] + (b[wk, :2] - s2) * [1, -1] * k
            target = a * via_face + (1 - a) * via_sh
            dlt = target - P[f, wi, :2]
            cap = max_shift * sc3
            n = np.linalg.norm(dlt)
            if n > cap:
                dlt *= cap / n
            deltas[h, f] = dlt
            conf[h, f] = b[wk, 2]

    newG = G.copy()
    for h, (_, si, ei, wi) in enumerate(ARMS):
        dsm = _smooth(deltas[h], conf[h])
        for t in range(T):
            if not np.any(dsm[t]):
                continue
            S, E, W = P[t, si], P[t, ei], P[t, wi]
            tgt = W + np.array([dsm[t, 0], dsm[t, 1], 0.0])
            la, lb = np.linalg.norm(J[ei] - J[si]), np.linalg.norm(J[wi] - J[ei])
            dv = tgt - S
            dist = np.clip(np.linalg.norm(dv), abs(la - lb) + 1e-3, la + lb - 1e-3)
            u = dv / np.linalg.norm(dv)
            x = (la * la - lb * lb + dist * dist) / (2 * dist)
            hgt = np.sqrt(max(la * la - x * x, 0.0))
            pole = (E - S) - u * np.dot(E - S, u)
            pole = pole / np.linalg.norm(pole) if np.linalg.norm(pole) > 1e-6 else np.cross(u, [0, 0, 1.0])
            E2 = S + u * x + pole * hgt
            W2 = S + u * dist
            Gs = _rot_between(E - S, E2 - S) @ G[t, si]
            Ge_tmp = Gs @ local[t, ei]
            cur = Ge_tmp @ (J[wi] - J[ei])
            Ge = _rot_between(cur, W2 - E2) @ Ge_tmp
            parent_s = newG[t, PARENTS[si]]
            local[t, si] = parent_s.T @ Gs
            local[t, ei] = Gs.T @ Ge
            local[t, wi] = Ge.T @ G[t, wi]  # the hand keeps its orientation
            newG[t, si], newG[t, ei], newG[t, wi] = Gs, Ge, G[t, wi]
    d["local_rot"] = local.astype(np.float32)
    d["refined"] = np.array(1)
    _, P2 = fk(J, local, root)
    d["joints"] = P2.astype(np.float32)
    np.savez(motion_npz, **d)
    if verbose:
        moved = [np.linalg.norm(P2[:, wi] - P[:, wi], axis=1) for *_, wi in ARMS]
        print(f"[refine] wrists moved mean {np.mean(moved) * 100:.1f} cm, max {np.max(moved) * 100:.1f} cm "
              f"(frames with 2D: L {int((conf[0] > 0).sum())}, R {int((conf[1] > 0).sum())} of {T})")
    return motion_npz
