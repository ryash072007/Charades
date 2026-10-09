"""motion.npz (prepare_motion.py) -> ai-dumb-charades clip files.

<id>.bin      int16 millimetres, (frames, 24, 3), 30 fps, y-up, faces +z at frame 0, root at origin
              (HumanML3D 22 joints + 22/23 hand tips) - the site's existing clip format.
<id>.w2m.bin  int16 (frames, 59) sidecar, values x1000, same frames:
                [0:40]  finger angles, hand (L, R) x finger (thumb..pinky) x [curl1, curl2, curl3, spread], radians
                [40:47] face: jawOpen, mouthSmile, mouthFrown, browInnerUp, browDown, eyeBlink, eyeWide (0..1)
                [47:59] hand frames, unit vectors in clip space: L dir, L thumb-side, R dir, R thumb-side
"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from body_dims import betas_from_joints, measure, scaled
from prepare_motion import PARENTS, fk

FPS_OUT = 30
SIDECAR_COLS = 59
HAND_LEN = 0.085  # wrist -> hand tip (site joints 22/23)


def normalize_transform(P):
    """Scale / floor / yaw / shift of ai-dumb-charades mocap/build_library.py normalize(), as parameters."""
    leg = np.linalg.norm(P[:, 1] - P[:, 4], axis=1) + np.linalg.norm(P[:, 4] - P[:, 7], axis=1)
    s = 0.86 / max(np.median(leg), 1e-6)
    Q = P * s
    floor = np.percentile(Q[:, [7, 8, 10, 11], 1].min(axis=1), 3)
    across = (Q[0, 1] - Q[0, 2]) + (Q[0, 16] - Q[0, 17])
    across[1] = 0
    across /= np.linalg.norm(across) + 1e-9
    fwd = np.cross(across, [0, 1, 0])
    ang = np.arctan2(fwd[0], fwd[2])
    c, sn = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
    shift = ((Q[0, 0] - [0, floor, 0]) @ R.T) * [1, 0, 1]
    return s, floor, R, shift


def apply_transform(P, tf):
    s, floor, R, shift = tf
    return (P * s - [0, floor, 0]) @ R.T - shift


def site_normalize(P, extra_vecs=()):
    """Same result as ai-dumb-charades mocap/build_library.py normalize(); also rotates direction vectors."""
    tf = normalize_transform(P)
    return apply_transform(P, tf), [v @ tf[2].T for v in extra_vecs]


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


def export_rig(motion_npz, out_json, smpl_model):
    """<id>.rig.json: the clip's SMPL skeleton + per-frame local rotations for skinned characters (30 fps,
    same normalisation as the .bin), plus fingers, face and body measurements (body_dims.py)."""
    d = np.load(motion_npz)
    J, local, root = d["rest_joints"].astype(np.float64), d["local_rot"].astype(np.float64), d["root_pos"].astype(np.float64)
    fps = float(d["fps"])
    T = len(local)
    _, P = fk(J, local, root)
    tf = normalize_transform(P)
    s, _, R, _ = tf
    local = local.copy()
    local[:, 0] = R @ local[:, 0]
    root_n = apply_transform(root[:, None], tf)[:, 0]

    t_in = np.arange(T) / fps
    t_out = np.arange(0, t_in[-1] + 1e-9, 1 / FPS_OUT)
    quats = np.stack([Slerp(t_in, Rotation.from_matrix(local[:, j]))(t_out).as_quat() for j in range(22)], 1)  # x y z w
    for j in range(22):  # keep neighbouring quaternions in the same hemisphere for smooth blending
        q = quats[:, j]
        flip = np.cumsum(np.r_[0, (np.sum(q[1:] * q[:-1], 1) < 0)]) % 2 == 1
        q[flip] *= -1
    root30 = resample(root_n, fps, FPS_OUT)
    fingers = resample(d["finger_angles"].reshape(T, 40), fps, FPS_OUT)
    face = resample(d["face"].reshape(T, 7), fps, FPS_OUT)

    betas = d["betas"] if "betas" in d.files else betas_from_joints(smpl_model, J)
    dims = scaled(measure(smpl_model, betas), s)
    r4 = lambda a: np.round(a, 4).tolist()
    # the site floors the toe joints; how far to lift a character so its measured soles touch the floor instead
    G30, P30 = fk(J * s, Rotation.from_quat(quats.reshape(-1, 4)).as_matrix().reshape(len(t_out), 22, 3, 3), root30)
    sole = np.einsum("tkij,j->tki", G30[:, [7, 8]], [0.0, dims["foot"]["sole"], 0.0])[..., 1] + P30[:, [7, 8], 1]
    lift = float(-np.percentile(sole.min(1), 3))
    rig_extra = {}
    if len(np.load(smpl_model)["J_regressor"]) >= 55:  # SMPL-X: also the exact fitted mesh for the "Exact fit" character
        import smplx_export
        smplx_export.ensure_template(smpl_model, os.path.dirname(os.path.abspath(out_json)))
        rot30 = Rotation.from_quat(quats.reshape(-1, 4)).as_matrix().reshape(len(t_out), 22, 3, 3)
        rig_extra["smplx"] = smplx_export.export(smpl_model, betas, s, rot30, root30, out_json.replace(".rig.json", ".smplx.bin"))
    rig = {"version": 1, "fps": FPS_OUT, "frames": int(len(t_out)), "parents": PARENTS.tolist(), "rest": r4(J * s), "lift": round(lift, 4),
           "root": r4(root30), "quat": r4(quats), "fingers": r4(fingers), "face": r4(face), "dims": dims, **rig_extra}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rig, f, separators=(",", ":"))
    return {"frames": rig["frames"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("out_bin")
    ap.add_argument("out_sidecar")
    a = ap.parse_args()
    print(export(a.motion, a.out_bin, a.out_sidecar))
