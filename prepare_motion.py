"""SMPL forward kinematics on GVHMR output -> Blender-ready motion file.

Output npz (all in SMPL coordinates: Y up, character faces +Z):
    rest_joints   (22, 3)       SMPL rest joints for the clip's betas
    parents       (22,)
    local_rot     (T, 22, 3, 3) per-joint local rotations (joint 0 includes facing yaw)
    root_pos      (T, 3)        pelvis joint position (centred, feet grounded)
    joints        (T, 22, 3)    posed world joints (for camera framing)
    + finger_angles / face arrays copied from the hands/face npz
    fps

Run with ComfyUI's embedded Python.
"""
import numpy as np
from scipy.spatial.transform import Rotation

PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])

# Radii of the procedural character's feet (must match blender_character.py).
ANKLE_RADIUS = 0.055
TOE_RADIUS = 0.045


def rest_joints(smpl_model_path, betas):
    m = np.load(smpl_model_path, allow_pickle=True)
    v = m["v_template"] + np.einsum("vci,i->vc", m["shapedirs"][:, :, :len(betas)], betas)
    jr = m["J_regressor"]
    jr = jr.toarray() if hasattr(jr, "toarray") else jr
    return (jr @ v)[:22]


def fk(J, local_rot, root_pos):
    T = len(local_rot)
    G = np.zeros((T, 22, 3, 3))
    P = np.zeros((T, 22, 3))
    G[:, 0] = local_rot[:, 0]
    P[:, 0] = root_pos
    for i in range(1, 22):
        p = PARENTS[i]
        G[:, i] = G[:, p] @ local_rot[:, i]
        P[:, i] = P[:, p] + np.einsum("tij,j->ti", G[:, p], J[i] - J[p])
    return G, P


def _smooth(x, k=5):
    if len(x) < k:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def prepare(gvhmr_npz, hands_face_npz, smpl_model_path, out_path, ground="per_frame", face_camera=True):
    d = np.load(gvhmr_npz)
    T = len(d["body_pose"])
    betas = d["betas"].mean(0)
    J = rest_joints(smpl_model_path, betas)

    aa = np.concatenate([d["global_orient"][:, None], d["body_pose"].reshape(T, 21, 3)], 1)
    local = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 22, 3, 3)
    root = J[0] + d["transl"]

    # Turn the clip so the character's average facing points at +Z (towards the camera).
    if face_camera:
        fwd = local[:, 0] @ np.array([0.0, 0.0, 1.0])
        f = fwd[:, [0, 2]].mean(0)
        yaw = np.arctan2(f[0], f[1])
        Ry = Rotation.from_rotvec([0, -yaw, 0]).as_matrix()
        local[:, 0] = Ry @ local[:, 0]
        root = root @ Ry.T

    # Centre horizontally on the clip's mean pelvis position.
    root[:, [0, 2]] -= root[:, [0, 2]].mean(0)

    # Ground: lowest foot surface touches y = 0.
    G, P = fk(J, local, root)
    toe_tip = P[:, [10, 11]] + np.einsum("tkij,j->tki", G[:, [10, 11]], [0, 0, 0.035])
    low = np.minimum.reduce([
        P[:, 7, 1] - ANKLE_RADIUS, P[:, 8, 1] - ANKLE_RADIUS,
        P[:, 10, 1] - TOE_RADIUS, P[:, 11, 1] - TOE_RADIUS,
        toe_tip[:, 0, 1] - TOE_RADIUS + 0.005, toe_tip[:, 1, 1] - TOE_RADIUS + 0.005,
    ])
    if ground == "per_frame":
        offset = _smooth(low)          # keeps at least one foot planted every frame
    else:
        offset = np.full(T, np.percentile(low, 5))
    root[:, 1] -= offset
    G, P = fk(J, local, root)

    hf = np.load(hands_face_npz)
    np.savez(out_path, rest_joints=J.astype(np.float32), parents=PARENTS, local_rot=local.astype(np.float32), betas=betas,
             root_pos=root.astype(np.float32), joints=P.astype(np.float32),
             finger_angles=hf["finger_angles"][:T], hand_valid=hf["hand_valid"][:T],
             face=hf["face"][:T], face_channels=hf["face_channels"], fps=float(hf["fps"]))
    print(f"[prepare] {T} frames, height {P[:, 15, 1].max() + 0.22:.2f} m, wrote {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gvhmr", required=True)
    ap.add_argument("--hands-face", required=True)
    ap.add_argument("--smpl", required=True, help="SMPL_NEUTRAL.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ground", choices=["per_frame", "global"], default="per_frame")
    a = ap.parse_args()
    prepare(a.gvhmr, a.hands_face, a.smpl, a.out, a.ground)
