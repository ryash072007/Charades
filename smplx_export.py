"""SMPL-X mesh data for the site's "Exact fit" character.

GVHMR fits SMPL-X, so the SMPL-X body with a clip's betas is exactly the body that was fitted to the video.
smplx-template.bin (written once into the clips folder, shared by every clip), little-endian:
    uint32 nv, uint32 nf
    uint16 faces[nf*3]
    uint8  skin_index[nv*4], uint8 skin_weight[nv*4] (top 4 joints per vertex, weights /255)
    uint8  part[nv]   0 skin, 1 eyes, 2 top, 3 bottom, 4 shoes, 5 hair (for colouring)
    uint8  parents[55] (255 = root)
<id>.smplx.bin: float32 rest vertices (nv*3) for that clip's betas, scaled like the clip.
"""
import os
import struct

import numpy as np

PARTS = {1: [23, 24], 2: [3, 6, 9, 13, 14, 16, 17], 3: [0, 1, 2, 4, 5], 4: [7, 8, 10, 11]}  # 5 = hair (scalp)


def load(model_path):
    return np.load(model_path, allow_pickle=True)


def write_template(model_path, out_path):
    m = load(model_path)
    W = m["weights"]
    nv, nf = len(W), len(m["f"])
    idx = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, idx, 1)
    w = w / w.sum(1, keepdims=True)
    wq = np.round(w * 255).astype(np.int32)
    wq[:, 0] += 255 - wq.sum(1)  # exact 255 total so weights sum to 1
    owner = W.argmax(1)
    part = np.zeros(nv, np.uint8)
    for code, joints in PARTS.items():
        part[np.isin(owner, joints)] = code
    # hair: the scalp of the head (above the brow, and the back of the head down to the ears)
    v = m["v_template"]
    head = np.isin(owner, [15, 22])
    hv = v[head]
    front = hv[:, 2].max()
    top = hv[:, 1].max()
    brow = top - 0.075
    scalp = head & ((v[:, 1] > brow) & (v[:, 2] < front - 0.03) | (v[:, 1] > brow - 0.06) & (v[:, 2] < front - 0.1))
    part[scalp] = 5
    parents = m["kintree_table"][0].astype(np.int64)
    parents = np.where(parents > 1000, 255, parents).astype(np.uint8)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", nv, nf))
        f.write(m["f"].astype("<u2").tobytes())
        f.write(idx.astype(np.uint8).tobytes())
        f.write(np.clip(wq, 0, 255).astype(np.uint8).tobytes())
        f.write(part.tobytes())
        f.write(parents.tobytes())
    return out_path


def ensure_template(model_path, clips_dir):
    path = os.path.join(clips_dir, "smplx-template.bin")
    if not os.path.isfile(path):
        write_template(model_path, path)
    return path


def body(model_path, betas):
    m = load(model_path)
    v = m["v_template"] + np.einsum("vci,i->vc", m["shapedirs"][:, :, :len(betas)], betas)
    return m, v, m["J_regressor"] @ v


def fk55(J, parents, rot22, root):
    """World rotations/positions for all 55 joints; joints past 21 keep their rest orientation."""
    T = len(rot22)
    G = np.tile(np.eye(3), (T, len(J), 1, 1))
    P = np.zeros((T, len(J), 3))
    G[:, 0] = rot22[:, 0]
    P[:, 0] = root
    for i in range(1, len(J)):
        p = parents[i]
        loc = rot22[:, i] if i < 22 else np.eye(3)
        G[:, i] = G[:, p] @ loc
        P[:, i] = P[:, p] + np.einsum("tij,j->ti", G[:, p], J[i] - J[p])
    return G, P


def export(model_path, betas, scale, rot22, root, out_bin):
    """Write <id>.smplx.bin and return the rig's "smplx" section (joints, parents, floor lift)."""
    m, v, J = body(model_path, betas)
    v, J = v * scale, J * scale
    parents = m["kintree_table"][0].astype(np.int64)
    parents[0] = -1
    v.astype("<f4").tofile(out_bin)
    # floor: lowest foot vertex over the clip (skinned with linear blend skinning)
    W = m["weights"]
    feet = np.isin(W.argmax(1), [7, 8, 10, 11])
    G, P = fk55(J, parents, rot22[::2], root[::2])
    A = np.einsum("tjab,jb->tja", G, -J) + P  # translation part of each joint's skinning transform
    Wf, vf = W[feet], v[feet]
    y = np.einsum("vj,tjb,vb->tv", Wf, G[:, :, 1, :], vf) + (Wf @ A[:, :, 1].T).T
    lift = float(-np.percentile(y.min(1), 3))
    return {"joints": np.round(J, 4).tolist(), "parents": parents.tolist(), "lift": round(lift, 4)}
