"""Body measurements from the SMPL model with a clip's betas, for drawing characters that match the captured body.

Everything is in SMPL rest-pose coordinates (T-pose, Y up, facing +Z, metres) and relative to the SMPL joints,
so a character built on the same skeleton puts its surfaces where the fitted body had them: a hand that touched
the mouth in the video touches the character's mouth. Only these numbers leave the machine, never the SMPL mesh.
"""
import numpy as np

# (name, joint, child joint): limb segments measured as tapered tubes along the bone
SEGMENTS = [("thigh_l", 1, 4), ("thigh_r", 2, 5), ("shin_l", 4, 7), ("shin_r", 5, 8),
            ("upperarm_l", 16, 18), ("upperarm_r", 17, 19), ("forearm_l", 18, 20), ("forearm_r", 19, 21),
            ("neck", 12, 15), ("collar_l", 13, 16), ("collar_r", 14, 17)]
NOSE_VERTEX = 332  # SMPL nose tip


def smpl_body(model_path, betas):
    m = np.load(model_path, allow_pickle=True)
    nb = len(betas)
    v = m["v_template"] + np.einsum("vci,i->vc", m["shapedirs"][:, :, :nb], betas)
    jr = m["J_regressor"]
    jr = jr.toarray() if hasattr(jr, "toarray") else jr
    return v, jr @ v, m["weights"].argmax(1)


def betas_from_joints(model_path, rest_joints, nb=10):
    """Recover betas from rest joints (they're linear in betas) for motion files saved without them."""
    m = np.load(model_path, allow_pickle=True)
    jr = m["J_regressor"]
    jr = jr.toarray() if hasattr(jr, "toarray") else jr
    j0 = (jr @ m["v_template"])[:22]
    A = np.einsum("jv,vci->jci", jr[:22], m["shapedirs"][:, :, :nb]).reshape(-1, nb)
    b, *_ = np.linalg.lstsq(A, (rest_joints - j0).reshape(-1), rcond=None)
    return b


def _tube(v, J, a, b):
    """Median surface radius near each end of the bone a->b (vertices owned by joint a)."""
    d = J[b] - J[a]
    L = np.linalg.norm(d)
    u = d / L
    p = v - J[a]
    t = p @ u / L
    r = np.linalg.norm(p - np.outer(p @ u, u), axis=1)
    keep = (t > -0.05) & (t < 1.05)
    t, r = t[keep], r[keep]
    near = r[(t >= 0.05) & (t <= 0.3)]
    far = r[(t >= 0.7) & (t <= 0.95)]
    return float(np.median(near) if len(near) else np.median(r)), float(np.median(far) if len(far) else np.median(r))


def measure(model_path, betas):
    v, J, owner = smpl_body(model_path, betas)
    out = {"segments": {}}
    for name, a, b in SEGMENTS:
        out["segments"][name] = _tube(v[owner == a], J, a, b)

    # torso: cross-section profile along the spine (pelvis/spine/collar vertices, arms excluded)
    tv = v[np.isin(owner, [0, 3, 6, 9, 13, 14])]
    tv = tv[np.abs(tv[:, 0]) < abs(J[16, 0]) + 0.02]
    y0, y1 = np.percentile(tv[:, 1], 0.5), J[12, 1]
    ys = np.linspace(y0, y1, 14)
    prof = []
    for i in range(len(ys)):
        lo, hi = ys[max(i - 1, 0)], ys[min(i + 1, len(ys) - 1)]
        s = tv[(tv[:, 1] >= lo) & (tv[:, 1] <= hi)]
        if len(s) < 8:
            continue
        prof.append([float(ys[i]), float(np.percentile(np.abs(s[:, 0]), 97)),
                     float(np.percentile(s[:, 2], 97)), float(np.percentile(s[:, 2], 3))])
    out["torso"] = prof  # [y, half-width, front z, back z]

    # head: ellipsoid around the head vertices, plus where the nose / eyes / mouth sit
    hv = v[owner == 15]
    lo, hi = np.percentile(hv, 1, axis=0), np.percentile(hv, 99, axis=0)
    c = (lo + hi) / 2
    nose = v[NOSE_VERTEX]
    out["head"] = {"center": (c - J[15]).tolist(), "radii": ((hi - lo) / 2).tolist(),
                   "nose": (nose - J[15]).tolist(), "eyes_y": float(nose[1] + 0.035 - J[15, 1]),
                   "mouth_y": float(nose[1] - 0.04 - J[15, 1])}

    # hands (left, mirrored for the right): length along the forearm direction, width, thickness
    hv = v[np.isin(owner, [20, 22])]
    d = J[20] - J[18]
    d[1] = 0
    d /= np.linalg.norm(d)
    along = (hv - J[20]) @ d
    out["hand"] = {"length": float(np.percentile(along, 99.5)), "palm": float(np.linalg.norm(J[22] - J[20]) * 1.05),
                   "width": float(np.ptp(np.percentile(hv[:, 2], [3, 97]))),
                   "thickness": float(np.ptp(np.percentile(hv[:, 1], [3, 97])))}

    # feet: sole height below the ankle, heel / toe reach, width
    fv = v[np.isin(owner, [7, 10])]
    out["foot"] = {"sole": float(fv[:, 1].min() - J[7, 1]), "heel": float(fv[:, 2].min() - J[7, 2]),
                   "toe": float(fv[:, 2].max() - J[7, 2]), "width": float(np.ptp(np.percentile(fv[:, 0], [3, 97])))}
    out["height"] = float(v[:, 1].max() - v[:, 1].min())
    return out


def scaled(d, s):
    """Multiply every length in a measure() result by s."""
    if isinstance(d, dict):
        return {k: scaled(x, s) for k, x in d.items()}
    if isinstance(d, (list, tuple)):
        return [scaled(x, s) for x in d]
    return round(d * s, 5) if isinstance(d, float) else d
