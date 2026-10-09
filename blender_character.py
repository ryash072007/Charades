"""Build a procedural soft-vinyl character from motion.npz, animate it, render and export GLB.

    blender -b --factory-startup --python blender_character.py -- \
        --motion motion.npz --frames-dir out/frames --glb out/anim.glb [--still 24 --still-out x.png]

Everything is built in SMPL coordinates (Y up, facing +Z) inside an armature whose
object transform rotates it to Blender's Z-up, facing -Y (towards the camera).
Body parts are rigidly skinned to the SMPL bones (torso is blended over the spine),
fingers have 3 bones each, and the face is one mesh with shape keys.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

SMPL_NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
              "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
              "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# Tail target for each SMPL bone (joint index, or None for a hand-made offset).
BONE_TAIL = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 12: 15,
             13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21}

# Character proportions (metres). ANKLE/TOE radii must match prepare_motion.py.
R = dict(thigh=(0.085, 0.066), shin=(0.064, 0.052), foot=(0.055, 0.045), toe=(0.045, 0.040),
         shoulder=(0.064, 0.060), upper_arm=(0.062, 0.052), forearm=(0.051, 0.042), neck=0.055)
HEAD_CENTER_OFFSET = Vector((0.0, 0.095, 0.012))
HEAD_RADII = Vector((0.122, 0.128, 0.116))

# Face feature exaggeration: (lo, hi) of the raw blendshape mapped to 0..1.
FACE_MAP = {"Open": ("jawOpen", 0.04, 0.30), "Smile": ("mouthSmile", 0.08, 0.45),
            "Frown": ("mouthFrown", 0.05, 0.35), "BrowUp": ("browInnerUp", 0.12, 0.55),
            "BrowDown": ("browDown", 0.05, 0.35), "Blink": ("eyeBlink", 0.35, 0.70),
            "Wide": ("eyeWide", 0.05, 0.35)}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--frames-dir", default="")
    ap.add_argument("--glb", default="")
    ap.add_argument("--blend", default="")
    ap.add_argument("--still", type=int, default=-1, help="render only this frame (0-based)")
    ap.add_argument("--still-out", default="")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=1248)
    ap.add_argument("--color", default="#78cdb3", help="pastel body colour")
    ap.add_argument("--samples", type=int, default=48)
    return ap.parse_args(argv)


def hex_rgb(h):
    h = h.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# =============================================================================== mesh builder
class MeshBuilder:
    """Collect verts/faces with per-vertex bone weights, then emit one skinned mesh."""

    def __init__(self):
        self.verts, self.faces, self.weights = [], [], []   # weights: list of {bone: w}

    def add(self, verts, faces, weights):
        base = len(self.verts)
        self.verts.extend(verts)
        self.faces.extend([tuple(i + base for i in f) for f in faces])
        if isinstance(weights, dict):
            weights = [weights] * len(verts)
        self.weights.extend(weights)
        return base

    def lathe(self, rings, segs=24, weight_fn=None, bone=None):
        """rings: list of (center Vector, axis_u, axis_v, ru, rv); first/last radius 0 => poles."""
        verts, faces, wts, ring_idx = [], [], [], []
        for c, u, v, ru, rv in rings:
            if ru < 1e-6 and rv < 1e-6:
                ring_idx.append([len(verts)])
                verts.append(c.copy())
            else:
                idx = []
                for s in range(segs):
                    a = 2 * math.pi * s / segs
                    idx.append(len(verts))
                    verts.append(c + u * (ru * math.cos(a)) + v * (rv * math.sin(a)))
                ring_idx.append(idx)
        for r0, r1 in zip(ring_idx[:-1], ring_idx[1:]):
            if len(r0) == 1 and len(r1) == 1:
                continue
            if len(r0) == 1:
                faces += [(r0[0], r1[(s + 1) % segs], r1[s]) for s in range(segs)]
            elif len(r1) == 1:
                faces += [(r0[s], r0[(s + 1) % segs], r1[0]) for s in range(segs)]
            else:
                faces += [(r0[s], r0[(s + 1) % segs], r1[(s + 1) % segs], r1[s]) for s in range(segs)]
        wts = [weight_fn(p) for p in verts] if weight_fn else {bone: 1.0}
        return self.add(verts, faces, wts)

    def capsule(self, a, b, ra, rb, bone, segs=24, cap_rings=8):
        a, b = Vector(a), Vector(b)
        w = b - a
        L = w.length
        w.normalize()
        u = w.orthogonal().normalized()
        v = w.cross(u).normalized()
        rings = []
        for k in range(cap_rings + 1):                       # hemisphere at a
            t = (math.pi / 2) * k / cap_rings
            r = ra * math.sin(t)
            rings.append((a - w * (ra * math.cos(t)), u, v, r, r))
        for k in range(1, cap_rings + 1):                    # hemisphere at b
            t = (math.pi / 2) * k / cap_rings
            r = rb * math.cos(t)
            rings.append((b + w * (rb * math.sin(t)), u, v, r, r))
        rings[0] = (rings[0][0], u, v, 0, 0)
        rings[-1] = (rings[-1][0], u, v, 0, 0)
        return self.lathe(rings, segs, bone=bone)

    def ellipsoid(self, c, axes, radii, bone, exponent=2.0, segs=28, rings=16):
        """Superellipsoid; axes = (ax, ay, az) orthonormal Vectors, radii along each."""
        ax, ay, az = axes
        rr = []
        for k in range(rings + 1):
            t = math.pi * k / rings                    # 0..pi along ax
            ct, st = math.cos(t), math.sin(t)
            e = 2.0 / exponent
            cz = math.copysign(abs(ct) ** e, ct)
            sz = abs(st) ** e
            rr.append((c - ax * (radii[0] * cz), ay, az, radii[1] * sz, radii[2] * sz))
        rr[0] = (rr[0][0], ay, az, 0, 0)
        rr[-1] = (rr[-1][0], ay, az, 0, 0)
        if exponent == 2.0:
            return self.lathe(rr, segs, bone=bone)
        # squarer cross-section too
        base = self.lathe(rr, segs, bone=bone)
        for i in range(base, len(self.verts)):
            p = self.verts[i] - c
            y, z = p.dot(ay), p.dot(az)
            rho = math.hypot(y, z)
            if rho < 1e-9:
                continue
            ang = math.atan2(z, y)
            cy, sy = math.cos(ang), math.sin(ang)
            n = (abs(cy) ** exponent + abs(sy) ** exponent) ** (1 / exponent)
            self.verts[i] = c + ax * p.dot(ax) + (ay * y + az * z) / n
        return base

    def to_object(self, name, arm_obj, material):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in self.verts], [], self.faces)
        me.update()
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        ob = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(ob)
        ob.parent = arm_obj
        groups = {}
        for vi, wd in enumerate(self.weights):
            for bone, wv in wd.items():
                if wv <= 0:
                    continue
                if bone not in groups:
                    groups[bone] = ob.vertex_groups.new(name=bone)
                groups[bone].add([vi], wv, "REPLACE")
        mod = ob.modifiers.new("Armature", "ARMATURE")
        mod.object = arm_obj
        me.materials.append(material)
        return ob


# =============================================================================== materials
def vinyl_material(rgb):
    m = bpy.data.materials.new("SoftVinyl")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = 0.48
    b.inputs["Specular IOR Level"].default_value = 0.35
    b.inputs["Coat Weight"].default_value = 0.12
    b.inputs["Coat Roughness"].default_value = 0.35
    b.inputs["Subsurface Weight"].default_value = 0.0
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.6, 0.45)
    b.inputs["Subsurface Scale"].default_value = 0.02
    return m


def face_material():
    m = bpy.data.materials.new("FaceInk")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.018, 0.02, 0.035, 1)
    b.inputs["Roughness"].default_value = 0.22
    return m


def backdrop_material():
    m = bpy.data.materials.new("Backdrop")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.80, 0.81, 0.83, 1)
    b.inputs["Roughness"].default_value = 0.95
    b.inputs["Specular IOR Level"].default_value = 0.1
    return m


# =============================================================================== armature
def build_armature(J):
    arm = bpy.data.armatures.new("Rig")
    ob = bpy.data.objects.new("Character", arm)
    bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones
    up = Vector((0, 1, 0))
    for i, name in enumerate(SMPL_NAMES):
        b = eb.new(name)
        b.head = Vector(J[i])
        if i in BONE_TAIL:
            tail = Vector(J[BONE_TAIL[i]])
        elif i == 15:
            tail = Vector(J[15]) + up * 0.18
        elif i in (10, 11):
            tail = Vector(J[i]) + Vector((0, 0, 0.05))
        else:  # wrists: continue the forearm
            d = (Vector(J[i]) - Vector(J[i - 2])).normalized()
            tail = Vector(J[i]) + d * 0.08
        if (tail - b.head).length < 0.02:
            tail = b.head + up * 0.05
        b.tail = tail
        b.align_roll(Vector((0, 0, 1)) if abs((tail - b.head).normalized().z) < 0.9 else up)
    for i, name in enumerate(SMPL_NAMES):
        if i:
            eb[name].parent = eb[SMPL_NAMES[[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19][i]]]
    hand_geo = {}
    for side, wi, ei in (("L", 20, 18), ("R", 21, 19)):
        hand_geo[side] = build_hand_bones(eb, side, Vector(J[wi]), Vector(J[ei]))
    bpy.ops.object.mode_set(mode="OBJECT")
    return ob, hand_geo


def hand_frame(wrist, elbow):
    d = (wrist - elbow)
    d.y = 0.0                                      # SMPL rest arms are horizontal
    d.normalize()
    n = Vector((0, -1, 0))                         # palm faces down in rest pose
    a = Vector((0, 0, 1))                          # thumb side points forward
    a = (a - d * a.dot(d)).normalized()
    return d, a, n


def build_hand_bones(eb, side, wrist, elbow):
    d, a, n = hand_frame(wrist, elbow)
    wrist_bone = eb[f"{side}_Wrist"]
    geo = {"d": d, "a": a, "n": n, "wrist": wrist, "fingers": {}}
    knuck = wrist + d * 0.088
    offs = {"Index": 0.031, "Middle": 0.0105, "Ring": -0.0105, "Pinky": -0.030}
    scale = {"Index": 1.0, "Middle": 1.08, "Ring": 1.0, "Pinky": 0.82}
    for f in FINGERS:
        if f == "Thumb":
            start = wrist + d * 0.028 + a * 0.032 + n * 0.008
            fd = (d * 0.55 + a * 0.75 + n * 0.25).normalized()
            lens = (0.030, 0.024, 0.020)
            roll_to = (n - a * 0.8).normalized()
        else:
            start = knuck + a * offs[f]
            fd = d.copy()
            lens = tuple(l * scale[f] for l in (0.030, 0.021, 0.017))
            roll_to = n
        pts = [start]
        for l in lens:
            pts.append(pts[-1] + fd * l)
        parent = wrist_bone
        for k in range(3):
            b = eb.new(f"{side}_{f}{k + 1}")
            b.head, b.tail = pts[k], pts[k + 1]
            b.align_roll(roll_to)
            b.parent = parent
            parent = b
        geo["fingers"][f] = pts
    return geo


# =============================================================================== body geometry
def build_body(J, hand_geo, rgb, arm_obj):
    mb = MeshBuilder()
    V = lambda i: Vector(J[i])
    for s, (hip, knee, ank, toe) in {"L": (1, 4, 7, 10), "R": (2, 5, 8, 11)}.items():
        mb.capsule(V(hip), V(knee), *R["thigh"], bone=SMPL_NAMES[hip])
        mb.capsule(V(knee), V(ank), *R["shin"], bone=SMPL_NAMES[knee])
        mb.capsule(V(ank), V(toe), *R["foot"], bone=SMPL_NAMES[ank])
        mb.capsule(V(toe), V(toe) + Vector((0, 0, 0.035)), *R["toe"], bone=SMPL_NAMES[toe])
    for col, sh, el, wr in ((13, 16, 18, 20), (14, 17, 19, 21)):
        mb.capsule(V(col), V(sh), *R["shoulder"], bone=SMPL_NAMES[col])
        mb.capsule(V(sh), V(el), *R["upper_arm"], bone=SMPL_NAMES[sh])
        mb.capsule(V(el), V(wr), *R["forearm"], bone=SMPL_NAMES[el])
    mb.capsule(V(12), V(15), R["neck"], R["neck"], bone="Neck")
    mb.ellipsoid(V(15) + HEAD_CENTER_OFFSET, (Vector((0, 1, 0)), Vector((1, 0, 0)), Vector((0, 0, 1))),
                 (HEAD_RADII.y, HEAD_RADII.x, HEAD_RADII.z), bone="Head")
    build_torso(mb, J)
    for side, g in hand_geo.items():
        d, a, n, w = g["d"], g["a"], g["n"], g["wrist"]
        mb.ellipsoid(w + d * 0.052 + n * 0.004, (d, a, n), (0.056, 0.047, 0.026), bone=f"{side}_Wrist",
                     exponent=2.6)
        for f, pts in g["fingers"].items():
            r0 = 0.0135 if f == "Thumb" else 0.0118
            for k in range(3):
                mb.capsule(pts[k], pts[k + 1], r0 - 0.0008 * k, r0 - 0.0008 * (k + 1),
                           bone=f"{side}_{f}{k + 1}", segs=16, cap_rings=6)
    return mb.to_object("Body", arm_obj, vinyl_material(rgb))


def build_torso(mb, J):
    """Chubby pill torso as a lathe of elliptical rings, skinned along the spine."""
    spine = [0, 3, 6, 9, 12]
    ys = np.array([J[i][1] for i in spine])
    zs = np.array([J[i][2] for i in spine])
    y0, y1 = J[0][1] - 0.135, J[12][1] + 0.015
    # Pill profile (flat sides, round ends) with a soft belly bulge.
    n = 44
    s = np.linspace(0, 1, n)
    e = np.clip(1 - np.abs(2 * s - 1) ** 2.6, 0, 1) ** (1 / 2.6)
    rx = e * (0.150 + 0.020 * np.exp(-((s - 0.40) / 0.24) ** 2))
    rz = e * (0.112 + 0.024 * np.exp(-((s - 0.42) / 0.24) ** 2))
    rings = []
    for si, a, b in zip(s, rx, rz):
        y = y0 + si * (y1 - y0)
        z = float(np.interp(y, ys, zs)) + 0.012
        rings.append((Vector((0, y, z)), Vector((1, 0, 0)), Vector((0, 0, 1)), a, b))
    # Smooth (gaussian) weights around each spine bone's segment midpoint, so bending
    # spreads over the whole torso instead of folding at one ring.
    mids = np.array([(J[spine[k]][1] + J[spine[k + 1]][1]) / 2 for k in range(4)] + [J[12][1] + 0.06])
    names = ["Pelvis", "Spine1", "Spine2", "Spine3", "Neck"]
    falloff = np.array([1.0, 1.0, 1.0, 1.0, 0.35])       # neck only nudges the top

    def wfn(p):
        y = min(max(p.y, mids[0]), mids[-1])
        w = falloff * np.exp(-0.5 * ((y - mids) / 0.075) ** 2)
        w /= w.sum()
        return {n: float(x) for n, x in zip(names, w) if x > 0.01}
    mb.lathe(rings, 32, weight_fn=wfn)


# =============================================================================== face
def head_surface(J):
    c = Vector(J[15]) + HEAD_CENTER_OFFSET

    def at(u, v, lift=0.0025):
        d = Vector((u, v, 1.0)).normalized()
        t = 1.0 / math.sqrt((d.x / HEAD_RADII.x) ** 2 + (d.y / HEAD_RADII.y) ** 2 + (d.z / HEAD_RADII.z) ** 2)
        p = d * t
        nrm = Vector((p.x / HEAD_RADII.x ** 2, p.y / HEAD_RADII.y ** 2, p.z / HEAD_RADII.z ** 2)).normalized()
        return c + p + nrm * lift
    return at


def strip(upper, lower, at):
    """Quad strip between two (u, v) polylines of equal length, mapped onto the head."""
    n = len(upper)
    verts = [at(*p) for p in upper] + [at(*p) for p in lower]
    faces = [(i, i + 1, n + i + 1, n + i) for i in range(n - 1)]
    return verts, faces


def face_shapes():
    """Return dict key -> list of (upper, lower) polylines per feature, same topology for every key."""
    N = 17
    x = np.linspace(-1, 1, N)
    feats = {}

    # eyes: ellipses as strips between upper and lower arcs
    def eye(uc, hw, hh, vc):
        arc = np.sqrt(np.clip(1 - x ** 2, 0, 1))
        return ([(uc + hw * xi, vc + hh * ai) for xi, ai in zip(x, arc)],
                [(uc + hw * xi, vc - hh * ai) for xi, ai in zip(x, arc)])
    for side, uc in (("L", 0.33), ("R", -0.33)):
        feats[f"eye{side}"] = {
            "Basis": eye(uc, 0.105, 0.150, 0.05),
            "Blink": ([(uc + 0.125 * xi, 0.0 - 0.05 * (1 - xi ** 2) + 0.020) for xi in x],
                      [(uc + 0.125 * xi, 0.0 - 0.05 * (1 - xi ** 2) - 0.020) for xi in x]),
            "Wide": eye(uc, 0.128, 0.190, 0.065),
        }

    # brows: arched bars; t in [0,1] from outer to inner end
    t = (x + 1) / 2
    def brow(uc, sgn, dv=0.0, inner=0.0):
        cu = [uc + sgn * 0.15 * (1 - 2 * ti) for ti in t]                 # sgn=+1: outer on +u side
        cv = [0.33 + dv + 0.045 * (1 - (2 * ti - 1) ** 2) + inner * ti for ti in t]
        th = [0.026 + 0.010 * (1 - (2 * ti - 1) ** 2) for ti in t]
        return ([(u, v + h) for u, v, h in zip(cu, cv, th)], [(u, v - h) for u, v, h in zip(cu, cv, th)])
    for side, uc, sgn in (("L", 0.34, 1), ("R", -0.34, -1)):
        feats[f"brow{side}"] = {"Basis": brow(uc, sgn), "BrowUp": brow(uc, sgn, 0.10, 0.08),
                                "BrowDown": brow(uc, sgn, -0.06, -0.09)}

    # mouth: centre line c(x), lips thickness, optional opening
    def mouth(hw=0.24, curve=0.0, open_=0.0, vm=-0.30, th=0.024):
        c = vm + curve * (x ** 2 - 0.35)
        o = open_ * np.sqrt(np.clip(1 - x ** 2, 0, 1))
        return ([(hw * xi, ci + th + 0.25 * oi) for xi, ci, oi in zip(x, c, o)],
                [(hw * xi, ci - th - oi) for xi, ci, oi in zip(x, c, o)])
    feats["mouth"] = {"Basis": mouth(), "Smile": mouth(hw=0.29, curve=0.15), "Frown": mouth(hw=0.22, curve=-0.12),
                      "Open": mouth(hw=0.19, open_=0.26)}
    return feats


def build_face(J, arm_obj):
    at = head_surface(J)
    feats = face_shapes()
    keys = ["Basis", "Blink", "Wide", "BrowUp", "BrowDown", "Smile", "Frown", "Open"]
    verts = {k: [] for k in keys}
    faces = []
    for name, shp in feats.items():
        base_v, f = strip(*shp["Basis"], at)
        off = len(verts["Basis"])
        faces += [tuple(i + off for i in q) for q in f]
        for k in keys:
            v = strip(*shp[k], at)[0] if k in shp else base_v
            verts[k].extend(v)
    me = bpy.data.meshes.new("Face")
    me.from_pydata([tuple(v) for v in verts["Basis"]], [], faces)
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    ob = bpy.data.objects.new("Face", me)
    bpy.context.collection.objects.link(ob)
    ob.parent = arm_obj
    vg = ob.vertex_groups.new(name="Head")
    vg.add(list(range(len(me.vertices))), 1.0, "REPLACE")
    ob.modifiers.new("Armature", "ARMATURE").object = arm_obj
    me.materials.append(face_material())
    ob.shape_key_add(name="Basis")
    for k in keys[1:]:
        sk = ob.shape_key_add(name=k)
        for i, v in enumerate(verts[k]):
            sk.data[i].co = v
    return ob


# =============================================================================== animation
def animate(arm_obj, face_obj, data):
    local = data["local_rot"]
    root = data["root_pos"]
    J = data["rest_joints"]
    ang = data["finger_angles"]
    T = len(local)
    pose = arm_obj.pose
    bones = arm_obj.data.bones
    for pb in pose.bones:
        pb.rotation_mode = "QUATERNION"
    B = {b.name: b.matrix_local.to_3x3() for b in bones}
    Binv = {k: m.inverted() for k, m in B.items()}

    # spread sign per finger bone: positive rotation about local Z moves the finger toward the pinky
    spread_sign = {}
    for side in "LR":
        _, a, _ = hand_frame(Vector(J[20 if side == "L" else 21]), Vector(J[18 if side == "L" else 19]))
        for f in FINGERS[1:]:
            X = B[f"{side}_{f}1"].col[0]
            spread_sign[f"{side}_{f}1"] = 1.0 if X.dot(a) > 0 else -1.0

    prev = {}

    def key(pb, q, frame):
        p = prev.get(pb.name)
        if p is not None and p.dot(q) < 0:
            q = -q
        prev[pb.name] = q
        pb.rotation_quaternion = q
        pb.keyframe_insert("rotation_quaternion", frame=frame)

    clamp = lambda v, lo, hi: max(lo, min(hi, float(v)))
    for t in range(T):
        f = t + 1
        for i, name in enumerate(SMPL_NAMES):
            Rl = Matrix(local[t, i].tolist())
            q = (Binv[name] @ Rl @ B[name]).to_quaternion()
            key(pose.bones[name], q, f)
        pb = pose.bones["Pelvis"]
        pb.location = Binv["Pelvis"] @ (Vector(root[t]) - Vector(J[0]))
        pb.keyframe_insert("location", frame=f)

        for hi, side in enumerate("LR"):
            for fi, fname in enumerate(FINGERS):
                c1, c2, c3, sp = ang[t, hi, fi]
                if fname == "Thumb":
                    curls = (clamp(0.8 * (math.radians(50) - c1), 0, math.radians(45)),
                             clamp(c2, 0, math.radians(70)), clamp(c3, 0, math.radians(80)))
                    spread = 0.0
                else:
                    curls = tuple(clamp(c, 0, math.radians(100)) for c in (c1, c2, c3))
                    spread = clamp(sp, -math.radians(25), math.radians(25))
                for k in range(3):
                    bn = f"{side}_{fname}{k + 1}"
                    q = Quaternion((1, 0, 0), curls[k])
                    if k == 0 and spread:
                        q = Quaternion((0, 0, 1), spread * spread_sign[bn]) @ q
                    key(pose.bones[bn], q, f)

    # face
    chans = [str(c) for c in data["face_channels"]]
    face = data["face"]
    kb = face_obj.data.shape_keys.key_blocks
    vals = {}
    for k, (ch, lo, hi) in FACE_MAP.items():
        vals[k] = smoothstep((face[:, chans.index(ch)] - lo) / (hi - lo))
    vals["Wide"] = vals["Wide"] * (1 - vals["Blink"])
    vals["Frown"] = vals["Frown"] * (1 - 0.7 * vals["Smile"])
    for t in range(T):
        for k, v in vals.items():
            kb[k].value = float(v[t])
            kb[k].keyframe_insert("value", frame=t + 1)

    for fc in _all_fcurves(arm_obj) + _all_fcurves(face_obj.data.shape_keys):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def _all_fcurves(idblock):
    ad = idblock.animation_data
    if not ad or not ad.action:
        return []
    act = ad.action
    if hasattr(act, "fcurves") and len(getattr(act, "fcurves", [])):
        return list(act.fcurves)
    out = []                                   # layered actions (Blender 4.4+)
    for layer in getattr(act, "layers", []):
        for strip_ in layer.strips:
            for cb in strip_.channelbags:
                out.extend(cb.fcurves)
    return out


# =============================================================================== scene
def setup_scene(data, args):
    sc = bpy.context.scene
    T = len(data["local_rot"])
    sc.frame_start, sc.frame_end = 1, T
    sc.render.fps = int(round(float(data["fps"])))
    sc.render.resolution_x, sc.render.resolution_y = args.width, args.height
    sc.render.resolution_percentage = 100
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            sc.render.engine = eng
            break
        except TypeError:
            continue
    ee = sc.eevee
    if hasattr(ee, "taa_render_samples"):
        ee.taa_render_samples = args.samples
    for attr, val in (("use_shadows", True), ("shadow_ray_count", 2), ("shadow_step_count", 8),
                      ("use_raytracing", False), ("fast_gi_method", "GLOBAL_ILLUMINATION")):
        if hasattr(ee, attr):
            try:
                setattr(ee, attr, val)
            except Exception:
                pass
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    sc.view_settings.exposure = 0.0
    sc.render.film_transparent = False

    world = bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.86, 0.87, 0.89, 1)
    bg.inputs["Strength"].default_value = 0.35

    build_backdrop()
    add_lights()
    add_camera(data, args)


def build_backdrop():
    """Cyclorama: floor that sweeps up into a back wall, so the background reads as plain."""
    prof = []
    for y in np.linspace(-8, 1.5, 12):
        prof.append((y, 0.0))
    for k in range(1, 17):
        a = (math.pi / 2) * k / 16
        prof.append((1.5 + 2.5 * math.sin(a), 2.5 - 2.5 * math.cos(a)))
    prof.append((4.0, 10.0))
    xs = np.linspace(-10, 10, 2)
    verts = [(x, y, z) for (y, z) in prof for x in xs]
    faces = [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(len(prof) - 1)]
    me = bpy.data.meshes.new("Backdrop")
    me.from_pydata(verts, [], faces)
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    ob = bpy.data.objects.new("Backdrop", me)
    bpy.context.collection.objects.link(ob)
    me.materials.append(backdrop_material())


def _aim(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def add_lights():
    def area(name, loc, power, size, color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size, ld.color = power, size, color
        ld.shape = "DISK"
        ob = bpy.data.objects.new(name, ld)
        bpy.context.collection.objects.link(ob)
        ob.location = loc
        _aim(ob, (0, 0, 0.9))
        return ob
    area("Key", (-2.8, -1.4, 3.6), 1100, 1.0, (1.0, 0.97, 0.93))
    area("Fill", (2.4, -3.0, 1.6), 200, 4.0, (0.93, 0.96, 1.0))
    area("Top", (0.0, 0.8, 4.5), 180, 2.0)
    area("Rim", (1.5, 2.2, 2.6), 220, 1.5)


def add_camera(data, args):
    P = data["joints"].reshape(-1, 3)
    # SMPL (x, y, z) -> Blender (x, -z, y)
    pts = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)
    lo, hi = pts.min(0), pts.max(0)
    lo = np.minimum(lo - [0.20, 0.15, 0.0], [np.inf, np.inf, 0.0])
    hi = hi + [0.20, 0.15, 0.30]                       # head top / hands / margins
    lo[2] = 0.0
    c = (lo + hi) / 2
    cam_d = bpy.data.cameras.new("Cam")
    cam_d.lens = 50
    cam_d.sensor_fit = "VERTICAL"
    cam_d.sensor_height = 24
    ob = bpy.data.objects.new("Cam", cam_d)
    bpy.context.collection.objects.link(ob)
    bpy.context.scene.camera = ob
    vfov = 2 * math.atan(12 / 50)
    hfov = 2 * math.atan(math.tan(vfov / 2) * args.width / args.height)
    margin = 1.10
    hz = (hi[2] - lo[2]) / 2 * margin
    hx = (hi[0] - lo[0]) / 2 * margin
    dist = max(hz / math.tan(vfov / 2), hx / math.tan(hfov / 2)) + (hi[1] - lo[1]) / 2
    target = Vector((c[0], c[1], c[2] - 0.02))
    elev = math.radians(6)
    ob.location = target + Vector((0, -dist * math.cos(elev), dist * math.sin(elev)))
    _aim(ob, target)


# =============================================================================== main
def main():
    args = parse_args()
    data = dict(np.load(args.motion))
    bpy.ops.wm.read_homefile(use_empty=True)
    J = data["rest_joints"]
    arm_obj, hand_geo = build_armature(J)
    body = build_body(J, hand_geo, hex_rgb(args.color), arm_obj)
    face = build_face(J, arm_obj)
    animate(arm_obj, face, data)
    # SMPL Y-up/+Z-forward -> Blender Z-up/-Y-forward
    arm_obj.rotation_euler = (math.radians(90), 0, 0)
    setup_scene(data, args)

    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.blend))

    sc = bpy.context.scene
    sc.render.image_settings.file_format = "PNG"
    if args.still >= 0:
        sc.frame_set(args.still + 1)
        sc.render.filepath = os.path.abspath(args.still_out)
        bpy.ops.render.render(write_still=True)
    elif args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
        sc.render.filepath = os.path.join(os.path.abspath(args.frames_dir), "frame_")
        bpy.ops.render.render(animation=True)

    if args.glb:
        bpy.ops.object.select_all(action="DESELECT")
        for ob in (arm_obj, body, face):
            ob.select_set(True)
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.export_scene.gltf(filepath=os.path.abspath(args.glb), export_format="GLB", use_selection=True,
                                  export_animations=True, export_skins=True, export_morph=True,
                                  export_apply=False, export_yup=True,
                                  export_animation_mode="ACTIVE_ACTIONS")  # body + face in one clip
        print(f"[blender] wrote {args.glb}")


if __name__ == "__main__":
    main()
