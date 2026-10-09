"""WANPhrase2Motion speed-ups for ComfyUI.

W2MPoseKpsToBBoxMask      DWPose keypoints -> white-box-on-black frames. GVHMR only reads a bounding box
                          from its mask video, so this replaces a ~20 s SAM3 pass.
W2MWanVaceToVideoCached   WanVaceToVideo without control video / masks / reference image, with the
                          (constant) grey control-video latent cached per VAE + size + length.
"""
import numpy as np
import torch

import comfy.latent_formats
import comfy.model_management
import node_helpers


class W2MPoseKpsToBBoxMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pose_kps": ("POSE_KEYPOINT",),
            "margin": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.5, "step": 0.01,
                                 "tooltip": "extra border, as a fraction of the box size"}),
            "min_confidence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask_frames",)
    FUNCTION = "run"
    CATEGORY = "WANPhrase2Motion"

    @staticmethod
    def _points(person, W, H, min_conf):
        pts = []
        for key in ("pose_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d", "face_keypoints_2d"):
            a = person.get(key)
            if not a:
                continue
            a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
            a = a[a[:, 2] > min_conf]
            if len(a) and a[:, :2].max() <= 1.0 + 1e-6:  # normalised coordinates
                a[:, 0] *= W
                a[:, 1] *= H
            pts.append(a[:, :2])
        return np.concatenate(pts) if pts else np.zeros((0, 2))

    def run(self, pose_kps, margin, min_confidence):
        W, H = int(pose_kps[0]["canvas_width"]), int(pose_kps[0]["canvas_height"])
        out = torch.zeros((len(pose_kps), H, W, 3), dtype=torch.float32)
        last = None
        for i, fr in enumerate(pose_kps):
            best = None
            for p in fr.get("people") or []:
                pts = self._points(p, W, H, min_confidence)
                if len(pts) < 4:
                    continue
                lo, hi = pts.min(0), pts.max(0)
                area = np.prod(hi - lo)
                if best is None or area > best[0]:
                    best = (area, lo, hi)
            if best is not None:
                _, lo, hi = best
                pad = (hi - lo) * margin
                last = (np.clip(lo - pad, 0, [W - 1, H - 1]).astype(int), np.clip(hi + pad, 0, [W - 1, H - 1]).astype(int))
            if last is not None:  # missing detection: hold the previous box
                (x0, y0), (x1, y1) = last
                out[i, y0:y1 + 1, x0:x1 + 1] = 1.0
        return (out,)


class W2MWanVaceToVideoCached:
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "width": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
            "length": ("INT", {"default": 49, "min": 1, "max": 4096, "step": 4}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "run"
    CATEGORY = "WANPhrase2Motion"

    def run(self, positive, negative, vae, width, height, length, batch_size, strength):
        # Same maths as comfy_extras.nodes_wan.WanVaceToVideo with no control video, masks or reference:
        # control = 0.5 grey, mask = 1 -> inactive and reactive are both all-0.5 frames.
        latent_length = ((length - 1) // 4) + 1
        key = (id(vae), width, height, length)
        cvl = self._cache.get(key)
        if cvl is None:
            grey = torch.ones((length, height, width, 3)) * 0.5
            enc = vae.encode(grey)  # inactive == reactive == grey here, so one encode serves both
            cvl = torch.cat((enc, enc), dim=1)
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = cvl

        vae_stride = 8
        hm, wm = height // vae_stride, width // vae_stride
        mask = torch.ones((length, hm, vae_stride, wm, vae_stride))
        mask = mask.permute(2, 4, 0, 1, 3).reshape(vae_stride * vae_stride, length, hm, wm)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, hm, wm), mode="nearest-exact").squeeze(0)
        mask = mask.unsqueeze(0)

        vals = {"vace_frames": [cvl], "vace_mask": [mask], "vace_strength": [strength]}
        positive = node_helpers.conditioning_set_values(positive, vals, append=True)
        negative = node_helpers.conditioning_set_values(negative, vals, append=True)
        latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8],
                             device=comfy.model_management.intermediate_device())
        return (positive, negative, {"samples": latent}, 0)


NODE_CLASS_MAPPINGS = {
    "W2MPoseKpsToBBoxMask": W2MPoseKpsToBBoxMask,
    "W2MWanVaceToVideoCached": W2MWanVaceToVideoCached,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "W2MPoseKpsToBBoxMask": "Pose Keypoints to BBox Mask (W2M)",
    "W2MWanVaceToVideoCached": "WanVaceToVideo, cached blank control (W2M)",
}
