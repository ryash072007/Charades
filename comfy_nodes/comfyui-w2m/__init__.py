"""WANPhrase2Motion speed-ups for ComfyUI.

W2MPoseKpsToBBoxMask      DWPose keypoints -> white-box-on-black frames. GVHMR only reads a bounding box
                          from its mask video, so this replaces a ~20 s SAM3 pass.
W2MWanVaceToVideoCached   WanVaceToVideo without control video / masks / reference image, with the
                          (constant) grey control-video latent cached per VAE + size + length.
W2MWanImageToVideoCached  WanImageToVideo (start image only) with the encoded start-image video cached.
W2MCompileWan             torch.compile on each Wan transformer block, with the weight-casting ops (NVFP4 Linears,
                          dynamic-VRAM streaming) and the attention kernel hidden behind opaque custom ops so
                          every block traces as one graph and inductor fuses the glue between them.
"""
import logging
import os

import numpy as np
import torch

import comfy.latent_formats
import comfy.model_management
import comfy.ops
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


class W2MWanImageToVideoCached:
    """WanImageToVideo (start image only) with the encoded start-image video cached. The stock node VAE-encodes the
    start image plus (length - 1) grey frames on every run (~5 s at 480x832x49), but that only depends on the image,
    size and length, which stay the same for every phrase with a fixed reference character."""
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "start_image": ("IMAGE",),
            "width": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
            "length": ("INT", {"default": 49, "min": 1, "max": 4096, "step": 4}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "run"
    CATEGORY = "WANPhrase2Motion"

    def run(self, positive, negative, vae, start_image, width, height, length, batch_size):
        import hashlib
        import comfy.utils
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
                             device=comfy.model_management.intermediate_device())
        img = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        key = (id(vae), width, height, length, hashlib.sha1(img.float().cpu().numpy().tobytes()).hexdigest())
        cached = self._cache.get(key)
        if cached is None:
            video = torch.ones((length, height, width, img.shape[-1]), device=img.device, dtype=img.dtype) * 0.5
            video[:img.shape[0]] = img
            concat = vae.encode(video[:, :, :, :3])
            mask = torch.ones((1, 1, latent.shape[2], concat.shape[-2], concat.shape[-1]), device=img.device, dtype=img.dtype)
            mask[:, :, :((img.shape[0] - 1) // 4) + 1] = 0.0
            if len(self._cache) > 8:
                self._cache.clear()
            cached = self._cache[key] = (concat, mask)
        concat, mask = cached
        vals = {"concat_latent_image": concat, "concat_mask": mask}
        positive = node_helpers.conditioning_set_values(positive, vals)
        negative = node_helpers.conditioning_set_values(negative, vals)
        return (positive, negative, {"samples": latent})


# ComfyUI caps pinned RAM at 40% on Windows. The two Wan 2.2 A14B NVFP4 experts are 16.8 GB and only one fits in 16 GB
# of VRAM, so both stream in for every clip; whatever isn't pinned streams from pageable memory several times slower.
if os.environ.get("W2M_PIN_FRACTION") and comfy.model_management.MAX_PINNED_MEMORY > 0:
    comfy.model_management.MAX_PINNED_MEMORY = (comfy.model_management.get_total_memory(torch.device("cpu"))
                                                * float(os.environ["W2M_PIN_FRACTION"]))
    logging.info("W2M: pinned memory cap %d MB", comfy.model_management.MAX_PINNED_MEMORY // (1024 * 1024))


# Opaque ops for W2MCompileWan. Dynamo can't trace comfy-kitchen's quantized tensors, ComfyUI's dynamic-VRAM weight
# streaming (uint8 buffers reinterpreted as other dtypes) or SageAttention's pybind kernels, so each one runs eagerly
# inside a custom op and the compiled graph only sees "tensor in, tensor out".
_W2M_MODULES = []


@torch.library.custom_op("w2m::module", mutates_args=())
def _w2m_module(x: torch.Tensor, idx: int) -> torch.Tensor:
    y = _W2M_MODULES[idx]._w2m_eager_forward(x)
    return y if y.dtype == x.dtype else y.to(x.dtype)


@_w2m_module.register_fake
def _(x, idx):
    m = _W2M_MODULES[idx]
    return x.new_empty((*x.shape[:-1], m.out_features) if hasattr(m, "out_features") else x.shape)


_W2M_ATTN = {}


def _w2m_sage3(q, k, v, heads, per_block_mean):
    """SageAttention3 (FP4 attention on Blackwell) on Wan's (B, L, heads*dim) tensors."""
    from sageattn3 import sageattn3_blackwell
    b, d = q.shape[0], q.shape[-1] // heads
    q, k, v = (t.view(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
    out = sageattn3_blackwell(q, k, v, is_causal=False, per_block_mean=per_block_mean)
    return out.transpose(1, 2).reshape(b, -1, heads * d)


def _w2m_attn_eager(q, k, v, heads, kind):
    if kind == "default":
        return _W2M_ATTN["fn"](q, k, v, heads=heads)
    return _w2m_sage3(q, k, v, heads, per_block_mean=(kind == "sage3_per_block_mean"))


@torch.library.custom_op("w2m::sage3_core", mutates_args=())
def _w2m_sage3_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, delta_s: torch.Tensor, kl: int,
                    per_block_mean: bool) -> torch.Tensor:
    """SageAttention3's FP4 quantization + attention kernel on already centred, 128-padded (B, H, L, D) tensors."""
    from sageattn3.api import (scale_and_quant_fp4, scale_and_quant_fp4_permute, scale_and_quant_fp4_transpose,
                               blockscaled_fp4_attn)
    q, k, v, delta_s = (t.contiguous() for t in (q, k, v, delta_s))
    return blockscaled_fp4_attn(scale_and_quant_fp4(q), scale_and_quant_fp4_permute(k), scale_and_quant_fp4_transpose(v),
                                delta_s, kl, False, per_block_mean, q.dtype == torch.bfloat16)[0]


@_w2m_sage3_core.register_fake
def _(q, k, v, delta_s, kl, per_block_mean):
    return torch.empty_like(q)


def _w2m_sage3_traced(q, k, v, heads, per_block_mean):
    """SageAttention3 with its preprocessing (k / q mean removal, pad to 128, layout) written as plain tensor ops, so
    inductor fuses it with the surrounding RoPE / norm instead of running it as separate eager kernels. Same maths as
    sageattn3.api.preprocess_qkv, minus its in-place k -= mean."""
    b, d = q.shape[0], q.shape[-1] // heads
    q, k, v = (t.view(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
    L = q.shape[2]
    pad = (128 - L % 128) % 128
    k = k - k.mean(dim=-2, keepdim=True)
    q, k, v = (torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    if per_block_mean:
        g = q.view(b, heads, -1, 128, d)
        qm = g.mean(dim=-2)
        q = (g - qm.unsqueeze(-2)).view(b, heads, -1, d)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    delta_s = torch.matmul(qm, k.transpose(-2, -1)).float()
    out = torch.ops.w2m.sage3_core(q, k, v, delta_s, L, per_block_mean)[:, :, :L]
    return out.transpose(1, 2).reshape(b, -1, heads * d)


@torch.library.custom_op("w2m::attn", mutates_args=())
def _w2m_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, kind: str) -> torch.Tensor:
    return _w2m_attn_eager(q, k, v, heads, kind)


@_w2m_attn.register_fake
def _(q, k, v, heads, kind):
    return q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))


def _w2m_patch_wan_attention():
    """Route Wan's attention through w2m::attn. The kernel choice rides in transformer_options["w2m_attention"], set
    only on models patched by W2MCompileWan; everything else keeps ComfyUI's own attention."""
    import comfy.ldm.wan.model as wan
    if "fn" in _W2M_ATTN:
        return
    orig = _W2M_ATTN["fn"] = wan.optimized_attention

    def attn(q, k, v, heads, mask=None, transformer_options={}, **kw):
        kind = transformer_options.get("w2m_attention")
        if kind is not None and mask is None and not kw:
            if torch.compiler.is_compiling():
                # sage3 only for video self-attention; text cross-attention (512 keys) is tiny, stays on the default
                if kind != "default" and q.shape[1] > 1024 and k.shape[1] > 1024:
                    return _w2m_sage3_traced(q, k, v, heads, per_block_mean=(kind == "sage3_per_block_mean"))
                return torch.ops.w2m.attn(q, k, v, heads, kind)
            if kind != "default":
                return _w2m_attn_eager(q, k, v, heads, kind)
        return orig(q, k, v, heads, mask=mask, transformer_options=transformer_options, **kw)
    wan.optimized_attention = attn


def _w2m_patch_module(m):
    if hasattr(m, "_w2m_eager_forward"):
        return
    m._w2m_eager_forward = m.forward
    idx = len(_W2M_MODULES)
    _W2M_MODULES.append(m)

    def forward(x):
        if torch.compiler.is_compiling():
            return torch.ops.w2m.module(x, idx)
        return m._w2m_eager_forward(x)
    m.forward = forward


def _w2m_raise_recompile_limits():
    # all blocks share one forward code object and each is its own graph (it bakes in its modules' ids), x2 experts,
    # x every resolution / length used in this session
    for k in ("recompile_limit", "accumulated_recompile_limit"):
        if hasattr(torch._dynamo.config, k):
            setattr(torch._dynamo.config, k, max(getattr(torch._dynamo.config, k), 1024))


def _w2m_limits_wrapper(executor, *args, **kwargs):
    _w2m_raise_recompile_limits()  # torch config overrides are thread-local; this runs on the sampling thread
    return executor(*args, **kwargs)


class W2MCompileWan:
    """torch.compile (inductor) per Wan transformer block. Needs Triton and a C compiler (MSVC on Windows). The first
    clip at a new resolution / length compiles (a few minutes); later clips reuse the kernels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mode": (["default", "max-autotune-no-cudagraphs"], {"default": "default"}),
            "fullgraph": ("BOOLEAN", {"default": True}),
            "compile": ("BOOLEAN", {"default": True}),
            "attention": (["default", "sage3", "sage3_per_block_mean"], {"default": "default",
                          "tooltip": "default = the global attention (--use-sage-attention); sage3 = SageAttention3 FP4"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "W2M"

    def patch(self, model, mode, fullgraph, compile=True, attention="default"):
        from comfy_api.torch_helpers import set_torch_compile_wrapper
        m = model.clone()
        dm = m.get_model_object("diffusion_model")
        _w2m_patch_wan_attention()
        to = m.model_options["transformer_options"] = dict(m.model_options.get("transformer_options", {}))
        to["w2m_attention"] = attention
        if not compile:
            return (m,)
        for block in dm.blocks:
            for mod in block.modules():
                # every comfy op that casts / streams a weight: Linears (incl. the mixed-precision NVFP4 one, a plain
                # nn.Module) and affine norms. Weightless norms stay in the graph and fuse with the modulation.
                if isinstance(mod, comfy.ops.CastWeightBiasOp) and getattr(mod, "weight", None) is not None:
                    _w2m_patch_module(mod)
        from comfy.patcher_extension import WrappersMP
        m.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "w2m_recompile_limits", _w2m_limits_wrapper)
        torch._dynamo.config.force_parameter_static_shapes = True
        # transformer_options carries per-step values (sigmas etc.); guarding on it would recompile every step. Keep
        # only the guard on the attention choice, which is baked into the graph.
        keep = lambda entries: ["transformer_options" not in e.name or "w2m_attention" in e.name for e in entries]
        options = {"guard_filter_fn": keep}
        if mode != "default":  # torch.compile takes mode or options, not both
            options.update(torch._inductor.list_mode_options(mode))
        set_torch_compile_wrapper(model=m, keys=[f"diffusion_model.blocks.{i}" for i in range(len(dm.blocks))],
                                  backend="inductor", fullgraph=fullgraph, dynamic=False, options=options)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "W2MCompileWan": W2MCompileWan,
    "W2MPoseKpsToBBoxMask": W2MPoseKpsToBBoxMask,
    "W2MWanVaceToVideoCached": W2MWanVaceToVideoCached,
    "W2MWanImageToVideoCached": W2MWanImageToVideoCached,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "W2MCompileWan": "Compile Wan blocks, opaque quant/attention (W2M)",
    "W2MPoseKpsToBBoxMask": "Pose Keypoints to BBox Mask (W2M)",
    "W2MWanVaceToVideoCached": "WanVaceToVideo, cached blank control (W2M)",
    "W2MWanImageToVideoCached": "WanImageToVideo, cached start-image encode (W2M)",
}
