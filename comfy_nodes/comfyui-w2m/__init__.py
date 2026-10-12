"""Speed-ups for the Wan 2.2 A14B stickman workflow in ComfyUI.

W2MWanImageToVideoCached  WanImageToVideo (start image only) with the encoded start-image video cached.
W2MCompileWan             torch.compile on each Wan transformer block, with the weight-casting ops (NVFP4 Linears,
                          dynamic-VRAM streaming) and the attention kernel hidden behind opaque custom ops so
                          every block traces as one graph and inductor fuses the glue between them.
"""
import logging
import os

import torch

import comfy.model_management
import comfy.ops
import node_helpers


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


def _w2m_attn_eager(q, k, v, heads):
    return _W2M_ATTN["fn"](q, k, v, heads=heads)


@torch.library.custom_op("w2m::attn", mutates_args=())
def _w2m_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    return _w2m_attn_eager(q, k, v, heads)


@_w2m_attn.register_fake
def _(q, k, v, heads):
    return q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))


def _w2m_patch_wan_attention():
    """Route Wan's attention through w2m::attn while a W2MCompileWan model is being traced; everything else keeps
    ComfyUI's own attention (SageAttention 2 with --use-sage-attention)."""
    import comfy.ldm.wan.model as wan
    if "fn" in _W2M_ATTN:
        return
    orig = _W2M_ATTN["fn"] = wan.optimized_attention

    def attn(q, k, v, heads, mask=None, transformer_options={}, **kw):
        if torch.compiler.is_compiling() and mask is None and not kw:
            return torch.ops.w2m.attn(q, k, v, heads)
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
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "W2M"

    def patch(self, model, mode, fullgraph):
        from comfy_api.torch_helpers import set_torch_compile_wrapper
        m = model.clone()
        dm = m.get_model_object("diffusion_model")
        _w2m_patch_wan_attention()
        for block in dm.blocks:
            for mod in block.modules():
                # every comfy op that casts / streams a weight: Linears (incl. the mixed-precision NVFP4 one, a plain
                # nn.Module) and affine norms. Weightless norms stay in the graph and fuse with the modulation.
                if isinstance(mod, comfy.ops.CastWeightBiasOp) and getattr(mod, "weight", None) is not None:
                    _w2m_patch_module(mod)
        from comfy.patcher_extension import WrappersMP
        m.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "w2m_recompile_limits", _w2m_limits_wrapper)
        torch._dynamo.config.force_parameter_static_shapes = True
        # transformer_options carries per-step values (sigmas etc.); guarding on it would recompile every step.
        keep = lambda entries: ["transformer_options" not in e.name for e in entries]
        options = {"guard_filter_fn": keep}
        if mode != "default":  # torch.compile takes mode or options, not both
            options.update(torch._inductor.list_mode_options(mode))
        set_torch_compile_wrapper(model=m, keys=[f"diffusion_model.blocks.{i}" for i in range(len(dm.blocks))],
                                  backend="inductor", fullgraph=fullgraph, dynamic=False, options=options)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "W2MCompileWan": W2MCompileWan,
    "W2MWanImageToVideoCached": W2MWanImageToVideoCached,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "W2MCompileWan": "Compile Wan blocks, opaque quant/attention (W2M)",
    "W2MWanImageToVideoCached": "WanImageToVideo, cached start-image encode (W2M)",
}
