# Speeding up Wan 2.2 A14B image-to-video (stickman)

Same model family (Wan 2.2 A14B image-to-video), same reference image (`relaxed_stickman_appendages.jpg`), same
prompts and seed. All numbers are for a **new phrase** (prompt not cached) on the RTX 5060 Ti 16 GB, warm models,
49 frames, output 480×832 at 16 fps. Per-node timings are in `results.csv`.

| Step | Variant (folder in `clips/`) | Time per clip | vs start |
|---|---|---|---|
| start | `baseline`: Q4_K_M GGUF experts + Lightning I2V LoRAs, full Wan 2.1 VAE | **66.2 s** | 1.0× |
| + tiny VAE decode (lighttaew2_1): 7.7 s → 0.5 s | `tae` | 59.2 s | 1.1× |
| + start image encoded once and cached (`W2MWanImageToVideoCached`): 5.2 s → 0 s | `cache_tae` | 54.6 s | 1.2× |
| + **LightX2V LightWan2.2 NVFP4 experts** (Blackwell FP4 tensor cores, 4-step distillation built in): sampling 50 s → 23 s | `nvfp4_cache_tae` | **26.3–26.8 s** | **2.5×** |
| + sample at 384×672, Lanczos up to 480×832, light sharpen: sampling 23 s → 14 s | `nvfp4_384_sharp` | **18.1 s** | **3.7×** |
| (too far) 320×560 | `nvfp4_320_sharp` | 14.1 s | 4.7×, but gestures degrade (thumbs up → finger-gun) |
| + torch.compile (`W2MCompileWan`, see below): sampling step 1.95–2.05 s → 1.88 s | `nvfp4_384_sharp_w2mc` | 17.7–18.1 s | 3.7× (−0.4 s) |

Tried and dropped:
- SageAttention 2 kernel variants (`nvfp4_sage_fp8`, `nvfp4_sage_fp8pp`): no gain. The default kernel is already the fastest here.
- ESRGAN anime ×4 upscale instead of Lanczos (`nvfp4_lowres`): crisper, but the upscale costs 22 s.
- SageAttention3 (FP4 attention), built from source for Python 3.13 + torch 2.12 + CUDA 13.3 (thu-ml/SageAttention with
  MSVC `/Zc:preprocessor` + C++20, plus mengqin/SageAttention's Windows fix for the 128-byte-aligned kernel params). At
  Wan's shape (40 heads × 128, 13k tokens) its kernel is 23% faster than SageAttention 2's (17.8 vs 23.1 ms), but its
  Python preprocessing eats the gain (26 ms per call either way) and it is less accurate (cosine 0.981 vs 0.999 against
  exact attention). With that preprocessing fused by torch.compile: 13.3–13.5 s sampling vs 13.6 s. Not worth the
  quality risk (`nvfp4_384_sharp_sage3`, `_w2mc_sage3`, `_w2mc_sage3pbm`).
- torch.compile with `max-autotune-no-cudagraphs` (`nvfp4_384_sharp_w2mc_at`): same 1.88 s step, 85 s first compile.
- KJNodes' `TorchCompileModelWanVideoV2` (`nvfp4_384_sharp_compile`): fails with dynamic VRAM (inductor can't trace the
  uint8 weight-streaming buffers); without dynamic VRAM it graph-breaks at every NVFP4 Linear and is slower (33 s sampling).
- LightX2V's own engine (sparse attention): their published RTX 5090 numbers, scaled to this GPU and 49 frames, land
  near what ComfyUI already does here; sparse attention mainly helps at 720p and above.

## torch.compile

Making it work at all: the portable Python is 3.13.12 but shipped 3.12.7 headers and `python312.lib`, so Triton's
compiled launcher couldn't load. `python_embeded/Include` and `libs` now hold 3.13 ones (from uv's CPython 3.13.14, same
ABI); the 3.12 ones are kept in `Include_py312_backup` / `libs_py312_backup`. ComfyUI runs inside the Visual Studio
environment (`comfy_generate.msvc_env`).

`W2MCompileWan` (comfyui-w2m) compiles each of the 40 transformer blocks with inductor, `fullgraph=True`. Dynamo can't
trace comfy-kitchen's NVFP4 tensors, ComfyUI's dynamic-VRAM weight streaming or SageAttention's kernels, so the node
hides those behind opaque custom ops (`w2m::module` for every weight-casting Linear / affine norm, `w2m::attn` for
attention). Every block is one graph with no breaks, dynamic VRAM keeps working, and inductor fuses the glue between them
(modulation, RoPE, residuals, GELU, weightless norms).

Why it's only ~3%: of a 1.9 s step at 384, SageAttention 2 is ~1.0 s (40 × 26 ms) and the NVFP4 matmuls most of the
rest, both hand-written kernels inductor can't improve. At 480×832 compile gains nothing (23.1 vs 22.9 s sampling). The
first clip after a ComfyUI start compiles (~16 s extra with inductor's disk cache, 60–80 s cold), so it pays back after
~35 clips per session: worth turning on for batch runs only.

Where the rest goes: each expert's first step is ~2 s slower than its second. The two NVFP4 experts (16.8 GB) don't both
fit in 16 GB of VRAM, so both stream in for every clip, and this GPU's link runs at PCIe 3.0 x8 (7 GB/s, pinned or not).
Raising ComfyUI's Windows pinned-memory cap from 40% to 50% (`W2M_PIN_FRACTION=0.5`) changed nothing.

## Does a higher resolution help quality?

Not for this stickman. Same four phrases and seed, compiled, at 384×672 (upscaled to 480×832), 480×832 and 576×1024
(`nvfp4_384_sharp_w2mc`, `nvfp4_480_w2mc`, `nvfp4_576_w2mc`: 18 / 26.5 / 45 s per clip). Line and hand detail look the
same at all three (the strokes are thick). Gesture correctness changes the way a different seed would rather than
improving: 384 had the cleanest thumbs up and a good peace sign; 576 blurred the thumbs up and turned the peace sign into
crossed arms. 384 stays the pick.

## What to grade

The quality question is the NVFP4 distilled model and the 384 render against the round-2 setup (`baseline`, the same
clips as `../stickman_t2v/clips/wan22_a14b_i2v_app`). Fill in `grades.csv` (1–5) for the three folders on the same five
criteria as before (gesture, readable, stickman style, body integrity, motion), plus **same as baseline?** (is the
quality loss acceptable).
