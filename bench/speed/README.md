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

Tried and dropped:
- SageAttention 2 kernel variants (`nvfp4_sage_fp8`, `nvfp4_sage_fp8pp`): no gain. The default kernel is already the fastest here.
- ESRGAN anime ×4 upscale instead of Lanczos (`nvfp4_lowres`): crisper, but the upscale costs 22 s.
- torch.compile: needs Python 3.13 headers and libs that the portable install doesn't ship (only 3.12's). MSVC is now
  available to ComfyUI (`comfy_generate.msvc_env`), but the compiled kernel DLL won't load.
- SageAttention3 (FP4 attention): no wheel exists for Python 3.13 + torch 2.12 + CUDA 13, and building one hits the same headers problem.
- LightX2V's own engine (sparse attention): their published RTX 5090 numbers, scaled to this GPU and 49 frames, land
  near what ComfyUI already does here; sparse attention mainly helps at 720p and above.

## What to grade

The quality question is the NVFP4 distilled model and the 384 render against the round-2 setup (`baseline`, the same
clips as `../stickman_t2v/clips/wan22_a14b_i2v_app`). Fill in `grades.csv` (1–5) for the three folders on the same five
criteria as before (gesture, readable, stickman style, body integrity, motion), plus **same as baseline?** (is the
quality loss acceptable).
