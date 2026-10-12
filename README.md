# WANPhrase2Motion: stickman charades clips

Turns a one-sentence mime ("The stickman holds one fist out in front of the chest with the thumb pointing straight up…")
into a 3 s, 16 fps clip of the **same stickman** (`refs/relaxed_stickman_appendages.jpg`) acting it out. It is
**Wan 2.2 A14B image-to-video** in ComfyUI, tuned to about **18 s per clip** on an RTX 5060 Ti 16 GB (it started at 66 s).

| thumbs up | peace sign | teeth | eating burger |
|---|---|---|---|
| [examples/thumbs_up.mp4](examples/thumbs_up.mp4) | [examples/peace_sign.mp4](examples/peace_sign.mp4) | [examples/teeth.mp4](examples/teeth.mp4) | [examples/eating_burger.mp4](examples/eating_burger.mp4) |

## The final setup (`nvfp4_384_sharp`)

```
reference image ─► (VAE-encode once, cached) ─┐
mime text ─► UMT5 ─────────────────────────────┴─► high-noise expert (2 steps) ─► low-noise expert (2 steps)
                                                    LightX2V NVFP4, 4-step distillation built in, CFG 1
 ─► tiny VAE decode ─► Lanczos 384×672 → 480×832 ─► light unsharp mask ─► H.264 mp4
```

Time per clip for a new phrase, warm models, 49 frames:

| Change | Time |
|---|---|
| Q4_K_M GGUF experts + Lightning LoRAs, full VAE (start) | 66 s |
| tiny VAE decode (`lighttaew2_1`) | 59 s |
| start image encoded once and cached (`W2MWanImageToVideoCached`) | 55 s |
| LightX2V NVFP4 experts (Blackwell FP4 tensor cores) | 26 s |
| sample at 384×672, Lanczos to 480×832, sharpen | **18 s** |

## Setup

Needs an NVIDIA Blackwell GPU (RTX 50-series) for the NVFP4 speed-up, and a ComfyUI recent enough to load the NVFP4
checkpoints, started with `--use-sage-attention`.

1. Copy `comfy_nodes/comfyui-w2m` to `ComfyUI/custom_nodes/comfyui-w2m`. Also install `ComfyUI-VideoHelperSuite` (for the mp4 save node).
2. Models:
   - `diffusion_models/Wan2.2-I2V-A14B_NVFP4_Sparse_high_comfy.safetensors` and `…_low_comfy.safetensors` (LightX2V, LightWan2.2 NVFP4)
   - `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `vae/wan_2.1_vae.safetensors`
   - `vae_approx/lighttaew2_1.safetensors` (from `huggingface.co/lightx2v/Autoencoders`)
3. Point the scripts at ComfyUI: `$env:COMFYUI_DIR = "C:\path\to\ComfyUI_windows_portable"` (default `C:\ComfyUI_windows_portable`).
4. Run with ComfyUI's embedded Python (it already has aiohttp). The script starts ComfyUI if it isn't running and copies the reference image into `ComfyUI/input`:

```powershell
& "$env:COMFYUI_DIR\python_embeded\python.exe" stickman_i2v.py thumbs_up
& "$env:COMFYUI_DIR\python_embeded\python.exe" stickman_i2v.py --mime "The stickman waves both hands overhead, then holds still." --name wave
```

Clips go to `out/<name>.mp4`. `--seed N` for another take, `--time-twice` to time a warm run, `--compile` for batch runs (below).

## What we learned

**Writing the mime** (the only per-clip input). Ask for the most obvious charades sign: big arm and hand actions held in front
of the body facing the viewer, point at body parts (teeth = point at the mouth), do the literal hand sign (thumbs up = fist held
out in front of the chest, thumb up). Never face-only or tiny actions. End with "then holds the pose still", which gives a
clean ending. `MIMES` in `stickman_i2v.py` has four that work. They were written by an LLM from the rules above.

**Model choice** ([docs/model-comparison.md](docs/model-comparison.md)). Wan 2.2 A14B was clearly the best of Wan 5B, Wan A14B and LTX-2.3.
Text-to-video drew a slightly different character each time; image-to-video from a fixed reference solved that. LTX-2.3
followed prompts a bit better and is the fallback if Wan stalls.

**Speed** ([docs/speed.md](docs/speed.md), timings in `docs/speed_results.csv`). What worked: NVFP4 experts (2×), 384 sampling (1.4×),
tiny VAE and a cached start-image encode (a few seconds each). What didn't:
- 320×560 is faster (14 s) but gestures break (thumbs up becomes a finger gun). 480 and 576 cost 26 s and 45 s and look no better.
- SageAttention 2 kernel variants: no gain. SageAttention3: 23% faster kernel, but its preprocessing eats it, and it is less
  accurate. Rejected.
- Real-ESRGAN instead of Lanczos: crisper, costs 22 s.
- `torch.compile` (`W2MCompileWan`): works, but only ~3%, because attention and the NVFP4 matmuls are hand-written kernels.
  It pays back only after ~35 clips per ComfyUI session. KJNodes' compile node fails with dynamic VRAM.
- The remaining floor: the two NVFP4 experts (16.8 GB) don't both fit in 16 GB of VRAM, and the GPU link is PCIe 3.0 x8
  (~7 GB/s), so ~2 s per expert per clip is weight streaming. Raising ComfyUI's pinned-memory cap changed nothing.
  Further speed has to come from fewer streamed bytes, fewer tokens or frames, or the text-encoder step (2.4 s per new
  phrase), not from kernel swaps.

**`--compile` prerequisites (Windows).** Triton needs a C compiler and Python headers that match the interpreter. The ComfyUI
portable build had Python 3.13 with 3.12 `Include`/`libs`; replace them with 3.13 ones. Visual Studio's `vcvars64.bat`
environment is applied automatically by `comfy_client.msvc_env`.

## Earlier work

The first version of this repo turned a Wan 2.1 VACE clip of a person into a 3D character via GVHMR/SMPL-X, MediaPipe hands and
face, and Blender. It was replaced by the stickman approach and removed from the working tree; it is still in git history
(commit `275c600` and earlier).
