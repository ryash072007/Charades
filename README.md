# Phrase2Stickman: charades clips of one stickman

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

Needs an NVIDIA Blackwell GPU (RTX 50-series) for the NVFP4 speed-up (16 GB VRAM and 32 GB RAM or more recommended),
~20 GB of disk, Python 3.10+ and git.

1. **Install ComfyUI** (recent enough to load NVFP4 checkpoints). Either the [Windows portable build](https://github.com/comfyanonymous/ComfyUI/releases)
   or a normal checkout with its requirements installed. Run it once so the folders exist.
2. **Run the setup script.** It installs the `comfyui-w2m` node and ComfyUI-VideoHelperSuite, copies the reference image into
   ComfyUI's `input`, and downloads the models (resumable, skips what exists):
   - [LightX2V LightWan2.2 NVFP4](https://huggingface.co/lightx2v/LightWan2.2-A14B) `Wan2.2-I2V-A14B_NVFP4_Sparse_{high,low}_comfy.safetensors`
   - `umt5_xxl_fp8_e4m3fn_scaled.safetensors` and `wan_2.1_vae.safetensors` from [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
   - `lighttaew2_1.safetensors` (tiny VAE) from [lightx2v/Autoencoders](https://huggingface.co/lightx2v/Autoencoders)
3. **Generate.** Tell the scripts where ComfyUI is (the folder containing `python_embeded`, or the checkout) and run them.
   `stickman_i2v.py` starts ComfyUI itself (with `--use-sage-attention`, which the timings assume) if it isn't running, and stops it afterwards.

```powershell
pip install -r requirements.txt            # aiohttp; skip if you run the scripts with ComfyUI's embedded Python
$env:COMFYUI_DIR = "C:\path\to\ComfyUI_windows_portable"
python setup.py                            # once
python stickman_i2v.py thumbs_up
python stickman_i2v.py --mime "The stickman waves both hands overhead, then holds still." --name wave
```

**ComfyUI lifetime.** If `stickman_i2v.py` has to start ComfyUI, it stops it again when the clip is done, which frees the GPU. A ComfyUI you started yourself is never touched. For batch runs start ComfyUI yourself, or pass `--keep-alive` to leave the one the script started running.

**In the ComfyUI UI.** Drag [workflows/phrase2stickman.json](workflows/phrase2stickman.json) onto the canvas (after `setup.py`), edit the
mime sentence at the end of the green Prompt node, and press Run. It is the same graph as `stickman_i2v.py`, but saves with
ComfyUI's core Create Video + Save Video nodes. `python export_workflow.py` regenerates it (`--compile` adds the compile nodes);
it needs ComfyUI running.

Clips go to `out/<name>.mp4`. `--seed N` for another take, `--time-twice` to time a warm run, `--compile` for batch runs (below).

## Writing the prompts with a local LLM (movie charades)

A second, independent stage turns an English movie title into the prompts for the clips above. It runs
**Gemma 4 26B-A4B (QAT NVFP4 GGUF)** through [llama.cpp](https://github.com/ggml-org/llama.cpp), which this repo downloads
itself. It needs no Ollama and nothing outside the repo. On the RTX 5060 Ti the whole model sits in VRAM at 4k context:
about 117 tokens/s generating, 5,200 tokens/s on the prompt, and a plan takes about 10 s including the model load.
The scripts only write prompts and frame counts. They do not call ComfyUI.

```powershell
python llm_setup.py                          # once: llama.cpp (CUDA 13.4 build) + the 13.5 GB model, resumable
python llama_runs.py "suspicious" [--think]  # one word -> one clip -> charades_prompts.json
python charades_plan.py "Titanic" [--think]  # movie title -> clip list -> charades_plan.json
python build_movie_list.py                   # optional, ~10 min: rebuild movies.json
```

| File | What it does |
|---|---|
| `llm_setup.py` | Downloads llama.cpp into `llama.cpp/` and the model into `models/` (both gitignored). 12 parallel range requests; rerun to resume. |
| `llama_runs.py` | Starts `llama-server` once per process and exposes `chat_json` (schema-constrained output) and `generate_clip`, which makes the Klein start-frame edit, the Wan action prompt (`BASE_PROMPT` + action) and one frame count (49, 65, 81 or 97). One word in, one clip out. `--think` turns reasoning on (capped at 1,024 tokens). |
| `charades_plan.py` | Plans a whole movie as 1–4 clips chosen by popularity (below). The player already knows it is a movie, so there is no "movie" clip. Each piece goes through `generate_clip`. |
| `build_movie_list.py` | Builds `movies.json` from IMDb's non-commercial datasets (`data/`, gitignored): the 1,500 most-voted movies, rated by the LLM for English language and mimeability (1–5), keeping about 1,170. The 40 most-voted films of the last 10 years plus `MEGA_HITS` (Titanic, Iron Man, Spider-Man) are `very_popular`, the rest `popular`. `--retier` re-applies the tier rules to the existing file without re-rating. |
| `movies.json` | The result. `ollama_runs.py` and `ollama_runs_v2.py` are the older Ollama versions of `llama_runs.py`. |

**How a title is planned.** The title's popularity comes from `movies.json`; a title that isn't there is rated by one extra LLM
call (`very_popular`, `popular` or obscure). Then:
- `very_popular` (only the most popular recent films and `MEGA_HITS`, so not Shawshank): one iconic action (Iron Man firing repulsors, arms wide on a ship's bow).
- everything else (`popular`, obscure, unlisted): drop filler words and mime each meaningful word as its own clip, up to four. The schema only allows `title_words` for these tiers.
- The LLM judge's `very_popular` is demoted to `popular` unless the film is from the last 10 years or in `MEGA_HITS`.

`charades_plan.json` records `popularity` and `popularity_source` (`catalog` or `llm`). Run the scripts one at a time. Two at once
start two model servers that fight over the GPU and time out.

**Known gaps.** Catalog lookup needs the exact title (case-insensitive), so "Spiderman" falls through to the LLM rating. An
invented title can be spelled out letter by letter. The mimeability ratings are one noisy pass from a small model. The Wan
renders of these prompts have not been checked.

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
