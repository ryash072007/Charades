# WANPhrase2Motion: clip → 3D character

This turns one generated charades clip into a soft-vinyl cartoon character that performs the same mime. The output is a 16 fps MP4 and an animated GLB.

```
phrase ─► Ollama ─► Wan 2.1 VACE clip ─┬─► DWPose keypoints ─┐
                                        ├─► GVHMR SMPL params ├─► extract_hands_face.py ─► prepare_motion.py ─► Blender ─► MP4 + GLB
                                        └─► RGB frames ───────┘   (MediaPipe hands/face)     (SMPL FK, grounding)   (EEVEE)
```

| Part | Source |
|---|---|
| Body + root motion | GVHMR `smpl_params_*.npz`, SMPL forward kinematics with the clip's betas |
| Fingers | MediaPipe HandLandmarker on crops around each DWPose wrist (3D world landmarks). Falls back to DWPose 2D hand keypoints, then gap interpolation. Converted to curl/spread angles per joint |
| Face | MediaPipe FaceLandmarker blendshapes, used for: `jawOpen, mouthSmile, mouthFrown, browInnerUp, browDown, eyeBlink, eyeWide` |
| Smoothing | Zero-lag (two-pass) One Euro filter on finger angles and face channels |
| Character | Procedural: tapered capsule limbs, pill torso skinned along the spine, mitten hands with 3-bone capsule fingers, round head with ink eyes/brows/mouth driven by shape keys |

## Fast path (default) and the phrase worker

`run_pipeline.py --phrase` and `w2m_worker.py` use **`wanphrase2motion_fast.json`**. It differs from the mocap workflow in four ways:
- the LLM runs outside ComfyUI
- SAM3 is replaced by a box mask from DWPose
- the blank control video's VAE encoding is cached
- decoding uses the tiny `lighttaew2_1` VAE

The default output is the charades-site clip (`.bin` + `.w2m.bin` fingers/face sidecar); the Blender MP4 and GLB are opt-in (`--mp4`, `--glb`).

Time per 49-frame clip on the RTX 5060 Ti, measured on 2026-10-09/10:

| Stage | Before | After | Change |
|---|---|---|---|
| Mime LLM | 6–9 s (Ollama, model reloaded every run, thinking on) | ~1 s (resident llama-server; ~5 s on first use) | `mime_llm.py` |
| Text encode | 2.5–5.8 s | 2–5 s | unchanged (more variable when VRAM is tight) |
| VACE conditioning | 9.5–10 s | ~0 s (4.6 s on first run) | cached blank-control latent; frames bit-identical |
| Sampling | 9.5 s | 9.4 s | unchanged |
| VAE decode | 8–12.6 s | 0.8 s | tiny VAE; GVHMR joints within 2 mm, fingers/face within MediaPipe noise |
| DWPose | 2.3 s | 2.3 s | unchanged |
| SAM3 mask | 21 s | ~0 s | DWPose box; GVHMR joints within 6 mm (p95 15 mm) |
| GVHMR | 10–10.6 s | 8.5–10 s | unchanged |
| Preview nodes / saves | ~2 s | 0.3 s | removed |
| Hands + face | 1.7 s | 1.2–2.4 s | unchanged |
| Blender MP4 + ffmpeg | ~40 s | 0 s (opt-in, ~35 s) | site plays the motion in three.js |
| **Total** | **~2 min** | **26–28 s warm, ~31 s through the site, 43 s cold, instant if cached** | |

Measured on 2026-10-09: the 12-minute run reported earlier didn't reproduce; the same command took 43 s after ComfyUI. The first run was most likely slowed by something else using the GPU.

The worker (the site starts it automatically):

```powershell
& $PY w2m_worker.py --clips-dir <site>\data\mocap\clips --cache-dir <site>\data\w2m-wan   # port 8767
```

- `POST /generate {"text"}` streams NDJSON stages and ends with `{"done": true, "id", ...}`.
- `GET /cached?text=` checks the cache without GPU work.
- `POST /unload` frees the GPU (ComfyUI `/free` plus stopping the LLM).
- It starts ComfyUI with your usual flags if it isn't running.
- It frees the GPU after 5 idle minutes.
- Results are cached per normalised phrase. Bump `PIPELINE_VERSION` to invalidate them.

Install the custom nodes once (a new folder; no existing node changes): copy `comfy_nodes\comfyui-w2m` to `ComfyUI\custom_nodes\comfyui-w2m` and restart ComfyUI. The tiny VAE goes in `ComfyUI\models\vae_approx\lighttaew2_1.safetensors` (from `huggingface.co/lightx2v/Autoencoders`). The LLM uses `C:\qd\llama-g4\build\bin\llama-server.exe` (override with `W2M_LLAMA_SERVER`) on the GGUF that Ollama already downloaded; it falls back to Ollama if llama-server is missing.

## Requirements

- ComfyUI portable at `C:\Users\ryash\OneDrive\Documents\AI\ComfyUI_windows_portable` with the custom nodes already installed (`comfyui-motioncapture`, `comfyui_controlnet_aux`, `comfyui-videohelpersuite`, `comfyui-ollama`, SAM3). No new pip packages are needed: `mediapipe`, `av`, `scipy` and `imageio-ffmpeg` are already in `python_embeded`.
- MediaPipe models in `ComfyUI/models/mediapipe/`: `hand_landmarker.task` and `face_landmarker.task` (already downloaded). If you need them again, download from:
  - https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  - https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- Blender 5.2 at `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe` (override with `--blender`).

## Workflow

Use **`wanphrase2motion_mocap.json`**. It's a copy of `wanphrase2motion.json`, which is unchanged, with these additions:

- node 200 saves the **RGB clip** to `output/w2m/clip_rgb_XXXXX.mp4`. The original only saved the DWPose skeleton render.
- node 201 saves the **DWPose keypoints** to `output/w2m/clip_kps_XXXXX.json`.
- node 202 shows the GVHMR npz path as text, so the API can report it.
- the skeleton video now saves as `output/w2m/clip_dwpose_XXXXX.mp4`.

## Running

All commands are run from this folder using ComfyUI's embedded Python:

```powershell
$PY = "C:\Users\ryash\OneDrive\Documents\AI\ComfyUI_windows_portable\python_embeded\python.exe"

# A) generate a new clip end-to-end (ComfyUI on :8188 and Ollama must be running)
& $PY run_pipeline.py --phrase "eating burger" --name eating_burger

# B) re-process a clip the mocap workflow already produced
$OUT = "C:\Users\ryash\OneDrive\Documents\AI\ComfyUI_windows_portable\ComfyUI\output"
& $PY run_pipeline.py --video "$OUT\w2m\clip_rgb_00001.mp4" --kps "$OUT\w2m\clip_kps_00001.json" `
                      --npz "$OUT\smpl_params_0004.npz" --name eating_burger
```

Results go to `ComfyUI/output/w2m/<name>/`:

- `<name>.mp4`: the render (720×1248, 16 fps, H.264)
- `<name>_compare.mp4`: the source clip and the render side by side
- `<name>.glb`: the skinned character with skeletal animation and face morph-target animation
- `hands_face.npz`, `motion.npz`: intermediate data

### Example input prompts

`--phrase` is the charades word or title, not a video prompt. The Ollama node turns it into a mime description and decides the clip length (1–5 s). Phrases with a clear, full-body action read best. Hand gestures and facial expressions come through as well, but they're small on screen.

```powershell
# everyday actions: clear arm and hand motion
& $PY run_pipeline.py --phrase "eating burger"         --name eating_burger
& $PY run_pipeline.py --phrase "brushing teeth"        --name brushing_teeth
& $PY run_pipeline.py --phrase "drinking coffee"       --name drinking_coffee
& $PY run_pipeline.py --phrase "typing on a laptop"    --name typing
& $PY run_pipeline.py --phrase "talking on the phone"  --name phone_call

# whole-body motion: tests root motion and grounding
& $PY run_pipeline.py --phrase "swimming"              --name swimming
& $PY run_pipeline.py --phrase "playing tennis"        --name tennis
& $PY run_pipeline.py --phrase "climbing a ladder"     --name ladder
& $PY run_pipeline.py --phrase "skipping rope" --ground global --name skipping   # allows jumps

# emotions and faces: tests the face and brows
& $PY run_pipeline.py --phrase "surprised"             --name surprised
& $PY run_pipeline.py --phrase "crying baby"           --name crying_baby
& $PY run_pipeline.py --phrase "sneezing"              --name sneezing

# hand-heavy: tests finger curl and spread
& $PY run_pipeline.py --phrase "playing piano"         --name piano
& $PY run_pipeline.py --phrase "counting money"        --name counting_money
& $PY run_pipeline.py --phrase "thumbs up"             --name thumbs_up

# movie titles: the LLM picks an iconic action to act out
& $PY run_pipeline.py --phrase "Titanic"               --name titanic
& $PY run_pipeline.py --phrase "Spider-Man"            --name spiderman
& $PY run_pipeline.py --phrase "The Karate Kid"        --name karate_kid
& $PY run_pipeline.py --phrase "Jaws"                  --name jaws
```

Tips:
- Each run produces a new clip. The LLM seed and the second sampler's noise seed are fixed in the workflow, so re-running the same phrase usually gives a very similar clip. To get a different take, change `noise_seed` on node 117 in `wanphrase2motion_mocap.json`.
- When a hand or the mouth is hidden (e.g. behind a prop the actor pretends to hold), the pipeline fills in from nearby frames. The face shows what's visible, so a mouth hidden behind a burger won't read as chewing.
- Actions that turn the body sideways or away from the camera (e.g. "swimming", "climbing a ladder") give weaker finger and face tracking, but the body motion stays good.
- To try a look without waiting for a full render, preview one frame with `--samples 16`, or use the single-frame command in the table below.

Useful options: `--color "#f2a7b5"` (body colour), `--width/--height`, `--samples` (EEVEE quality vs. speed), `--ground global` (allows jumps instead of pinning a foot to the floor every frame), `--no-glb`, `--no-compare`, `--keep-frames`.

Rendering takes about 10–20 s per frame on EEVEE, so roughly 10–15 minutes for a 49-frame clip. For quicker previews, lower `--samples` to 16.

## Files

| File | What it does |
|---|---|
| `run_pipeline.py` | Single entry point: generate (optional) → extract → prepare → Blender → ffmpeg |
| `comfy_generate.py` | Queues `wanphrase2motion_mocap.json` through the ComfyUI API and collects the output paths |
| `extract_hands_face.py` | Hand crops + MediaPipe/DWPose fusion → finger angles; face blendshapes; One Euro smoothing |
| `prepare_motion.py` | SMPL FK, turns the figure to face the camera, centres it, grounds the feet → `motion.npz` |
| `blender_character.py` | Builds the rig and character, keys the animation, sets up the studio and camera, renders, exports the GLB. Run it alone with `blender -b --python blender_character.py -- --motion motion.npz --still 24 --still-out test.png` to preview one frame |

## Notes and limits

- Frames are aligned by index. GVHMR, DWPose and the RGB save all consume the same VAE Decode output, so frame *i* is the same image in all three.
- The figure is turned so its average facing points at the camera, and centred on its mean pelvis position. GVHMR's root translation is otherwise kept.
- Finger curls use unsigned joint angles, so hyperextension reads as straight. Thumb opposition is approximate.
- Face expressions are deliberately exaggerated. The input→output ranges are in `FACE_MAP` in `blender_character.py`.
- The pack's `SMPL to Mixamo` node isn't used: it only moves the hips and doesn't transfer joint rotations.
