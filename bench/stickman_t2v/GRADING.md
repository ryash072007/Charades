# Stickman text-to-video: grading guide

The clips are in `clips/<model>/<phrase>.mp4`. Every model got the same prompt: the stickman style description plus the
same Groq-written mime (see `MIMES` in `run_bench.py`), the same seed, and no style LoRA yet. Generation times are in
`results.csv`.

| Folder | Model | Notes |
|---|---|---|
| `wan22_5b` | Wan 2.2 TI2V 5B, fp16 | text only, 20 steps, CFG 5, 24 fps |
| `wan22_5b_ref` | Wan 2.2 TI2V 5B, fp16 | same, but starts from `tpose_stickman.png` as the first frame |
| `wan22_a14b` | Wan 2.2 A14B T2V, Q4_K_M GGUF | 4-step Lightning LoRA, 16 fps |
| `ltx23` | LTX-2.3 22B distilled 1.1, Q4_K_M GGUF | 8 distilled steps, 24 fps (audio generated and dropped) |
| `ltx23_ref` | LTX-2.3 | same, conditioned on `tpose_stickman.png` |
| `wan22_a14b_i2v` | Wan 2.2 A14B **image-to-video**, Q4_K_M GGUF | starts from `refs/relaxed_stickman.png` so every clip is the same character; I2V Lightning 4-step, 16 fps |

| `wan22_a14b_i2v_app` | Wan 2.2 A14B image-to-video | same, from `refs/relaxed_stickman_appendages.jpg` (fingers, ears, nose); prompt mentions them |

Round 2 adds `wan22_a14b_i2v`. For it, also note whether all four clips show **the same character as the reference**.

Score each clip from **1 (bad) to 5 (great)** in `grades.csv` on:

1. **Gesture**: does it do the right action with the right hand shape and in the right place? Thumb up, not a fist; the finger pointing at the mouth; the V shape for peace.
2. **Readable**: would a charades player guess the phrase from this clip alone?
3. **Stickman style**: a clean, well-defined 2D stickman on plain white, with a torso, clear limbs and head. No realistic person, 3D shading or scenery.
4. **Body integrity**: the right number of arms, hands and legs; no melting, morphing or limbs merging; the same character from start to end.
5. **Motion**: deliberate and clear (not frozen, not jittery or flickering), and the pose is held at the end.

Add a short note for anything notable, e.g. "great style but thumb missing". The style score matters less than the
others: a style LoRA trained for the chosen model will fix it. Gesture, readability, body integrity and motion come
from the model itself, so those decide which model to pick.
