# Stickman text-to-video: results

## Round 1 (2026-10-10): same 4 phrases, same seed, no style LoRA

Graded by the owner:

- **Wan 2.2 A14B T2V** (Q4_K_M GGUF + Lightning 4-step, about 61–71 s per 3 s clip): **easily the best.** Good enough as it is,
  except for **character consistency**: all 4 clips looked similar-ish, but not the same character (and not like
  `tpose_stickman.png`).
- **LTX-2.3 22B distilled** (Q4_K_M GGUF): **runner-up, not bad.** It **followed the prompt a bit better** than Wan
  A14B. Keep it as the alternative if Wan's consistency work stalls.
- Wan 2.2 TI2V 5B (text-only and with the reference image) and LTX-2.3 with the reference image: not usable.

Next: make Wan 2.2 A14B draw the same character every time (I2V from the reference stickman, then a character LoRA if
needed).

## Round 2 and 3

- **Wan 2.2 A14B image-to-video from a reference stickman** fixed the character-consistency problem: every clip starts
  from the same picture. Adding hands with fingers, ears and a nose to the reference (and one sentence about them in the
  prompt) improved the hands, which matter most for charades.
- Candidates and settings that were tried: Wan 2.2 TI2V 5B (20 steps, CFG 5, 24 fps), Wan 2.2 A14B T2V and I2V (Q4_K_M
  GGUF + Lightning 4-step LoRAs, CFG 1, 16 fps), LTX-2.3 22B distilled (Q4_K_M GGUF, 8 steps, 24 fps).
- Next step was speed: see [speed.md](speed.md). That takes the I2V clip from 66 s to 18 s.

`model_comparison_results.csv` has the generation times of round 1 and 2. The comparison clips were removed from the repo.
