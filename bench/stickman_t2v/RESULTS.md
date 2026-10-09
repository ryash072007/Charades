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
