"""Generate a charades-style stickman clip with Wan 2.2 A14B image-to-video (the final `nvfp4_384_sharp` setup).

    python_embeded\\python.exe stickman_i2v.py thumbs_up
    python_embeded\\python.exe stickman_i2v.py --mime "The stickman waves both hands overhead, then holds still." --name wave
    python_embeded\\python.exe stickman_i2v.py peace_sign --compile      # batch runs: ~0.4 s/clip faster after a 16-80 s compile

The stickman is always the one in refs/relaxed_stickman_appendages.jpg, so every clip is the same character. The mime
(one or two sentences saying what the stickman does) is the only per-phrase input; see README.md for how to write one.
Clips land in out/<name>.mp4 (480x832, 16 fps, 49 frames). Run setup.py once first (installs the nodes and downloads the models); see README.md.

Pipeline: LightX2V NVFP4 experts (4-step distillation built in) -> sample at 384x672 -> tiny VAE decode ->
Lanczos up to 480x832 -> light unsharp mask.
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
import time

import comfy_client as cc

HERE = os.path.dirname(os.path.abspath(__file__))
REF = "relaxed_stickman_appendages.jpg"
SEED = 424242
OUT_W, OUT_H = 480, 832      # delivered size
SAMPLE_W, SAMPLE_H = 384, 672  # sampled size (1.25x smaller each way); 320x560 breaks gestures, 480+ doesn't look better
FRAMES, FPS = 49, 16         # 3 s

STYLE = ("2D cartoon stick figure animation on a plain white background, flat hand-drawn style, static camera. "
         "A simple stickman with a round white head with dot eyes and a small smile, spiky teal hair, thick rounded blue "
         "outline limbs, a white rectangular torso outlined in blue, teal shorts, standing in the center facing the viewer, "
         "whole body visible from head to feet. ")
APPENDAGES = ("The stickman has round blue hands with five short fingers, small round ears on both sides of the head "
              "and a small button nose. ")
NEGATIVE = ("photo, realistic, 3D render, real person, human skin, shading, gradient background, scenery, text, watermark, "
            "extra limbs, extra arms, extra hands, missing limbs, deformed, blurry, morphing, flicker, cropped, out of frame")

# Mimes that worked (written by an LLM with a "most obvious charades sign, big, in front of the body" prompt)
MIMES = {
    "thumbs_up": "The stickman holds one fist out in front of the chest with the thumb pointing straight up, nods once, then holds the sign still.",
    "teeth": "The stickman points the index finger at its mouth, then shows a wide grin with its teeth and taps the front teeth twice, then holds the pose still.",
    "eating_burger": "The stickman holds both hands together as if cradling a burger, brings the hands to the mouth, pretends to bite and chew twice, then holds the pose with hands near the mouth.",
    "peace_sign": "The stickman holds one hand out in front of the chest, palm facing the viewer, with the index and middle fingers raised in a V and the other fingers folded, then holds the sign still.",
}

HIGH = "Wan2.2-I2V-A14B_NVFP4_Sparse_high_comfy.safetensors"
LOW = "Wan2.2-I2V-A14B_NVFP4_Sparse_low_comfy.safetensors"


def node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def build_workflow(mime, seed=SEED, compile_blocks=False, prefix="stickman/out"):
    """ComfyUI API-format graph. Two samplers split the 4 distilled steps between the high- and low-noise experts."""
    high, low = ["1", 0], ["2", 0]
    if compile_blocks:  # W2MCompileWan: per-block torch.compile, ~3% faster once compiled
        high, low = ["62", 0], ["63", 0]
    wf = {
        "1": node("UNETLoader", unet_name=HIGH, weight_dtype="default"),
        "2": node("UNETLoader", unet_name=LOW, weight_dtype="default"),
        "5": node("ModelSamplingSD3", model=["1", 0], shift=5.0),
        "6": node("ModelSamplingSD3", model=["2", 0], shift=5.0),
        "7": node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default"),
        "8": node("VAELoader", vae_name="wan_2.1_vae.safetensors"),
        "50": node("VAELoader", vae_name="lighttaew2_1.safetensors"),  # tiny decoder for Wan 2.1 latents: 7.7 s -> 0.5 s
        "9": node("CLIPTextEncode", text=STYLE + APPENDAGES + mime, clip=["7", 0]),
        "10": node("CLIPTextEncode", text=NEGATIVE, clip=["7", 0]),  # ignored at CFG 1, but the node needs one
        "r1": node("LoadImage", image=REF),
        "r2": node("ImageScale", image=["r1", 0], upscale_method="lanczos", width=OUT_W, height=OUT_H, crop="center"),
        # the start image is VAE-encoded once and cached (5 s -> 0)
        "11": node("W2MWanImageToVideoCached", positive=["9", 0], negative=["10", 0], vae=["8", 0], start_image=["r2", 0],
                   width=SAMPLE_W, height=SAMPLE_H, length=FRAMES, batch_size=1),
        "12": node("KSamplerAdvanced", model=["5", 0], positive=["11", 0], negative=["11", 1], latent_image=["11", 2],
                   add_noise="enable", noise_seed=seed, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple",
                   start_at_step=0, end_at_step=2, return_with_leftover_noise="enable"),
        "13": node("KSamplerAdvanced", model=["6", 0], positive=["11", 0], negative=["11", 1], latent_image=["12", 0],
                   add_noise="disable", noise_seed=seed, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple",
                   start_at_step=2, end_at_step=4, return_with_leftover_noise="disable"),
        "14": node("VAEDecode", samples=["13", 0], vae=["50", 0]),
        "82": node("ImageScale", image=["14", 0], upscale_method="lanczos", width=OUT_W, height=OUT_H, crop="disabled"),
        "83": node("ImageSharpen", image=["82", 0], sharpen_radius=1, sigma=0.6, alpha=0.6),
        "15": node("VHS_VideoCombine", images=["83", 0], frame_rate=FPS, loop_count=0, filename_prefix=prefix,
                   format="video/h264-mp4", pix_fmt="yuv420p", crf=18, save_metadata=False, trim_to_audio=False,
                   pingpong=False, save_output=True),
    }
    if compile_blocks:
        wf["62"] = node("W2MCompileWan", model=["5", 0], mode="default", fullgraph=True)
        wf["63"] = node("W2MCompileWan", model=["6", 0], mode="default", fullgraph=True)
        wf["12"]["inputs"]["model"], wf["13"]["inputs"]["model"] = ["62", 0], ["63", 0]
    return wf


def install_reference(comfy_dir):
    dst = os.path.join(cc.comfy_paths(comfy_dir)[1], "input", REF)
    if not os.path.isfile(dst):
        shutil.copy(os.path.join(HERE, "refs", REF), dst)


def generate(mime, name, seed=SEED, compile_blocks=False, comfy_dir=cc.COMFY_DIR, server=cc.SERVER):
    """Make one clip, write out/<name>.mp4, return (path, seconds, per-node-class seconds)."""
    cc.ensure_comfy(server, comfy_dir)
    install_reference(comfy_dir)
    wf = build_workflow(mime, seed, compile_blocks)
    t0 = time.time()
    pid, times = asyncio.run(cc.run_workflow(server, wf))
    total = time.time() - t0
    out = cc.history(server, pid)
    vid = next(v for n in out.values() for v in (n.get("gifs") or []) if v["filename"].endswith(".mp4"))
    dst = os.path.join(HERE, "out", f"{name}.mp4")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(cc.comfy_paths(comfy_dir)[1], "output", vid.get("subfolder", ""), vid["filename"]), dst)
    by_class = {}
    for k, v in times.items():
        c = wf[k]["class_type"]
        by_class[c] = round(by_class.get(c, 0) + v, 1)
    return dst, total, by_class


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phrase", nargs="?", choices=sorted(MIMES), help="one of the built-in example mimes")
    ap.add_argument("--mime", help="your own description of what the stickman does")
    ap.add_argument("--name", help="output file name (default: the phrase)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--compile", action="store_true", help="torch.compile the transformer blocks (needs Triton + MSVC)")
    ap.add_argument("--time-twice", action="store_true", help="run once untimed first so model loading isn't counted")
    ap.add_argument("--comfy-dir", default=cc.COMFY_DIR)
    a = ap.parse_args()
    mime = a.mime or (MIMES[a.phrase] if a.phrase else None)
    if not mime:
        ap.error("give a built-in phrase or --mime")
    name = a.name or a.phrase or "clip"
    if a.time_twice:
        generate(mime, name, 7, a.compile, a.comfy_dir)
    path, total, parts = generate(mime, name, a.seed, a.compile, a.comfy_dir)
    print(f"[stickman] {name}: {total:.1f}s -> {os.path.relpath(path, HERE)}  {parts}", flush=True)
