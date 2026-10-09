"""Stickman text-to-video model comparison: same prompts on each candidate model, clips saved here for grading.

    python_embeded\\python.exe bench\\stickman_t2v\\run_bench.py <model> [<model> ...]

Models: wan22_5b, wan22_5b_ref (first frame = tpose_stickman.png), wan22_a14b, ltx23.
Writes bench/stickman_t2v/clips/<model>/<phrase>.mp4 and appends timings to results.csv.
Needs ComfyUI running (comfy_generate.ensure_comfy starts it).
"""
import csv
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import comfy_generate as cg  # noqa: E402

COMFY = os.path.join(cg.DEFAULT_COMFY, "ComfyUI")
SERVER = "http://127.0.0.1:8188"
SEED = 424242

STYLE = ("2D cartoon stick figure animation on a plain white background, flat hand-drawn style, static camera. "
         "A simple stickman with a round white head with dot eyes and a small smile, spiky teal hair, thick rounded blue "
         "outline limbs, a white rectangular torso outlined in blue, teal shorts, standing in the center facing the viewer, "
         "whole body visible from head to feet. ")
NEGATIVE = ("photo, realistic, 3D render, real person, human skin, shading, gradient background, scenery, text, watermark, "
            "extra limbs, extra arms, extra hands, missing limbs, deformed, blurry, morphing, flicker, cropped, out of frame")
# The mimes Groq wrote for these phrases with the charades-first prompt (fixed so every model gets the same text)
MIMES = {
    "thumbs_up": "The stickman holds one fist out in front of the chest with the thumb pointing straight up, nods once, then holds the sign still.",
    "teeth": "The stickman points the index finger at its mouth, then shows a wide grin with its teeth and taps the front teeth twice, then holds the pose still.",
    "eating_burger": "The stickman holds both hands together as if cradling a burger, brings the hands to the mouth, pretends to bite and chew twice, then holds the pose with hands near the mouth.",
    "peace_sign": "The stickman holds one hand out in front of the chest, palm facing the viewer, with the index and middle fingers raised in a V and the other fingers folded, then holds the sign still.",
}
W, H = 480, 832


def node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def wan22_5b(prompt, ref=False):
    wf = {
        "1": node("UNETLoader", unet_name="wan2.2_ti2v_5B_fp16.safetensors", weight_dtype="default"),
        "2": node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default"),
        "3": node("VAELoader", vae_name="wan2.2_vae.safetensors"),
        "4": node("CLIPTextEncode", text=prompt, clip=["2", 0]),
        "5": node("CLIPTextEncode", text=NEGATIVE, clip=["2", 0]),
        "6": node("Wan22ImageToVideoLatent", vae=["3", 0], width=W, height=H, length=73, batch_size=1),
        "7": node("ModelSamplingSD3", model=["1", 0], shift=8.0),
        "8": node("KSampler", model=["7", 0], positive=["4", 0], negative=["5", 0], latent_image=["6", 0], seed=SEED,
                  steps=20, cfg=5.0, sampler_name="uni_pc", scheduler="simple", denoise=1.0),
        "9": node("VAEDecode", samples=["8", 0], vae=["3", 0]),
        "10": node("VHS_VideoCombine", images=["9", 0], frame_rate=24, loop_count=0, filename_prefix="bench/out",
                   format="video/h264-mp4", pix_fmt="yuv420p", crf=18, save_metadata=False, trim_to_audio=False,
                   pingpong=False, save_output=True),
    }
    if ref:
        wf["11"] = node("LoadImage", image="tpose_stickman.png")
        wf["12"] = node("ImageScale", image=["11", 0], upscale_method="lanczos", width=W, height=H, crop="center")
        wf["6"]["inputs"]["start_image"] = ["12", 0]
    return wf


def save(images, fps):
    return node("VHS_VideoCombine", images=images, frame_rate=fps, loop_count=0, filename_prefix="bench/out",
                format="video/h264-mp4", pix_fmt="yuv420p", crf=18, save_metadata=False, trim_to_audio=False,
                pingpong=False, save_output=True)


def ref_image():
    return {"r1": node("LoadImage", image="tpose_stickman.png"),
            "r2": node("ImageScale", image=["r1", 0], upscale_method="lanczos", width=W, height=H, crop="center")}


def wan22_a14b(prompt):
    """Wan 2.2 A14B T2V, Q4_K_M GGUF experts + Lightning 4-step LoRAs (2 steps high noise, 2 low), CFG 1, 16 fps."""
    wf = {
        "1": node("UnetLoaderGGUF", unet_name="Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"),
        "2": node("UnetLoaderGGUF", unet_name="Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"),
        "3": node("LoraLoaderModelOnly", model=["1", 0], lora_name="wan22_t2v_a14b_lightning_seko_v2_high.safetensors", strength_model=1.0),
        "4": node("LoraLoaderModelOnly", model=["2", 0], lora_name="wan22_t2v_a14b_lightning_seko_v2_low.safetensors", strength_model=1.0),
        "5": node("ModelSamplingSD3", model=["3", 0], shift=5.0),
        "6": node("ModelSamplingSD3", model=["4", 0], shift=5.0),
        "7": node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default"),
        "8": node("VAELoader", vae_name="wan_2.1_vae.safetensors"),
        "9": node("CLIPTextEncode", text=prompt, clip=["7", 0]),
        "10": node("CLIPTextEncode", text=NEGATIVE, clip=["7", 0]),
        "11": node("EmptyHunyuanLatentVideo", width=W, height=H, length=49, batch_size=1),
        "12": node("KSamplerAdvanced", model=["5", 0], positive=["9", 0], negative=["10", 0], latent_image=["11", 0],
                   add_noise="enable", noise_seed=SEED, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple",
                   start_at_step=0, end_at_step=2, return_with_leftover_noise="enable"),
        "13": node("KSamplerAdvanced", model=["6", 0], positive=["9", 0], negative=["10", 0], latent_image=["12", 0],
                   add_noise="disable", noise_seed=SEED, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple",
                   start_at_step=2, end_at_step=4, return_with_leftover_noise="disable"),
        "14": node("VAEDecode", samples=["13", 0], vae=["8", 0]),
    }
    wf["15"] = save(["14", 0], 16)
    return wf


def ltx23(prompt, ref=False):
    """LTX-2.3 22B distilled 1.1, Q4_K_M GGUF, Gemma 3 12B fp4 text encoder; 8 distilled steps, CFG 1, 24 fps.
    Audio is generated alongside (the model is audio+video) and dropped."""
    wf = {
        "1": node("UnetLoaderGGUF", unet_name="ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"),
        "2": node("DualCLIPLoader", clip_name1="gemma_3_12B_it_fp4_mixed.safetensors",
                  clip_name2="ltx-2.3_text_projection_bf16.safetensors", type="ltxv", device="default"),
        "3": node("VAELoader", vae_name="LTX23_video_vae_bf16.safetensors"),
        "4": node("VAELoader", vae_name="LTX23_audio_vae_bf16.safetensors"),
        "5": node("CLIPTextEncode", text=prompt, clip=["2", 0]),
        "6": node("CLIPTextEncode", text=NEGATIVE, clip=["2", 0]),
        "7": node("LTXVConditioning", positive=["5", 0], negative=["6", 0], frame_rate=24.0),
        "8": node("EmptyLTXVLatentVideo", width=W, height=H, length=73, batch_size=1),
        "9": node("LTXVEmptyLatentAudio", frames_number=73, frame_rate=24, batch_size=1, audio_vae=["4", 0]),
        "10": node("LTXVConcatAVLatent", video_latent=["8", 0], audio_latent=["9", 0]),
        "11": node("RandomNoise", noise_seed=SEED),
        "12": node("CFGGuider", model=["1", 0], positive=["7", 0], negative=["7", 1], cfg=1.0),
        "13": node("KSamplerSelect", sampler_name="euler_ancestral_cfg_pp"),
        "14": node("ManualSigmas", sigmas="1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"),
        "15": node("SamplerCustomAdvanced", noise=["11", 0], guider=["12", 0], sampler=["13", 0], sigmas=["14", 0], latent_image=["10", 0]),
        "16": node("LTXVSeparateAVLatent", av_latent=["15", 0]),
        "17": node("LTXVTiledVAEDecode", vae=["3", 0], latents=["16", 0], horizontal_tiles=2, vertical_tiles=2, overlap=6, last_frame_fix=False),
    }
    wf["18"] = save(["17", 0], 24)
    if ref:
        wf.update(ref_image())
        wf["r3"] = node("LTXVPreprocess", image=["r2", 0], img_compression=18)
        wf["r4"] = node("LTXVImgToVideoConditionOnly", vae=["3", 0], image=["r3", 0], latent=["8", 0], strength=0.7, bypass=False)
        wf["10"]["inputs"]["video_latent"] = ["r4", 0]
    return wf


def wan22_a14b_i2v(prompt, start="relaxed_stickman.png"):
    """Wan 2.2 A14B image-to-video from the reference stickman (same character every clip), Q4_K_M GGUF + I2V
    Lightning 4-step (2 high noise, 2 low), CFG 1, 16 fps."""
    wf = wan22_a14b(prompt)
    wf["1"]["inputs"]["unet_name"] = "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"
    wf["2"]["inputs"]["unet_name"] = "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
    wf["3"]["inputs"]["lora_name"] = "wan22_i2v_a14b_lightning_seko_v1_high.safetensors"
    wf["4"]["inputs"]["lora_name"] = "wan22_i2v_a14b_lightning_seko_v1_low.safetensors"
    wf.update(ref_image())
    wf["r1"]["inputs"]["image"] = start
    wf["11"] = node("WanImageToVideo", positive=["9", 0], negative=["10", 0], vae=["8", 0], width=W, height=H,
                    length=49, batch_size=1, start_image=["r2", 0])
    for k in ("12", "13"):
        wf[k]["inputs"]["positive"] = ["11", 0]
        wf[k]["inputs"]["negative"] = ["11", 1]
    wf["12"]["inputs"]["latent_image"] = ["11", 2]
    return wf


APPENDAGES = ("The stickman has round blue hands with five short fingers, small round ears on both sides of the head "
              "and a small button nose. ")

MODELS = {
    "wan22_a14b_i2v": wan22_a14b_i2v,
    # the owner's improved reference: same stickman with hands, fingers, ears and a nose
    "wan22_a14b_i2v_app": lambda p: wan22_a14b_i2v(p.replace(STYLE, STYLE + APPENDAGES), start="relaxed_stickman_appendages.jpg"),
    "wan22_5b": lambda p: wan22_5b(p),
    "wan22_5b_ref": lambda p: wan22_5b(p, ref=True),
    "wan22_a14b": wan22_a14b,
    "ltx23": lambda p: ltx23(p),
    "ltx23_ref": lambda p: ltx23(p, ref=True),
}


def run(model, phrase, mime):
    import asyncio
    wf = MODELS[model](STYLE + mime)
    t0 = time.time()
    pid = asyncio.run(cg._run(SERVER, wf, lambda s: None, 3600))
    dt = time.time() - t0
    out = cg._history(SERVER, pid)
    vid = next(v for n in out.values() for v in (n.get("gifs") or []) if v["filename"].endswith(".mp4"))
    src = os.path.join(COMFY, "output", vid.get("subfolder", ""), vid["filename"])
    dst_dir = os.path.join(HERE, "clips", model)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{phrase}.mp4")
    shutil.copy(src, dst)
    with open(os.path.join(HERE, "results.csv"), "a", newline="") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M"), model, phrase, f"{dt:.1f}", os.path.relpath(dst, HERE)])
    print(f"[bench] {model:14s} {phrase:14s} {dt:6.1f}s -> {os.path.relpath(dst, HERE)}", flush=True)


if __name__ == "__main__":
    cg.ensure_comfy(SERVER)
    if not os.path.isfile(os.path.join(HERE, "results.csv")):
        with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
            csv.writer(f).writerow(["when", "model", "phrase", "seconds", "clip"])
    for m in sys.argv[1:]:
        for phrase, mime in MIMES.items():
            run(m, phrase, mime)
