"""Speed experiments for Wan 2.2 A14B image-to-video (the stickman setup the owner picked).

    python_embeded\\python.exe bench\\speed\\run_speed.py <variant> [<variant> ...] [--phrases thumbs_up,peace_sign]

Every variant runs the same phrases with the same seed. Each one is run twice: the first warms models, and the second is
timed (per node, over ComfyUI's websocket). Clips go to bench/speed/clips/<variant>/<phrase>.mp4; times are appended to
results.csv. The baseline is the round-2 setup (Q4_K_M GGUF experts + Lightning I2V LoRAs, Wan 2.1 VAE, 480x832, 49 frames).
"""
import asyncio
import csv
import json
import os
import shutil
import sys
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench", "stickman_t2v"))
import aiohttp  # noqa: E402
import comfy_generate as cg  # noqa: E402
import run_bench as rb  # noqa: E402

SERVER = "127.0.0.1:8188"
COMFY = os.path.join(cg.DEFAULT_COMFY, "ComfyUI")


def node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}


def baseline(prompt):
    return rb.MODELS["wan22_a14b_i2v_app"](prompt)


def tae(wf):
    """Tiny VAE decode (lighttaew2_1 decodes Wan 2.1 latents) instead of the full VAE."""
    wf["50"] = node("VAELoader", vae_name="lighttaew2_1.safetensors")
    wf["14"]["inputs"]["vae"] = ["50", 0]
    return wf


def nvfp4(wf, which="lightwan"):
    """LightX2V's NVFP4 experts with the 4-step distillation built in: plain UNETLoader, no Lightning LoRA."""
    names = {"lightwan": ("Wan2.2-I2V-A14B_NVFP4_Sparse_high_comfy.safetensors", "Wan2.2-I2V-A14B_NVFP4_Sparse_low_comfy.safetensors")}[which]
    wf["1"] = node("UNETLoader", unet_name=names[0], weight_dtype="default")
    wf["2"] = node("UNETLoader", unet_name=names[1], weight_dtype="default")
    wf["5"]["inputs"]["model"] = ["1", 0]
    wf["6"]["inputs"]["model"] = ["2", 0]
    wf.pop("3", None)
    wf.pop("4", None)
    return wf


def cached_i2v(wf):
    """Encode the reference start image once (comfyui-w2m W2MWanImageToVideoCached) instead of on every phrase."""
    i = wf["11"]["inputs"]
    wf["11"] = node("W2MWanImageToVideoCached", positive=i["positive"], negative=i["negative"], vae=i["vae"],
                    start_image=i["start_image"], width=i["width"], height=i["height"], length=i["length"], batch_size=1)
    return wf


def compiled(wf):
    """torch.compile (inductor) on the transformer blocks of both experts; compiled once, reused while the shape stays the same."""
    for src, dst in (("5", "60"), ("6", "61")):
        wf[dst] = node("TorchCompileModelWanVideoV2", model=[src, 0], backend="inductor", fullgraph=False, mode="default",
                       dynamic=False, compile_transformer_blocks_only=True, dynamo_cache_size_limit=64)
    wf["12"]["inputs"]["model"] = ["60", 0]
    wf["13"]["inputs"]["model"] = ["61", 0]
    return wf


def w2m_compiled(wf, mode="default", attention="default", compile=True):
    """torch.compile per block via comfyui-w2m W2MCompileWan: NVFP4 Linears and attention stay eager behind custom ops,
    so it works with dynamic VRAM and every block is one graph."""
    for src, dst in (("5", "62"), ("6", "63")):
        wf[dst] = node("W2MCompileWan", model=[src, 0], mode=mode, fullgraph=True, compile=compile, attention=attention)
    wf["12"]["inputs"]["model"] = ["62", 0]
    wf["13"]["inputs"]["model"] = ["63", 0]
    return wf


def sage(wf, mode):
    """KJNodes sage attention patch on both experts (overrides the global --use-sage-attention kernel choice)."""
    for src, dst in (("5", "70"), ("6", "71")):
        wf[dst] = node("PathchSageAttentionKJ", model=[src, 0], sage_attention=mode)
    wf["12"]["inputs"]["model"] = ["70", 0]
    wf["13"]["inputs"]["model"] = ["71", 0]
    return wf


def lowres(wf, w=384, h=672):
    """Sample at a lower resolution, decode, then upscale (Real-ESRGAN anime x4 -> resize) back to 480x832."""
    wf["11"]["inputs"]["width"], wf["11"]["inputs"]["height"] = w, h
    wf["80"] = node("UpscaleModelLoader", model_name="RealESRGAN_x4plus_anime_6B.pth")
    wf["81"] = node("ImageUpscaleWithModel", upscale_model=["80", 0], image=["14", 0])
    wf["82"] = node("ImageScale", image=["81", 0], upscale_method="lanczos", width=rb.W, height=rb.H, crop="disabled")
    wf["15"]["inputs"]["images"] = ["82", 0]
    return wf


def lowres_lanczos(wf, w=384, h=672):
    """Sample at a lower resolution, decode, then a plain Lanczos resize back to 480x832 (1.25x)."""
    wf["11"]["inputs"]["width"], wf["11"]["inputs"]["height"] = w, h
    wf["82"] = node("ImageScale", image=["14", 0], upscale_method="lanczos", width=rb.W, height=rb.H, crop="disabled")
    wf["15"]["inputs"]["images"] = ["82", 0]
    return wf


def render_res(wf, w, h):
    """Sample and output at w x h (no resize)."""
    wf["11"]["inputs"]["width"], wf["11"]["inputs"]["height"] = w, h
    return wf


def sharpen(wf):
    """A light unsharp mask after the resize, to bring back line crispness."""
    src = wf["15"]["inputs"]["images"]
    wf["83"] = node("ImageSharpen", image=src, sharpen_radius=1, sigma=0.6, alpha=0.6)
    wf["15"]["inputs"]["images"] = ["83", 0]
    return wf


VARIANTS = {
    "nvfp4_384_sharp_w2mc": lambda p: w2m_compiled(sharpen(lowres_lanczos(cached_i2v(tae(nvfp4(baseline(p))))))),
    "nvfp4_384_sharp_sage3": lambda p: w2m_compiled(sharpen(lowres_lanczos(cached_i2v(tae(nvfp4(baseline(p)))))), attention="sage3", compile=False),
    "nvfp4_384_sharp_sage3pbm": lambda p: w2m_compiled(sharpen(lowres_lanczos(cached_i2v(tae(nvfp4(baseline(p)))))), attention="sage3_per_block_mean", compile=False),
    "nvfp4_384_sharp_w2mc_sage3": lambda p: w2m_compiled(sharpen(lowres_lanczos(cached_i2v(tae(nvfp4(baseline(p)))))), attention="sage3"),
    "nvfp4_384_sharp_w2mc_sage3pbm": lambda p: w2m_compiled(sharpen(lowres_lanczos(cached_i2v(tae(nvfp4(baseline(p)))))), attention="sage3_per_block_mean"),
    "nvfp4_480_w2mc": lambda p: w2m_compiled(cached_i2v(tae(nvfp4(baseline(p))))),
    "nvfp4_576_w2mc": lambda p: w2m_compiled(render_res(cached_i2v(tae(nvfp4(baseline(p)))), 576, 1024)),
    "nvfp4_384_sharp_w2mc_at": lambda p: w2m_compiled(sharpen(lowres_lanczos(cached_i2v(tae(nvfp4(baseline(p)))))), "max-autotune-no-cudagraphs"),
    "nvfp4_384_sharp_compile": lambda p: compiled(sharpen(lowres_lanczos(cached_i2v(tae(nvfp4(baseline(p))))))),
    "nvfp4_384_sharp": lambda p: sharpen(lowres_lanczos(cached_i2v(tae(nvfp4(baseline(p)))))),
    "nvfp4_320_sharp": lambda p: sharpen(lowres_lanczos(cached_i2v(tae(nvfp4(baseline(p)))), 320, 560)),
    "nvfp4_lowres_lanczos": lambda p: lowres_lanczos(cached_i2v(tae(nvfp4(baseline(p))))),
    "nvfp4_lowres": lambda p: lowres(cached_i2v(tae(nvfp4(baseline(p))))),
    "nvfp4_sage_fp8pp": lambda p: sage(cached_i2v(tae(nvfp4(baseline(p)))), "sageattn_qk_int8_pv_fp8_cuda++"),
    "nvfp4_sage_fp8": lambda p: sage(cached_i2v(tae(nvfp4(baseline(p)))), "sageattn_qk_int8_pv_fp8_cuda"),
    "nvfp4_cache_tae_compile": lambda p: compiled(cached_i2v(tae(nvfp4(baseline(p))))),
    "cache_tae": lambda p: cached_i2v(tae(baseline(p))),
    "nvfp4_cache_tae": lambda p: cached_i2v(tae(nvfp4(baseline(p)))),
    "baseline": lambda p: baseline(p),
    "tae": lambda p: tae(baseline(p)),
    "nvfp4": lambda p: nvfp4(baseline(p)),
    "nvfp4_tae": lambda p: tae(nvfp4(baseline(p))),
}


async def _run(wf):
    cid = uuid.uuid4().hex
    times, cur, ct, steps = {}, None, None, []
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(f"ws://{SERVER}/ws?clientId={cid}", max_msg_size=0) as ws:
            r = urllib.request.Request(f"http://{SERVER}/prompt", data=json.dumps({"prompt": wf, "client_id": cid}).encode(),
                                       headers={"Content-Type": "application/json"})
            pid = json.loads(urllib.request.urlopen(r).read())["prompt_id"]
            t0 = ct = time.time()
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data)
                d = m.get("data", {})
                if d.get("prompt_id") not in (None, pid):
                    continue
                if m["type"] == "executing":
                    now = time.time()
                    if cur:
                        times[cur] = times.get(cur, 0) + now - ct
                    cur, ct = d.get("node"), now
                    if cur is None:
                        break
                elif m["type"] == "progress":
                    steps.append((d.get("node"), d.get("value"), round(time.time() - ct, 2)))
                elif m["type"] == "execution_error":
                    raise RuntimeError(f"{d.get('node_type')}: {d.get('exception_message', '')[:500]}")
    if os.environ.get("W2M_STEPS"):
        print("[steps] (node, step, s since node start)", steps, flush=True)
    return pid, times, time.time() - t0


def run(variant, phrase, warm=True):
    wf = VARIANTS[variant](rb.STYLE + rb.APPENDAGES + rb.MIMES[phrase])
    if warm:  # untimed pass with another seed so model loading isn't counted
        w = json.loads(json.dumps(wf))
        for n in w.values():
            if "noise_seed" in n["inputs"]:
                n["inputs"]["noise_seed"] = 7
        asyncio.run(_run(w))
    pid, times, total = asyncio.run(_run(wf))
    out = cg._history(f"http://{SERVER}", pid)
    vid = next(v for n in out.values() for v in (n.get("gifs") or []) if v["filename"].endswith(".mp4"))
    dst = os.path.join(HERE, "clips", variant, f"{phrase}.mp4")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(COMFY, "output", vid.get("subfolder", ""), vid["filename"]), dst)
    names = {k: v["class_type"] for k, v in wf.items()}
    parts = {}
    for k, v in times.items():
        key = names.get(k, k)
        parts[key] = round(parts.get(key, 0) + v, 1)
    new = not os.path.isfile(os.path.join(HERE, "results.csv"))
    with open(os.path.join(HERE, "results.csv"), "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["when", "variant", "phrase", "total_s", "breakdown"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), variant, phrase, f"{total:.1f}", json.dumps(parts)])
    print(f"[speed] {variant:14s} {phrase:12s} {total:6.1f}s  {parts}  per-node {dict((k, round(v, 1)) for k, v in times.items())}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    phrases = ["thumbs_up", "peace_sign"]
    if "--phrases" in sys.argv:
        phrases = sys.argv[sys.argv.index("--phrases") + 1].split(",")
        args = [a for a in args if a != sys.argv[sys.argv.index("--phrases") + 1]]
    cg.ensure_comfy(f"http://{SERVER}")
    for v in args:
        for i, p in enumerate(phrases):
            run(v, p, warm=(i == 0))
