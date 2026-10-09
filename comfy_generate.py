"""Run the fast mocap workflow (wanphrase2motion_fast.json) on a running ComfyUI server.

The mime prompt comes from mime_llm.py; this module injects it, picks the clip length, follows progress
over ComfyUI's websocket and returns the output files. Run with ComfyUI's embedded Python.
"""
import asyncio
import json
import os
import subprocess
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
FAST_WORKFLOW = os.path.join(HERE, "wanphrase2motion_fast.json")
DEFAULT_COMFY = r"C:\Users\ryash\OneDrive\Documents\AI\ComfyUI_windows_portable"
COMFY_ARGS = ["--fast", "fp16_accumulation", "--enable-manager", "--use-sage-attention"]

PROMPT_NODE, VACE_NODE, SAMPLER_NODE, DECODE_NODE = "91", "101", "117", "103"
RGB_NODE, KPS_NODE, NPZ_NODE, TINY_VAE_NODE, FULL_VAE_NODE = "200", "201", "202", "302", "97"
# node id -> stage name reported to callers
STAGES = {"114": "encode", "117": "video", "110": "video", "103": "decode", "80": "pose", "127": "body"}


def frames_for(seconds):
    """Same rule as the original workflow's math node: 17..81 frames, 4k+1, 16 fps."""
    return max(17, min(81, round(seconds * 16 / 4) * 4 + 1))


def _get(server, path, timeout=5):
    with urllib.request.urlopen(f"{server}{path}", timeout=timeout) as f:
        return json.loads(f.read())


def _post(server, path, data, timeout=30):
    r = urllib.request.Request(f"{server}{path}", data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read() or b"{}")


def comfy_up(server):
    try:
        _get(server, "/system_stats", timeout=2)
        return True
    except Exception:
        return False


def ensure_comfy(server="http://127.0.0.1:8188", portable_root=DEFAULT_COMFY, wait_s=180):
    """Start ComfyUI (same flags as the owner's launcher) if it isn't answering."""
    if comfy_up(server):
        return False
    py = os.path.join(portable_root, "python_embeded", "python.exe")
    log = open(os.path.join(HERE, "logs", "comfyui.log"), "a") if os.path.isdir(os.path.join(HERE, "logs")) else subprocess.DEVNULL
    subprocess.Popen([py, os.path.join("ComfyUI", "main.py"), *COMFY_ARGS], cwd=portable_root,
                     stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if comfy_up(server):
            print(f"[comfy] started in {time.time() - t0:.0f}s", flush=True)
            return True
        time.sleep(1)
    raise RuntimeError("ComfyUI did not start")


def free_vram(server="http://127.0.0.1:8188"):
    try:
        _post(server, "/free", {"unload_models": True, "free_memory": True})
    except Exception:
        pass


def build_workflow(prompt, seconds, job, seed=None, full_vae=False, fast_video=False, workflow=FAST_WORKFLOW):
    wf = json.load(open(workflow, encoding="utf-8"))
    if fast_video:  # the old 4-step / CFG 1 sampling: ~22 s faster, but follows hand signs much less reliably
        wf["117"]["inputs"].update(steps=4, end_at_step=2, cfg=1.0)
        wf["110"]["inputs"].update(steps=4, start_at_step=2, end_at_step=4, cfg=1.0)
        wf["111"]["inputs"]["strength_model"] = 0.8
    wf[PROMPT_NODE]["inputs"]["string_b"] = prompt
    wf[VACE_NODE]["inputs"]["length"] = frames_for(seconds)
    if seed is not None:
        wf[SAMPLER_NODE]["inputs"]["noise_seed"] = int(seed)
    if full_vae:
        wf[DECODE_NODE]["inputs"]["vae"] = [FULL_VAE_NODE, 0]
    wf[RGB_NODE]["inputs"]["filename_prefix"] = f"w2m/{job}/rgb"
    wf[KPS_NODE]["inputs"]["filename_prefix"] = f"w2m/{job}/kps"
    return wf


async def _run(server, wf, on_stage, timeout):
    import aiohttp
    cid = uuid.uuid4().hex
    ws_url = server.replace("http", "ws", 1) + f"/ws?clientId={cid}"
    t0 = time.time()
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(ws_url, max_msg_size=0, heartbeat=20) as ws:
            pid = _post(server, "/prompt", {"prompt": wf, "client_id": cid})["prompt_id"]
            while True:
                msg = await ws.receive(timeout=timeout - (time.time() - t0))
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        raise RuntimeError("ComfyUI websocket closed")
                    continue
                m = json.loads(msg.data)
                d = m.get("data", {})
                if d.get("prompt_id") not in (None, pid):
                    continue
                if m["type"] == "executing":
                    node = d.get("node")
                    if node is None:
                        break
                    if node in STAGES:
                        on_stage(STAGES[node])
                elif m["type"] == "execution_error":
                    raise RuntimeError(f"ComfyUI: {d.get('node_type')}: {d.get('exception_message', '')[:400]}")
                elif m["type"] == "execution_success":
                    break
    return pid


def generate(prompt, seconds, comfy_root, server="http://127.0.0.1:8188", seed=None, full_vae=False,
             job=None, on_stage=lambda s: None, timeout=900, fast_video=False):
    """Returns {"rgb", "kps", "npz", "frames"} for one generated clip."""
    job = job or uuid.uuid4().hex[:12]
    wf = build_workflow(prompt, seconds, job, seed, full_vae, fast_video)
    pid = asyncio.run(_run(server, wf, on_stage, timeout))
    for _ in range(50):  # the websocket can report completion a moment before history is written
        hist = _get(server, f"/history/{pid}")
        if pid in hist and hist[pid].get("outputs"):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError(f"ComfyUI history missing for {pid}")
    out = hist[pid]["outputs"]
    vid = (out[RGB_NODE].get("gifs") or out[RGB_NODE].get("videos"))[0]
    npz = out[NPZ_NODE]["text"]
    npz = npz[0] if isinstance(npz, list) else npz
    job_dir = os.path.join(comfy_root, "output", "w2m", job)
    kps = sorted(f for f in os.listdir(job_dir) if f.startswith("kps") and f.endswith(".json"))[-1]
    return {"rgb": os.path.join(comfy_root, "output", vid.get("subfolder", ""), vid["filename"]),
            "kps": os.path.join(job_dir, kps), "npz": npz, "frames": frames_for(seconds), "job": job, "dir": job_dir}


# ------------------------------------------------------------------ two-step: film takes, capture the best one
CAPTURE_NODES = ["80", "126", "127", "128", "201", "202", "300", "301"]


def _history(server, pid):
    for _ in range(50):  # the websocket can report completion a moment before history is written
        hist = _get(server, f"/history/{pid}")
        if pid in hist and hist[pid].get("outputs"):
            return hist[pid]["outputs"]
        time.sleep(0.1)
    raise RuntimeError(f"ComfyUI history missing for {pid}")


def film(prompt, seconds, comfy_root, server="http://127.0.0.1:8188", seed=None, full_vae=False, fast_video=False,
         job=None, take=1, on_stage=lambda s: None, timeout=900):
    """Generate one video take only (no capture). Returns {"rgb", "frames"}."""
    wf = build_workflow(prompt, seconds, job, seed, full_vae, fast_video)
    for k in CAPTURE_NODES:
        wf.pop(k, None)
    wf[RGB_NODE]["inputs"]["filename_prefix"] = f"w2m/{job}/take{take}"
    pid = asyncio.run(_run(server, wf, on_stage, timeout))
    vid = (_history(server, pid)[RGB_NODE].get("gifs") or [None])[0]
    return {"rgb": os.path.join(comfy_root, "output", vid.get("subfolder", ""), vid["filename"]), "frames": frames_for(seconds)}


def capture(rgb, comfy_root, server="http://127.0.0.1:8188", job=None, on_stage=lambda s: None, timeout=900):
    """DWPose + GVHMR on a saved take. Returns {"kps", "npz"}."""
    full = json.load(open(FAST_WORKFLOW, encoding="utf-8"))
    wf = {k: full[k] for k in CAPTURE_NODES}
    wf["400"] = {"class_type": "VHS_LoadVideoPath", "inputs": {"video": rgb, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                 "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1, "format": "None"}}
    wf["80"]["inputs"]["image"] = ["400", 0]
    wf["128"]["inputs"]["images"] = ["400", 0]
    wf[KPS_NODE]["inputs"]["filename_prefix"] = f"w2m/{job}/kps"
    pid = asyncio.run(_run(server, wf, on_stage, timeout))
    out = _history(server, pid)
    npz = out[NPZ_NODE]["text"]
    npz = npz[0] if isinstance(npz, list) else npz
    job_dir = os.path.join(comfy_root, "output", "w2m", job)
    kps = sorted(f for f in os.listdir(job_dir) if f.startswith("kps") and f.endswith(".json"))[-1]
    return {"kps": os.path.join(job_dir, kps), "npz": npz, "dir": job_dir}


if __name__ == "__main__":
    import argparse
    from mime_llm import MimeLLM
    ap = argparse.ArgumentParser()
    ap.add_argument("phrase")
    ap.add_argument("--comfy-root", default=os.path.join(DEFAULT_COMFY, "ComfyUI"))
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    a = ap.parse_args()
    llm = MimeLLM()
    plan = llm.plan(a.phrase)
    print(json.dumps(plan, indent=2))
    print(json.dumps(generate(plan["prompt"], plan["seconds"], a.comfy_root, a.server, on_stage=print), indent=2))
    llm.stop()
