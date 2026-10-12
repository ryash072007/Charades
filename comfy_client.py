"""Minimal ComfyUI client: start the server if needed, queue an API-format workflow, follow it over the websocket.

Run with ComfyUI's embedded Python (it has aiohttp). Set COMFYUI_DIR to your ComfyUI_windows_portable folder.
"""
import asyncio
import json
import os
import subprocess
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY_DIR = os.environ.get("COMFYUI_DIR", r"C:\ComfyUI_windows_portable")
SERVER = "http://127.0.0.1:8188"
# --use-sage-attention (SageAttention 2) is what the speed numbers assume; --fast fp16_accumulation is a small extra
COMFY_ARGS = ["--fast", "fp16_accumulation", "--use-sage-attention"] + os.environ.get("W2M_COMFY_EXTRA_ARGS", "").split()


def _get(server, path, timeout=5):
    with urllib.request.urlopen(f"{server}{path}", timeout=timeout) as f:
        return json.loads(f.read())


def _post(server, path, data, timeout=30):
    r = urllib.request.Request(f"{server}{path}", data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read() or b"{}")


def comfy_up(server=SERVER):
    try:
        _get(server, "/system_stats", timeout=2)
        return True
    except Exception:
        return False


def msvc_env():
    """The environment after Visual Studio's vcvars64.bat (cl.exe on PATH), so torch.compile / Triton can build
    kernels inside ComfyUI. None if Visual Studio isn't installed (only needed for W2MCompileWan)."""
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"), "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        return None
    try:
        path = subprocess.run([vswhere, "-latest", "-products", "*", "-property", "installationPath"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        bat = os.path.join(path, "VC", "Auxiliary", "Build", "vcvars64.bat")
        if not os.path.isfile(bat):
            return None
        out = subprocess.run(["cmd", "/c", bat, "&&", "set"], capture_output=True, text=True, timeout=120).stdout
        env = dict(line.split("=", 1) for line in out.splitlines() if "=" in line and not line.startswith("="))
        return env if "VCINSTALLDIR" in env else None
    except Exception:
        return None


def ensure_comfy(server=SERVER, comfy_dir=COMFY_DIR, wait_s=180):
    """Start ComfyUI if it isn't answering. Returns True if this call started it."""
    if comfy_up(server):
        return False
    py = os.path.join(comfy_dir, "python_embeded", "python.exe")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log = open(os.path.join(HERE, "logs", "comfyui.log"), "a")
    subprocess.Popen([py, os.path.join("ComfyUI", "main.py"), *COMFY_ARGS], cwd=comfy_dir, env=msvc_env() or None,
                     stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if comfy_up(server):
            print(f"[comfy] started in {time.time() - t0:.0f}s", flush=True)
            return True
        time.sleep(1)
    raise RuntimeError("ComfyUI did not start")


async def run_workflow(server, wf, timeout=3600):
    """Queue wf, wait for it to finish, return (prompt_id, seconds per node id)."""
    import aiohttp
    cid = uuid.uuid4().hex
    times, cur, t_cur = {}, None, time.time()
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(server.replace("http", "ws", 1) + f"/ws?clientId={cid}", max_msg_size=0, heartbeat=20) as ws:
            pid = _post(server, "/prompt", {"prompt": wf, "client_id": cid})["prompt_id"]
            t0 = t_cur = time.time()
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
                    now = time.time()
                    if cur:
                        times[cur] = times.get(cur, 0) + now - t_cur
                    cur, t_cur = d.get("node"), now
                    if cur is None:
                        break
                elif m["type"] == "execution_error":
                    raise RuntimeError(f"ComfyUI: {d.get('node_type')}: {d.get('exception_message', '')[:500]}")
    return pid, times


def history(server, pid):
    for _ in range(50):  # the websocket can report completion a moment before history is written
        hist = _get(server, f"/history/{pid}")
        if pid in hist and hist[pid].get("outputs"):
            return hist[pid]["outputs"]
        time.sleep(0.1)
    raise RuntimeError(f"ComfyUI history missing for {pid}")
