"""Queue the mocap workflow on a running ComfyUI server and collect its output files.

Run with ComfyUI's embedded Python (stdlib only).
"""
import json
import os
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORKFLOW = os.path.join(HERE, "wanphrase2motion_mocap.json")

PHRASE_NODE = "86"      # OllamaGenerateV2 – user phrase
RGB_NODE = "200"        # VHS_VideoCombine on VAE Decode
KPS_NODE = "201"        # SavePoseKpsAsJsonFile on DWPose
NPZ_NODE = "202"        # PreviewAny on GVHMR npz_path
PROMPT_NODE = "124"     # PreviewAny on final Wan prompt


def _req(url, data=None):
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=30) as f:
        return json.loads(f.read())


def _text(ui):
    t = (ui or {}).get("text")
    if isinstance(t, list):
        t = t[0] if t else ""
    return t or ""


def generate(phrase, comfy_root, server="http://127.0.0.1:8188", workflow=DEFAULT_WORKFLOW,
             seed=None, timeout=3600):
    wf = json.load(open(workflow, encoding="utf-8"))
    wf[PHRASE_NODE]["inputs"]["prompt"] = phrase
    if seed is not None:
        wf["117"]["inputs"]["noise_seed"] = int(seed)
    client_id = uuid.uuid4().hex
    res = _req(f"{server}/prompt", {"prompt": wf, "client_id": client_id})
    pid = res["prompt_id"]
    print(f"[generate] queued prompt {pid} for phrase {phrase!r}")

    t0 = time.time()
    while True:
        hist = _req(f"{server}/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("status_str") == "error" or (status.get("completed") is False and status.get("messages")
                                                       and any(m[0] == "execution_error" for m in status["messages"])):
                raise RuntimeError(f"ComfyUI execution failed: {json.dumps(status)[:2000]}")
            if status.get("completed") or entry.get("outputs"):
                break
        if time.time() - t0 > timeout:
            raise TimeoutError("ComfyUI generation timed out")
        time.sleep(3)
    print(f"[generate] finished in {time.time() - t0:.0f}s")

    out = entry["outputs"]
    out_dir = os.path.join(comfy_root, "output")
    gifs = out[RGB_NODE].get("gifs") or out[RGB_NODE].get("videos") or []
    mp4 = [g for g in gifs if g["filename"].endswith(".mp4")][0]
    rgb = os.path.join(out_dir, mp4.get("subfolder", ""), mp4["filename"])
    npz = _text(out.get(NPZ_NODE, {}))

    # SavePoseKpsAsJsonFile reports nothing in the UI: take the newest matching json.
    kps_dir = os.path.join(out_dir, "w2m")
    kps = sorted((os.path.join(kps_dir, f) for f in os.listdir(kps_dir)
                  if f.startswith("clip_kps") and f.endswith(".json")), key=os.path.getmtime)[-1]
    prompt_text = _text(out.get(PROMPT_NODE, {}))
    return {"rgb": rgb, "npz": npz, "kps": kps, "prompt": prompt_text, "phrase": phrase}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phrase")
    ap.add_argument("--comfy-root", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    a = ap.parse_args()
    print(json.dumps(generate(a.phrase, a.comfy_root, a.server), indent=2))
