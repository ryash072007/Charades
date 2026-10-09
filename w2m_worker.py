"""Phrase -> motion worker: one GPU queue, results cached per normalised phrase.

    python_embeded\\python.exe w2m_worker.py --clips-dir <site>/data/mocap/clips --cache-dir <site>/data/w2m-wan

GET  /health                     {"ok", "comfy", "llm", "busy", "queued"}
GET  /cached?text=...            {"hit", "result"}  (no GPU work)
POST /generate {"text", "fresh"?, "seed"?}
     -> NDJSON stream: {"stage": "queued"|"plan"|"encode"|"video"|"decode"|"pose"|"body"|"hands"|"export", "t"}...
        then {"done": true, "id", "frames", "seconds", "prompt", "cached", "timings"} or {"error": "..."}
POST /unload                     stop the LLM server and free ComfyUI's VRAM (the site calls this before HY/Gemma)

Clip files land in --clips-dir as <id>.bin (site format) + <id>.w2m.bin (fingers/face sidecar).
Run with ComfyUI's embedded Python.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)

import comfy_generate as cg  # noqa: E402
from export_clip import export  # noqa: E402
from extract_hands_face import extract  # noqa: E402
from mime_llm import MimeLLM  # noqa: E402
from prepare_motion import prepare  # noqa: E402

PIPELINE_VERSION = "fast-1"   # bump to invalidate cached clips when the pipeline changes


def slug(text):
    t = unicodedata.normalize("NFKD", text.lower())
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:60] or "phrase"


class Worker:
    def __init__(self, a):
        self.a = a
        self.server = a.server
        self.comfy_root = os.path.join(a.portable_root, "ComfyUI")
        self.smpl = os.path.join(self.comfy_root, "models", "motion_capture", "body_models", "smpl", "SMPL_NEUTRAL.npz")
        self.mp_models = os.path.join(self.comfy_root, "models", "mediapipe")
        os.makedirs(a.clips_dir, exist_ok=True)
        os.makedirs(a.cache_dir, exist_ok=True)
        self.llm = MimeLLM()
        self.gpu = threading.Lock()
        self.queued = 0
        self.busy = None
        self.last_job = 0.0
        threading.Thread(target=self._idle, daemon=True).start()

    def _idle(self):
        while True:
            time.sleep(20)
            if self.busy is None and self.last_job and time.time() - self.last_job > self.a.idle_s:
                self.unload(f"idle {self.a.idle_s:.0f}s")
                self.last_job = 0.0

    def unload(self, why="asked"):
        with self.gpu:
            self.llm.stop(why)
            cg.free_vram(self.server)
            print(f"[worker] freed GPU ({why})", flush=True)

    def cache_file(self, text):
        return os.path.join(self.a.cache_dir, f"{slug(text)}.json")

    def cached(self, text):
        try:
            r = json.load(open(self.cache_file(text), encoding="utf-8"))
            if r.get("version") == PIPELINE_VERSION and os.path.isfile(os.path.join(self.a.clips_dir, f"{r['id']}.bin")):
                return r
        except Exception:
            pass
        return None

    def generate(self, text, emit, fresh=False, seed=None):
        if not fresh and (hit := self.cached(text)):
            emit({"done": True, **hit, "cached": True})
            return
        t_req = time.time()
        if self.gpu.locked():
            emit({"stage": "queued", "t": 0, "ahead": self.queued + 1})
        self.queued += 1
        with self.gpu:
            self.queued -= 1
            self.busy = text
            try:
                self._run(text, emit, seed, t_req)
            finally:
                self.busy = None
                self.last_job = time.time()

    def _run(self, text, emit, seed, t_req):
        timings = {}
        t0 = time.time()
        mark = {"stage": None, "t": t0}

        def stage(name):
            now = time.time()
            if mark["stage"]:
                timings[mark["stage"]] = round(timings.get(mark["stage"], 0) + now - mark["t"], 2)
            mark.update(stage=name, t=now)
            if name:
                emit({"stage": name, "t": round(now - t0, 1)})

        stage("plan")
        plan = self.llm.plan(text)
        stage("load")  # ComfyUI start-up / queueing / model loads until the first sampling node
        cg.ensure_comfy(self.server, self.a.portable_root)
        sid = hashlib.sha1(f"{text}|{seed}|{time.time()}".encode()).hexdigest()[:6]
        cid = f"wan-{slug(text)[:40]}-{sid}"
        res = cg.generate(plan["prompt"], plan["seconds"], self.comfy_root, self.server, seed=seed,
                          full_vae=self.a.full_vae, job=cid, on_stage=stage)
        stage("hands")
        hf = os.path.join(res["dir"], "hands_face.npz")
        extract(res["rgb"], res["kps"], hf, self.mp_models, fps=16.0, verbose=False)
        stage("export")
        motion = os.path.join(res["dir"], "motion.npz")
        prepare(res["npz"], hf, self.smpl, motion)
        info = export(motion, os.path.join(self.a.clips_dir, f"{cid}.bin"), os.path.join(self.a.clips_dir, f"{cid}.w2m.bin"))
        stage(None)
        timings["total"] = round(time.time() - t0, 2)
        out = {"version": PIPELINE_VERSION, "text": text, "id": cid, "frames": info["frames"], "seconds": info["seconds"],
               "prompt": plan["prompt"], "rgb": os.path.relpath(res["rgb"], os.path.join(self.comfy_root, "output")),
               "timings": timings, "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S")}
        json.dump(out, open(self.cache_file(text), "w", encoding="utf-8"), indent=1)
        print(f"[worker] {text!r} -> {cid} in {timings['total']:.1f}s (queued {t0 - t_req:.1f}s) {timings}", flush=True)
        emit({"done": True, **out, "cached": False})


def make_handler(worker):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"   # stream NDJSON until we close the connection

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/cached?"):
                from urllib.parse import parse_qs, urlparse
                text = re.sub(r"\s+", " ", parse_qs(urlparse(self.path).query).get("text", [""])[0]).strip()[:80]
                hit = worker.cached(text) if text else None
                return self._json(200, {"hit": bool(hit), "result": hit})
            if self.path == "/health":
                return self._json(200, {"ok": True, "comfy": cg.comfy_up(worker.server), "llm": worker.llm.proc is not None and worker.llm.proc.poll() is None,
                                        "busy": worker.busy, "queued": worker.queued, "version": PIPELINE_VERSION})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._json(400, {"error": "bad json"})
            if self.path == "/unload":
                threading.Thread(target=worker.unload, daemon=True).start()
                return self._json(200, {"ok": True})
            if self.path != "/generate":
                return self._json(404, {"error": "not found"})
            text = re.sub(r"\s+", " ", str(body.get("text", ""))).strip()[:80]
            if len(text) < 2:
                return self._json(400, {"error": "text required"})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def emit(obj):
                try:
                    self.wfile.write((json.dumps(obj) + "\n").encode())
                    self.wfile.flush()
                except OSError:
                    pass  # client went away; the job still finishes and is cached
            try:
                worker.generate(text, emit, fresh=bool(body.get("fresh")), seed=body.get("seed"))
            except Exception as e:
                traceback.print_exc()
                emit({"error": str(e)[:500]})
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--portable-root", default=cg.DEFAULT_COMFY)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--idle-s", type=float, default=300, help="free the GPU after this long without jobs")
    ap.add_argument("--full-vae", action="store_true", help="decode with the full Wan VAE instead of the tiny one")
    ap.add_argument("--warm", action="store_true", help="start ComfyUI and load the LLM right away")
    a = ap.parse_args()
    w = Worker(a)
    if a.warm:
        threading.Thread(target=lambda: (cg.ensure_comfy(a.server, a.portable_root), w.llm.warm()), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(w))
    print(f"[worker] listening on 127.0.0.1:{a.port} (clips -> {a.clips_dir})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
