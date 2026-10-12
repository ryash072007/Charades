"""Phrase -> motion worker: one GPU queue, results cached per normalised phrase.

    python_embeded\\python.exe w2m_worker.py --clips-dir <site>/data/mocap/clips --cache-dir <site>/data/w2m-wan

GET  /health                     {"ok", "comfy", "llm", "busy", "queued"}
GET  /cached?text=...            {"hit", "result"}  (no GPU work)
POST /generate {"text", "fresh"?, "seed"?, "plan"?: {"prompt", "seconds", "model"}}  (no plan: the local LLM writes it)
     -> NDJSON stream: {"stage": "queued"|"plan"|"encode"|"video"|"decode"|"pose"|"body"|"hands"|"export", "t"}...
        then {"done": true, "id", "frames", "seconds", "prompt", "cached", "timings"} or {"error": "..."}
POST /unload                     stop the LLM server and free ComfyUI's VRAM (the site calls this before HY/Gemma)
POST /shutdown                   exit the worker; stops the LLM and the ComfyUI it started

Clip files land in --clips-dir as <id>.bin (site format), <id>.w2m.bin (fingers/face sidecar for the stick figure)
and <id>.rig.json (skeleton + rotations + body measurements for the skinned characters).
Run with ComfyUI's embedded Python.
"""
import argparse
import hashlib
import random
import json
import os
import re
import signal
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
from export_clip import export, export_rig  # noqa: E402
from extract_hands_face import extract  # noqa: E402
from mime_llm import MimeLLM  # noqa: E402
from prepare_motion import prepare  # noqa: E402
from refine_arms import refine  # noqa: E402
from take_judge import judge  # noqa: E402

PIPELINE_VERSION = "fast-4"   # bump to invalidate cached clips when the pipeline changes (fast-2: charades-first mime prompt; fast-3: 6-step CFG 2 video; fast-4: judged takes)


def slug(text):
    t = unicodedata.normalize("NFKD", text.lower())
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")[:60] or "phrase"


class Worker:
    def __init__(self, a):
        self.a = a
        self.server = a.server
        self.comfy_root = os.path.join(a.portable_root, "ComfyUI")
        # GVHMR fits SMPL-X, so its betas/pose belong to SMPL-X; SMPL is only needed to migrate older clips
        self.smpl = os.path.join(self.comfy_root, "models", "motion_capture", "body_models", "smplx", "SMPLX_NEUTRAL.npz")
        self.smpl_old = os.path.join(self.comfy_root, "models", "motion_capture", "body_models", "smpl", "SMPL_NEUTRAL.npz")
        self.mp_models = os.path.join(self.comfy_root, "models", "mediapipe")
        os.makedirs(a.clips_dir, exist_ok=True)
        os.makedirs(a.cache_dir, exist_ok=True)
        self.llm = MimeLLM()
        self.gpu = threading.Lock()
        self.queued = 0
        self.busy = None
        self.last_job = 0.0
        threading.Thread(target=self._idle, daemon=True).start()
        threading.Thread(target=self._backfill_rigs, daemon=True).start()

    def _backfill_rigs(self):
        """Bring clips made by older pipeline versions up to date from their saved motion files (CPU, ~2 s each):
        SMPL-X skeleton (GVHMR's own model; older clips used SMPL), re-grounded, arms refined against the video's 2D
        keypoints, clip + rig + exact-fit mesh re-exported. The first original is kept as motion_unrefined.npz."""
        import shutil
        import numpy as np
        from body_dims import betas_from_joints
        from prepare_motion import fk, ground_root, rest_joints
        for f in os.listdir(self.a.cache_dir):
            try:
                r = json.load(open(os.path.join(self.a.cache_dir, f), encoding="utf-8"))
                cid = r["id"]
                job = os.path.join(self.comfy_root, "output", "w2m", cid)
                motion, kps = os.path.join(job, "motion.npz"), os.path.join(job, "kps_00001.json")
                base = os.path.join(job, "motion_unrefined.npz")
                rig = os.path.join(self.a.clips_dir, f"{cid}.rig.json")
                if not os.path.isfile(motion) or not os.path.isfile(kps):
                    continue
                d = dict(np.load(motion))
                current = str(d.get("model", "smpl")) == "smplx" and "refined" in d                     and os.path.isfile(rig) and os.path.isfile(os.path.join(self.a.clips_dir, f"{cid}.smplx.bin"))
                if current:
                    continue
                if not os.path.isfile(base):
                    shutil.copy(motion, base)
                d = dict(np.load(base))  # start from GVHMR's arms, not an earlier refinement
                if str(d.get("model", "smpl")) != "smplx":
                    betas = d["betas"] if "betas" in d else betas_from_joints(self.smpl_old, d["rest_joints"].astype(np.float64))
                    J = rest_joints(self.smpl, betas)
                    local = d["local_rot"].astype(np.float64)
                    root = ground_root(J, local, d["root_pos"].astype(np.float64))
                    d.update(rest_joints=J.astype(np.float32), betas=betas, root_pos=root.astype(np.float32),
                             joints=fk(J, local, root)[1].astype(np.float32), model=np.array("smplx"))
                d.pop("refined", None)
                np.savez(motion, **d)
                refine(motion, kps, self.smpl, verbose=False)
                export(motion, os.path.join(self.a.clips_dir, f"{cid}.bin"), os.path.join(self.a.clips_dir, f"{cid}.w2m.bin"))
                export_rig(motion, rig, self.smpl)
                print(f"[worker] upgraded {cid} (SMPL-X skeleton, arms refined, rig + exact-fit mesh)", flush=True)
            except Exception as e:
                print(f"[worker] upgrade {f}: {e}", flush=True)

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

    def generate(self, text, emit, fresh=False, seed=None, plan=None):
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
                self._run(text, emit, seed, t_req, plan)
            finally:
                self.busy = None
                self.last_job = time.time()

    def _run(self, text, emit, seed, t_req, given=None):
        timings = {}
        t0 = time.time()
        mark = {"stage": None, "t": t0}

        def stage(name, **extra):
            now = time.time()
            if mark["stage"]:
                timings[mark["stage"]] = round(timings.get(mark["stage"], 0) + now - mark["t"], 2)
            mark.update(stage=name, t=now)
            if name:
                emit({"stage": name, "t": round(now - t0, 1), **extra})

        stage("plan")
        if given and len(str(given.get("prompt", ""))) >= 20:  # written by the caller (the site uses Groq)
            plan = {"prompt": str(given["prompt"])[:600], "seconds": float(given.get("seconds") or 3),
                    "planner": str(given.get("model") or "caller")[:80]}
        else:
            plan = {**self.llm.plan(text), "planner": "local qwen3.5:9b"}
        stage("load")  # ComfyUI start-up / queueing / model loads until the first sampling node
        cg.ensure_comfy(self.server, self.a.portable_root)
        sid = hashlib.sha1(f"{text}|{seed}|{time.time()}".encode()).hexdigest()[:6]
        cid = f"wan-{slug(text)[:40]}-{sid}"
        # Film up to N takes with different seeds; a vision model checks each one shows the mime and the first
        # good take wins (otherwise the best). The small video model often misses hand signs on a given seed.
        base_seed = int(seed) if seed is not None else random.randrange(2 ** 48)
        takes, best = [], None
        for k in range(max(1, self.a.takes)):
            res = cg.film(plan["prompt"], plan["seconds"], self.comfy_root, self.server, seed=base_seed + k * 7919,
                          full_vae=self.a.full_vae, fast_video=self.a.fast_video, job=cid, take=k + 1,
                          on_stage=lambda s, k=k: stage(s, take=k + 1))
            if self.a.takes <= 1:
                best = res
                break
            stage("judge", take=k + 1)
            v = judge(res["rgb"], text, plan["prompt"])
            takes.append({"take": k + 1, "seed": base_seed + k * 7919, "score": v and v["score"], "seen": v and v["seen"]})
            emit({"verdict": True, "take": k + 1, "score": v and v["score"], "seen": v and v["seen"]})
            if v is None:  # no judge available: keep this take
                best = res
                break
            if best is None or v["score"] > best["score"]:
                best = {**res, "score": v["score"]}
            if v["score"] >= self.a.accept:
                break
        stage("pose")
        cap = cg.capture(best["rgb"], self.comfy_root, self.server, job=cid, on_stage=stage)
        res = {**best, **cap}
        stage("hands")
        hf = os.path.join(res["dir"], "hands_face.npz")
        extract(res["rgb"], res["kps"], hf, self.mp_models, fps=16.0, verbose=False)
        stage("export")
        motion = os.path.join(res["dir"], "motion.npz")
        prepare(res["npz"], hf, self.smpl, motion)
        refine(motion, res["kps"], self.smpl, verbose=False)
        info = export(motion, os.path.join(self.a.clips_dir, f"{cid}.bin"), os.path.join(self.a.clips_dir, f"{cid}.w2m.bin"))
        export_rig(motion, os.path.join(self.a.clips_dir, f"{cid}.rig.json"), self.smpl)
        stage(None)
        timings["total"] = round(time.time() - t0, 2)
        out = {"version": PIPELINE_VERSION, "text": text, "id": cid, "frames": info["frames"], "seconds": info["seconds"],
               "prompt": plan["prompt"], "planner": plan["planner"], "takes": takes, "rgb": os.path.relpath(res["rgb"], os.path.join(self.comfy_root, "output")),
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
            if self.path == "/shutdown":  # main()'s finally stops the LLM and the ComfyUI it started
                self._json(200, {"ok": True})
                threading.Thread(target=_server.shutdown, daemon=True).start()
                return
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
                worker.generate(text, emit, fresh=bool(body.get("fresh")), seed=body.get("seed"),
                                plan=body.get("plan") if isinstance(body.get("plan"), dict) else None)
            except Exception as e:
                traceback.print_exc()
                emit({"error": str(e)[:500]})
    return H


_server = None  # the ThreadingHTTPServer, so POST /shutdown can stop it


def main():
    global _server
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--portable-root", default=cg.DEFAULT_COMFY)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--idle-s", type=float, default=300, help="free the GPU after this long without jobs")
    ap.add_argument("--full-vae", action="store_true", help="decode with the full Wan VAE instead of the tiny one")
    ap.add_argument("--fast-video", action="store_true",
                    help="old 4-step CFG 1 sampling: ~22 s faster per clip, but hand signs are often wrong")
    ap.add_argument("--takes", type=int, default=3, help="max video takes per phrase (judged by a vision model)")
    ap.add_argument("--accept", type=float, default=7, help="judge score (0-10) that accepts a take")
    ap.add_argument("--warm", action="store_true", help="start ComfyUI and load the LLM right away")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so the finally below runs on SIGTERM too
    cg.reap_stale_comfy()  # an earlier worker may have died without stopping its ComfyUI
    w = Worker(a)
    if a.warm:
        threading.Thread(target=lambda: (cg.ensure_comfy(a.server, a.portable_root), w.llm.warm()), daemon=True).start()
    srv = _server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(w))
    print(f"[worker] listening on 127.0.0.1:{a.port} (clips -> {a.clips_dir})", flush=True)
    try:
        srv.serve_forever()
    finally:
        # Ctrl+C / SIGTERM: stop the ComfyUI we started so it doesn't outlive the worker
        w.llm.stop("worker exiting")
        cg.stop_comfy()


if __name__ == "__main__":
    main()
