"""Phrase -> mime plan ({"prompt", "seconds"}) with qwen3.5:9b.

Uses a resident, text-only llama-server (flash attention, q8 KV cache, 4K context, thinking off) on the
GGUF that Ollama already downloaded, so there's no second copy of the model and no vision projector to load.
Measured on the RTX 5060 Ti: ~5 s to load once, then ~0.8-1.0 s per phrase (vs 6-9 s per phrase with
Ollama keep_alive=0 + thinking). Falls back to Ollama if llama-server isn't available.
"""
import json
import os
import re
import subprocess
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = open(os.path.join(HERE, "mime_system_prompt.txt"), encoding="utf-8").read()
LLAMA_EXE = os.environ.get("W2M_LLAMA_SERVER", r"C:\qd\llama-g4\build\bin\llama-server.exe")
OLLAMA_URL = os.environ.get("W2M_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("W2M_OLLAMA_MODEL", "qwen3.5:9b")
PORT = int(os.environ.get("W2M_LLM_PORT", 8094))
IDLE_S = float(os.environ.get("W2M_LLM_IDLE_S", 600))


def _post(url, body, timeout=120):
    r = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())


def ollama_gguf(model=OLLAMA_MODEL):
    """Path of the model weights blob in Ollama's store (text model only, no projector)."""
    root = os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models"))
    name, tag = (model.split(":") + ["latest"])[:2]
    man = json.load(open(os.path.join(root, "manifests", "registry.ollama.ai", "library", name, tag)))
    layer = next(l for l in man["layers"] if l["mediaType"] == "application/vnd.ollama.image.model")
    return os.path.join(root, "blobs", layer["digest"].replace(":", "-"))


def parse_plan(text):
    m = re.search(r"\{.*\}", text, re.S)
    plan = json.loads(m.group(0) if m else text)
    prompt = str(plan.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"LLM returned no prompt: {text[:200]!r}")
    try:
        seconds = float(plan.get("seconds", 3))
    except (TypeError, ValueError):
        seconds = 3.0
    return {"prompt": prompt, "seconds": seconds}


class MimeLLM:
    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.last = 0.0
        self._kill_strays()
        threading.Thread(target=self._idle, daemon=True).start()

    @staticmethod
    def _kill_strays():
        """A worker that was killed hard leaves its llama-server (~6 GB VRAM) behind: stop orphans on start-up."""
        try:
            import psutil
        except ImportError:
            return
        for p in psutil.process_iter(["name", "cmdline"]):
            cmd = p.info.get("cmdline") or []
            if (p.info.get("name") or "").lower().startswith("llama-server") and "--port" in cmd \
                    and cmd[cmd.index("--port") + 1] == str(PORT):
                try:
                    orphan = p.parent() is None
                except psutil.Error:
                    orphan = True
                if orphan:  # a server another live process owns is left alone (and shared, see _start)
                    print(f"[llm] stopping orphaned llama-server pid {p.pid}", flush=True)
                    p.kill()

    # ---------------------------------------------------------------- llama-server lifecycle
    def _healthy(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1) as f:
                return json.loads(f.read()).get("status") == "ok"
        except Exception:
            return False

    def _start(self):
        if self.proc and self.proc.poll() is None and self._healthy():
            return True
        if self.proc is None and self._healthy():
            return True  # another process (e.g. the site's worker) already serves the model: share it
        if not os.path.isfile(LLAMA_EXE):
            return False
        self.stop()
        log = open(os.path.join(HERE, "logs", "llama-server.log"), "a") if os.path.isdir(os.path.join(HERE, "logs")) else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [LLAMA_EXE, "-m", ollama_gguf(), "-ngl", "99", "-c", "4096", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
             "-np", "1", "--host", "127.0.0.1", "--port", str(PORT), "--no-webui"],
            stdout=subprocess.DEVNULL, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        t0 = time.time()
        while time.time() - t0 < 90:
            if self.proc.poll() is not None:
                return False
            if self._healthy():
                print(f"[llm] llama-server ready in {time.time() - t0:.1f}s", flush=True)
                return True
            time.sleep(0.2)
        self.stop()
        return False

    def stop(self, why=""):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            if why:
                print(f"[llm] stopped ({why})", flush=True)
        self.proc = None

    def _idle(self):
        while True:
            time.sleep(15)
            with self.lock:
                if self.proc and time.time() - self.last > IDLE_S:
                    self.stop(f"idle {IDLE_S:.0f}s")

    def warm(self):
        with self.lock:
            self.last = time.time()
            return self._start()

    # ---------------------------------------------------------------- generation
    def plan(self, phrase, seed=2007):
        with self.lock:
            self.last = time.time()
            if self._start():
                r = _post(f"http://127.0.0.1:{PORT}/v1/chat/completions", {
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": phrase}],
                    "temperature": 0.4, "seed": seed, "response_format": {"type": "json_object"},
                    "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 500})
                return parse_plan(r["choices"][0]["message"]["content"])
        # fallback: Ollama, unloaded right after so it doesn't sit on the GPU
        r = _post(f"{OLLAMA_URL}/api/generate", {
            "model": OLLAMA_MODEL, "system": SYSTEM, "prompt": phrase, "think": False, "format": "json",
            "stream": False, "keep_alive": 0, "options": {"temperature": 0.4, "seed": seed, "num_ctx": 4096}})
        return parse_plan(r["response"])


if __name__ == "__main__":
    import sys
    llm = MimeLLM()
    for p in sys.argv[1:] or ["eating burger"]:
        t = time.time()
        print(p, "->", llm.plan(p), f"({time.time() - t:.2f}s)")
    llm.stop()
