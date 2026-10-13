"""Fetch llama.cpp (CUDA, Windows) and the Gemma 4 26B-A4B NVFP4 GGUF into this repo.

Downloads use many parallel HTTP range connections and resume if interrupted.
Usage: python llm_setup.py
"""
import concurrent.futures as cf
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).parent
LLAMA_DIR = ROOT / "llama.cpp"
MODEL_DIR = ROOT / "models"

MODEL_URL = (
    "https://huggingface.co/FreedomAISVR/Gemma-4-26B-A4B-it-QAT-NVFP4-GGUF"
    "/resolve/main/gemma-4-26b-a4b-it-qat-nvfp4.gguf"
)
CUDA_TAG = "cuda-13.4-x64"
CONNECTIONS = 12
CHUNK = 64 * 2**20


def download(url: str, dest: Path) -> None:
    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    if dest.exists() and dest.stat().st_size == size:
        print(f"have {dest.name}")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    done_file = dest.with_suffix(dest.suffix + ".done")
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    if not part.exists() or part.stat().st_size != size:
        with open(part, "wb") as f:
            f.truncate(size)
        done = set()
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    total = len(ranges)

    def fetch(r):
        start, end = r
        if str(start) in done:
            return start
        for attempt in range(20):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    with open(part, "r+b") as f:
                        f.seek(start)
                        for block in resp.iter_content(1 << 20):
                            f.write(block)
                return start
            except requests.RequestException:
                time.sleep(min(2 + attempt, 15))
        raise RuntimeError(f"chunk {start} failed")

    with cf.ThreadPoolExecutor(CONNECTIONS) as pool, open(done_file, "a") as log:
        for n, start in enumerate(pool.map(fetch, ranges), 1):
            log.write(f"{start}\n")
            log.flush()
            print(f"\r{dest.name}: {n}/{total} chunks", end="", flush=True)
    print()
    part.replace(dest)
    done_file.unlink(missing_ok=True)


def setup_llama() -> None:
    if (LLAMA_DIR / "llama-server.exe").exists():
        print("have llama.cpp")
        return
    # /releases/latest points at a non-binary tag; take the newest build instead.
    rel = requests.get(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1",
        timeout=30).json()[0]
    LLAMA_DIR.mkdir(exist_ok=True)
    for asset in rel["assets"]:
        n = asset["name"]
        if n.endswith(f"bin-win-{CUDA_TAG}.zip") or n == f"cudart-llama-bin-win-{CUDA_TAG}.zip":
            z = LLAMA_DIR / n
            download(asset["browser_download_url"], z)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(LLAMA_DIR)
            z.unlink()
    if not (LLAMA_DIR / "llama-server.exe").exists():
        raise SystemExit(f"no {CUDA_TAG} Windows build in {rel['tag_name']}")
    print(f"llama.cpp {rel['tag_name']} ready")


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)
    setup_llama()
    download(MODEL_URL, MODEL_DIR / Path(MODEL_URL).name)
    print("done")
