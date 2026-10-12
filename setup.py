r"""One-time setup: install the custom nodes into a ComfyUI and download the models (about 17.5 GB).

    python setup.py --comfy-dir C:\path\to\ComfyUI_windows_portable      (or a plain ComfyUI checkout)

Safe to re-run: files that already exist are skipped and interrupted downloads resume. Needs git for the
VideoHelperSuite node (or install it from ComfyUI Manager). It does not install ComfyUI itself; see README.md.
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

import comfy_client as cc

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "https://huggingface.co"
# (url, folder under ComfyUI/models, file name, size for the progress line)
MODELS = [
    (f"{HF}/lightx2v/LightWan2.2-A14B/resolve/main/Wan2.2-I2V-A14B_NVFP4_Sparse_high_comfy.safetensors", "diffusion_models", None),
    (f"{HF}/lightx2v/LightWan2.2-A14B/resolve/main/Wan2.2-I2V-A14B_NVFP4_Sparse_low_comfy.safetensors", "diffusion_models", None),
    (f"{HF}/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders", None),
    (f"{HF}/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors", "vae", None),
    (f"{HF}/lightx2v/Autoencoders/resolve/main/lighttaew2_1.safetensors", "vae_approx", None),
]
NODES = {"ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"}


def download(url, dst):
    name = os.path.basename(dst)
    if os.path.isfile(dst):
        print(f"[skip] {name} already there")
        return
    part = dst + ".part"
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as r:
        if have and r.status != 206:  # server ignored the range
            have = 0
        total = have + int(r.headers.get("Content-Length", 0))
        with open(part, "ab" if have else "wb") as f:
            done = have
            while chunk := r.read(8 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r[get] {name} {done / 2**30:.2f}/{total / 2**30:.2f} GB", end="", flush=True)
    os.replace(part, dst)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy-dir", default=cc.COMFY_DIR)
    ap.add_argument("--skip-models", action="store_true")
    a = ap.parse_args()
    _, src, _ = cc.comfy_paths(a.comfy_dir)
    if not os.path.isfile(os.path.join(src, "main.py")):
        sys.exit(f"No ComfyUI found at {a.comfy_dir} (looked for {os.path.join(src, 'main.py')}). Install ComfyUI first; see README.md.")
    nodes = os.path.join(src, "custom_nodes")
    dst = os.path.join(nodes, "comfyui-w2m")
    shutil.copytree(os.path.join(HERE, "comfy_nodes", "comfyui-w2m"), dst, dirs_exist_ok=True)
    print(f"[nodes] comfyui-w2m -> {dst}")
    for name, url in NODES.items():
        if os.path.isdir(os.path.join(nodes, name)):
            print(f"[skip] {name} already installed")
        else:
            subprocess.run(["git", "clone", "--depth", "1", url, os.path.join(nodes, name)], check=True)
    shutil.copy(os.path.join(HERE, "refs", "relaxed_stickman_appendages.jpg"), os.path.join(src, "input"))
    if not a.skip_models:
        for url, folder, _ in MODELS:
            os.makedirs(os.path.join(src, "models", folder), exist_ok=True)
            download(url, os.path.join(src, "models", folder, os.path.basename(url)))
    print("Done. Try: python stickman_i2v.py thumbs_up")


if __name__ == "__main__":
    main()
