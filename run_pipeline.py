"""One clip -> procedural 3D character animation (MP4 + GLB).

Either generate a new clip through a running ComfyUI server:
    python_embeded\\python.exe run_pipeline.py --phrase "eating burger"
or process files that the mocap workflow already wrote:
    python_embeded\\python.exe run_pipeline.py --video output\\w2m\\clip_rgb_00001.mp4 \\
        --kps output\\w2m\\clip_kps_00001.json --npz output\\smpl_params_0004.npz

Outputs go to ComfyUI/output/w2m/<name>/: <name>.mp4, <name>_compare.mp4, <name>.glb,
plus the intermediate hands_face.npz / motion.npz.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_COMFY = r"C:\Users\ryash\OneDrive\Documents\AI\ComfyUI_windows_portable\ComfyUI"
DEFAULT_BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def run(cmd):
    print("[run]", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("input (either --phrase, or --video/--kps/--npz)")
    src.add_argument("--phrase", help="generate a new clip for this phrase via the ComfyUI API")
    src.add_argument("--video", help="RGB clip from the mocap workflow (w2m/clip_rgb_*.mp4)")
    src.add_argument("--kps", help="DWPose keypoints JSON (w2m/clip_kps_*.json)")
    src.add_argument("--npz", help="GVHMR smpl_params_*.npz")
    ap.add_argument("--name", help="output name (default: derived from the video file)")
    ap.add_argument("--comfy-root", default=DEFAULT_COMFY)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--blender", default=DEFAULT_BLENDER)
    ap.add_argument("--color", default="#78cdb3", help="pastel body colour (hex)")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=1248)
    ap.add_argument("--samples", type=int, default=48, help="EEVEE samples per frame")
    ap.add_argument("--ground", choices=["per_frame", "global"], default="per_frame",
                    help="per_frame keeps a foot on the floor every frame; global allows jumps")
    ap.add_argument("--no-glb", action="store_true")
    ap.add_argument("--no-compare", action="store_true", help="skip the side-by-side video")
    ap.add_argument("--keep-frames", action="store_true")
    a = ap.parse_args()

    comfy = os.path.abspath(a.comfy_root)
    t0 = time.time()

    if a.phrase:
        from comfy_generate import generate
        g = generate(a.phrase, comfy, a.server)
        video, kps, npz = g["rgb"], g["kps"], g["npz"]
    else:
        if not (a.video and a.kps and a.npz):
            ap.error("give --phrase, or all of --video, --kps and --npz")
        video, kps, npz = a.video, a.kps, a.npz
    for p in (video, kps, npz):
        if not os.path.isfile(p):
            sys.exit(f"missing input: {p}")

    name = a.name or re.sub(r"^clip_rgb_", "w2m_", os.path.splitext(os.path.basename(video))[0])
    out_dir = os.path.join(comfy, "output", "w2m", name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[pipeline] video={video}\n[pipeline] kps={kps}\n[pipeline] npz={npz}\n[pipeline] out={out_dir}")

    import av
    with av.open(video) as c:
        fps = float(c.streams.video[0].average_rate or 16)

    from extract_hands_face import extract
    from prepare_motion import prepare
    hf = os.path.join(out_dir, "hands_face.npz")
    extract(video, kps, hf, os.path.join(comfy, "models", "mediapipe"), fps=fps)
    motion = os.path.join(out_dir, "motion.npz")
    prepare(npz, hf, os.path.join(comfy, "models", "motion_capture", "body_models", "smpl", "SMPL_NEUTRAL.npz"),
            motion, ground=a.ground)

    frames = os.path.join(out_dir, "frames")
    shutil.rmtree(frames, ignore_errors=True)
    glb = os.path.join(out_dir, f"{name}.glb")
    cmd = [a.blender, "-b", "--factory-startup", "--python", os.path.join(HERE, "blender_character.py"), "--",
           "--motion", motion, "--frames-dir", frames, "--width", a.width, "--height", a.height,
           "--color", a.color, "--samples", a.samples]
    if not a.no_glb:
        cmd += ["--glb", glb]
    run(cmd)

    ff = ffmpeg_exe()
    mp4 = os.path.join(out_dir, f"{name}.mp4")
    run([ff, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", os.path.join(frames, "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", "-movflags", "+faststart", mp4])

    if not a.no_compare:
        cmp_ = os.path.join(out_dir, f"{name}_compare.mp4")
        run([ff, "-y", "-loglevel", "error", "-i", video, "-i", mp4, "-filter_complex",
             f"[0:v]scale=-2:{a.height},setsar=1[l];[1:v]setsar=1[r];[l][r]hstack=inputs=2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", f"{fps:g}", cmp_])

    if not a.keep_frames:
        shutil.rmtree(frames, ignore_errors=True)
    print(f"[pipeline] done in {time.time() - t0:.0f}s")
    print(f"[pipeline] MP4 : {mp4}")
    if not a.no_glb:
        print(f"[pipeline] GLB : {glb}")


if __name__ == "__main__":
    main()
