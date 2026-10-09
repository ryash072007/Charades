"""Score a filmed take: does the video clearly show the intended charades mime?

Four frames of the take go to a vision model on Groq (qwen/qwen3.8-27b by default) with the phrase and the mime,
and it returns a 0-10 score. The worker films up to N takes and keeps the first one that scores well.
Needs GROQ_API_KEY in the environment (the site's server passes its own); without it, judge() returns None.
"""
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

import av
from PIL import Image

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("W2M_JUDGE_MODEL", "qwen/qwen3.8-27b")

INSTRUCTIONS = """You check takes for a charades game. The actor in these {n} video frames (left to right, in time order)
was asked to act out "{phrase}" by doing this mime:
"{mime}"
Score how clearly the frames show that mime, as a charades player would read it from the body and hands:
10 = exactly that, obvious; 7 = clearly the right gesture with small differences; 4 = partly (right idea, wrong hand
shape or place); 0 = something else or nothing. Look closely at hand shapes (fist vs thumb up vs pointing finger) and
where the hands are relative to the body and face. Answer JSON only: {{"score": <0-10>, "seen": "<what the person does, 12 words max>"}}"""


def frames_grid(video, n=4, width=200):
    frames = [f.to_image() for f in av.open(video).decode(video=0)]
    pick = [frames[round(i * (len(frames) - 1) / (n - 1))] for i in range(n)]
    h = round(width * pick[0].height / pick[0].width)
    grid = Image.new("RGB", (width * n, h))
    for i, im in enumerate(pick):
        grid.paste(im.resize((width, h)), (i * width, 0))
    buf = io.BytesIO()
    grid.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def judge(video, phrase, mime, timeout=40):
    """{"score": 0-10, "seen": str, "model": str} or None when no judge is available."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    # ~1k tokens a call with reasoning off (Groq counts max_tokens against its 8k tokens/minute, so keep it small)
    body = {"model": MODEL, "temperature": 0.1, "max_tokens": 120, "reasoning_effort": "none", "messages": [{"role": "user", "content": [
        {"type": "text", "text": INSTRUCTIONS.format(n=4, phrase=phrase, mime=mime)},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frames_grid(video, width=160)}"}}]}]}
    req = urllib.request.Request(GROQ_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                          "User-Agent": "WANPhrase2Motion/1.0"})  # Groq rejects urllib's default agent
    text = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as f:
                text = json.loads(f.read())["choices"][0]["message"]["content"]
            break
        except urllib.error.HTTPError as e:
            wait = float(e.headers.get("retry-after") or 0)
            if e.code == 429 and attempt == 0 and wait <= 15:
                time.sleep(wait + 0.5)
                continue
            print(f"[judge] unavailable: {e.code} {e.reason}", flush=True)
            return None
        except Exception as e:  # outages: no verdict, the worker keeps the take
            print(f"[judge] unavailable: {str(e)[:160]}", flush=True)
            return None
    text = re.sub(r"^[\s\S]*?</think>", "", text).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    try:
        out = json.loads(m.group(0)) if m else {}
        score = float(out.get("score"))
    except (TypeError, ValueError, json.JSONDecodeError):
        n = re.search(r"score\D{0,5}(\d+(?:\.\d+)?)", text)
        if not n:
            return None
        score, out = float(n.group(1)), {}
    return {"score": max(0.0, min(10.0, score)), "seen": str(out.get("seen", ""))[:120], "model": MODEL}


if __name__ == "__main__":
    import sys
    print(judge(sys.argv[1], sys.argv[2], sys.argv[3]))
