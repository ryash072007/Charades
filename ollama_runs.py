import json
import sys
from pathlib import Path

import requests


# ----------------------------
# Configuration
# ----------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"

TARGET = " ".join(sys.argv[1:]).strip() or "suspicious"

OUTPUT_FILE = Path("charades_prompts.json")

FPS = 16
FRAMES_PER_CLIP = 49
SECONDS_PER_CLIP = FRAMES_PER_CLIP / FPS


# ----------------------------
# Fixed positive prompt
# Preserve the character description that already worked.
# Do not include the old fixed thumbs-up action here.
# ----------------------------

BASE_PROMPT = (
    "2D cartoon stick figure animation on a plain white background, "
    "flat hand-drawn style, static camera. "
    "A simple stickman with a round white head with dot eyes and "
    "a small, simply drawn mouth, spiky teal hair, "
    "thick rounded blue outline limbs, a white rectangular torso "
    "outlined in blue, teal shorts. The character is centered "
    "with the whole body visible from head to feet. "
    "The stickman has round blue hands with five short fingers, "
    "small round ears on both sides of the head and a small button nose. "
    "No visible props, no additional characters, no text or captions. "
)


# ----------------------------
# Compact system prompt
# ----------------------------

SYSTEM_PROMPT = """
You generate action prompts for a fixed Wan 2.2 I2V silent-charades workflow.

Your job is to translate an English word or phrase into a visually recognizable stickman action.

The video model already understands natural-language actions well. Give it simple, direct instructions. Do not over-explain obvious movements.

## Fixed constraints

* One stickman, using the existing character design.
* No visible props, additional characters, text, captions or sound.
* Facial expressions, body language and imaginary-object pantomime are allowed.
* Every clip starts independently from the same reference image.
* The final video is approximately 3 seconds long.

## Action rules

1. For an obvious action or gesture, describe it directly using ordinary language. Do not decompose it into anatomical instructions.
2. Prefer one recognizable action per clip. Let the video model determine the natural intermediate movements.
3. Keep actions short, usually 5–15 words.
4. Do not add unnecessary nodding, smiling, walking, looking around or other secondary actions.
5. Do not explain what a gesture symbolizes. Describe what the character visibly does.
6. Avoid specifying individual finger movements, precise limb angles or detailed body mechanics unless they are essential to recognizing the action.
7. Do not add emotional expressions unless they help communicate the intended meaning.
8. Do not turn a simple gesture into a multi-step performance.
9. Split a word or phrase into multiple clips only when the meaning is genuinely difficult to depict in one clip. Do not split familiar gestures or phrases that already have a recognizable physical representation.
10. For abstract words or idioms, use a simple physical metaphor or up to three recognizable clues. Each clip must work independently.
11. Do not generate camera directions, art styles, character descriptions, frame counts, durations or negative prompts. These are handled elsewhere.
12. Each segment's action must be a concise English sentence beginning with "The stickman".

## Examples

Target: thumbs up
Action: The stickman raises one hand and gives a thumbs-up.

Target: jumping
Action: The stickman bends his knees, jumps straight up and lands.

Target: shrugging
Action: The stickman raises both shoulders and spreads his hands in a shrug.

Target: sneaking
Action: The stickman crouches and tiptoes forward cautiously.

Target: embarrassed
Action: The stickman looks down, rubs the back of his head and gives an awkward smile.

## Output

Return only the JSON object matching the supplied schema.

Each segment contains a clue and its corresponding action. Use one segment by default. Create additional segments only when splitting genuinely improves recognizability.

Do not include explanations or markdown.

"""


# ----------------------------
# Structured output schema
# ----------------------------

SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "clue": {
                        "type": "string"
                    },
                    "action": {
                        "type": "string"
                    }
                },
                "required": ["clue", "action"],
                "additionalProperties": False
            }
        }
    },
    "required": ["segments"],
    "additionalProperties": False
}


# ----------------------------
# Generate actions
# ----------------------------

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Create silent-charades clips for: {TARGET}"
        }
    ],
    "format": SCHEMA,
    "stream": False,
    "think": True,
    "options": {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "num_ctx": 4096,
        "num_predict": 4096,
        "repeat_penalty": 1.05
    },
    "keep_alive": "10m"
}

try:
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )
    response.raise_for_status()
    data = response.json()

except requests.RequestException as exc:
    raise SystemExit(f"Ollama request failed: {exc}") from exc

content = data.get("message", {}).get("content", "")

try:
    generated = json.loads(content)

except json.JSONDecodeError as exc:
    print(f"JSON parsing failed: {exc}")
    print(f"Finish reason: {data.get('done_reason', 'unknown')}")
    print(f"Generated characters: {len(content)}")
    print("\nResponse tail:")
    print(content[-1500:])
    raise SystemExit(1) from exc


# ----------------------------
# Validate and construct prompts
# ----------------------------

segments = generated.get("segments")

if not isinstance(segments, list) or not 1 <= len(segments) <= 3:
    raise SystemExit("Invalid result: expected between 1 and 3 segments.")

output_segments = []

for index, segment in enumerate(segments, start=1):
    clue = segment.get("clue", "").strip()
    action = segment.get("action", "").strip()

    if not clue or not action:
        raise SystemExit(f"Segment {index} has an empty clue or action.")

    if not action.lower().startswith("the stickman"):
        raise SystemExit(
            f"Segment {index} must start with 'The stickman': {action}"
        )

    # Add the fixed style and character description in Python.
    # The LLM never regenerates these details.
    positive_prompt = BASE_PROMPT + action

    output_segments.append({
        "index": index,
        "clue": clue,
        "action": action,
        "positive_prompt": positive_prompt,
        "frame_count": FRAMES_PER_CLIP,
        "fps": FPS,
        "duration_seconds": SECONDS_PER_CLIP
    })


result = {
    "target": TARGET,
    "model": MODEL,
    "segments": output_segments
}

OUTPUT_FILE.write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(json.dumps(result, indent=2, ensure_ascii=False))
print(f"\nSaved prompts to: {OUTPUT_FILE.resolve()}")

