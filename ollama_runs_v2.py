import json
import sys
from pathlib import Path

import requests


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"

TARGET = " ".join(sys.argv[1:]).strip() or "suspicious"

OUTPUT_FILE = Path("charades_prompts.json")

FPS = 16

# Supported by the workflow's 4n+1 frame-count convention.
# Qwen selects the most suitable length for each segment.
ALLOWED_FRAME_COUNTS = [49, 65, 81, 97]


# ============================================================
# Fixed Wan positive-prompt prefix
# Preserve the character description that already worked.
# The generated action is appended to this.
# ============================================================

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


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """
You generate prompts for a silent-charades animation pipeline.

The pipeline uses:
1. Qwen to plan each clip.
2. FLUX.2 Klein 4B Distilled to edit the original reference image
   into a starting frame.
3. Wan 2.2 I2V to animate that starting frame.

Generate one or more clips for an English word or phrase.

## Fixed constraints

- Exactly one existing humanoid stickman.
- No visible props, objects, additional characters, text, captions or sound.
- Facial expressions, body language and imaginary-object pantomime are allowed.
- Every segment starts independently from the original reference image.
- Never depend on events or movements from a previous clip.
- The character's design and visual style are already established.
- The application supplies the fixed character description and visual style.

## 1. Understand the target

Choose a recognizable physical representation of the intended meaning.
Prefer the common meaning of the word or phrase.

Use one clip by default. Split a word or phrase into two or three clips
only when splitting genuinely makes it easier to guess.

Splitting may use meaningful word components, recognizable syllable-based
gestures, or distinct clues. Do not split familiar gestures unnecessarily.

Each segment must have a clue identifying the target or component
represented by that clip.

## 2. Starting-frame edit for Klein

Generate a short edit prompt of one sentence, occasionally two.
Aim for 5-15 words.

The reference image is provided separately. Describe ONLY the specific
change needed to prepare the starting frame.

Do not describe the whole image. Do not add preservation instructions,
long explanations, style descriptions, or lists of restrictions.
The image editor already preserves unchanged parts of the reference.

Examples:

Target: thumbs up
Edit: Keep both arms lowered, ready to begin the thumbs-up gesture.

Target: sad
Edit: Change the mouth to a small downward curve.

Target: jumping
Edit: Slightly bend the knees in preparation for jumping.

Target: suspicious
Edit: Give the character a subtle suspicious facial expression.

Target: bowing
Edit: Tilt the torso forward slightly in preparation for bowing.

If no change is needed, say: Leave the reference image unchanged.

Prepare the initial pose, not the completed action.
Keep facial expressions simple and consistent with the existing cartoon.
Do not request skin folds, wrinkles, realistic facial anatomy or
unnecessary details.

## 3. Wan action prompt

The video model already understands natural-language actions well.
Give it simple, direct instructions instead of elaborate choreography.

Rules:
- Begin every action with "The stickman".
- Usually use 5-15 words.
- Describe what the character visibly does, not what the gesture means.
- Prefer one recognizable action per clip.
- Let Wan determine natural intermediate movements.
- Do not over-specify finger mechanics, joint angles or anatomy.
- Avoid unnecessary nodding, smiling, walking or secondary actions.
- Use facial expressions and body language when they help communicate
  the target.
- For abstract concepts, use a simple physical metaphor.
- Imaginary-object pantomime is allowed, but actual objects must not appear.
- Avoid complicated turns unless they are important to the target.
- Do not repeat the character description, art style, camera,
  background or visual constraints in the action.
- The action must flow naturally from the edited starting frame.

The edit prompt and action prompt serve different purposes:
the edit prepares the starting pose; the action describes the movement
that Wan should animate from that pose.

## 4. Choose the clip length

Choose the shortest clip that gives the action enough time to be
recognizable without feeling rushed.

The workflow runs at 16 FPS. Select exactly one supported frame count:

- 49 frames (~3.06 seconds): a simple gesture or clear action.
- 65 frames (~4.06 seconds): an action needing more setup or resolution.
- 81 frames (~5.06 seconds): several connected movements or a more
  involved emotional or abstract clue.
- 97 frames (~6.06 seconds): a genuinely complex performance.

These are guidelines, not mandatory action stages.
Do not add movements just to fill the clip.
A simple thumbs-up should normally use 49 frames.

Choose the length independently for each segment.

## Output

Return only JSON matching the supplied schema.

Each segment must contain:
- clue: the target or component represented by the clip.
- start_frame_edit: a short, direct image-edit instruction for Klein.
- action: the concise movement instruction for Wan.
- frame_count: one of 49, 65, 81 or 97.

Use one segment by default and no more than three.
Do not include markdown, explanations, complete prompts, negative prompts,
or any additional fields.
"""


# ============================================================
# Structured output schema
# ============================================================

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
                    "start_frame_edit": {
                        "type": "string"
                    },
                    "action": {
                        "type": "string"
                    },
                    "frame_count": {
                        "type": "integer",
                        "enum": ALLOWED_FRAME_COUNTS
                    }
                },
                "required": [
                    "clue",
                    "start_frame_edit",
                    "action",
                    "frame_count"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["segments"],
    "additionalProperties": False
}


# ============================================================
# Generate action plans
# ============================================================

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                f"Create silent-charades clips for: {TARGET}\n"
                "Generate a concise Klein edit prompt, a direct Wan "
                "action prompt, and the shortest suitable clip length "
                "for each segment."
            )
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


# ============================================================
# Validate output and construct downstream prompts
# ============================================================

segments = generated.get("segments")

if not isinstance(segments, list) or not 1 <= len(segments) <= 3:
    raise SystemExit(
        "Invalid result: expected between 1 and 3 segments."
    )

output_segments = []

for index, segment in enumerate(segments, start=1):
    clue = segment.get("clue", "").strip()
    start_frame_edit = segment.get("start_frame_edit", "").strip()
    action = segment.get("action", "").strip()
    frame_count = segment.get("frame_count")

    if not clue:
        raise SystemExit(f"Segment {index} has an empty clue.")

    if not start_frame_edit:
        raise SystemExit(
            f"Segment {index} has an empty Klein edit prompt."
        )

    if not action:
        raise SystemExit(f"Segment {index} has an empty action.")

    if not action.lower().startswith("the stickman"):
        raise SystemExit(
            f"Segment {index} action must start with "
            f"'The stickman': {action}"
        )

    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count not in ALLOWED_FRAME_COUNTS
    ):
        raise SystemExit(
            f"Segment {index} has unsupported frame count: "
            f"{frame_count!r}"
        )

    duration_seconds = frame_count / FPS

    # Send the short, unmodified LLM-generated instruction to Klein.
    # No prefix, suffix, or extra preservation text is added.
    klein_edit_prompt = start_frame_edit

    # Add the fixed character/style description for Wan.
    positive_prompt = BASE_PROMPT + action

    output_segments.append({
        "index": index,
        "clue": clue,

        # Pass directly to the Klein image-edit workflow.
        "start_frame_edit_instruction": start_frame_edit,
        "klein_edit_prompt": klein_edit_prompt,

        # Pass to Wan with the image returned by Klein.
        "action": action,
        "positive_prompt": positive_prompt,

        # Pass frame_count into the Wan workflow's length input.
        "frame_count": frame_count,
        "fps": FPS,
        "duration_seconds": duration_seconds
    })


# ============================================================
# Save results
# ============================================================

result = {
    "target": TARGET,
    "model": MODEL,
    "allowed_frame_counts": ALLOWED_FRAME_COUNTS,
    "segments": output_segments
}

OUTPUT_FILE.write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(json.dumps(result, indent=2, ensure_ascii=False))
print(f"\nSaved prompts to: {OUTPUT_FILE.resolve()}")