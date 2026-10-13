"""Turn an English movie title into a sequence of charades clips.

Clip 1 is always the "movie" marker (cranking an old film camera). The rest
are chosen by the LLM: for very famous films an iconic scene or character,
otherwise the title's words/syllables. Each clip goes through llama_runs.

Usage: python charades_plan.py "The Lion King" [--think]
"""
import json
import sys
import time
from pathlib import Path

from llama_runs import MODEL, chat_json, generate_clip

OUTPUT_FILE = Path("charades_plan.json")

MOVIE_MARKER = (
    "the word 'movie': pantomime turning the crank of an old hand-cranked "
    "film camera"
)

MAX_PIECES = 4
MOVIES_FILE = Path(__file__).parent / "movies.json"

PLANNER_PROMPT = f"""
You plan a silent charades performance for an English movie title.
The performer is one stickman who can only mime: no props, objects,
text, speech or sound. The audience already knows the answer is a movie
(a "movie" clip is shown first), so do not mime that.

Break the title into 1 to {MAX_PIECES} pieces to be mimed one after another.
Players should find it easy, so prefer fewer, clearer pieces.

Two approaches:
- iconic_scene: mime the film's most iconic character, scene or action with
  one simple physical action (e.g. Spider-Man shooting webs from a wrist,
  arms spread wide at a ship's bow). Only valid for films that are
  unmistakable from that single action.
- title_words: drop filler words (the, a, of, and), then mime each
  meaningful word, or a sound-alike syllable when a word cannot be mimed.
  Put the words in the order they are read.

For each piece give "mime": a short plain description of what to mime
(a word, a sound-alike, or a scene), and nothing about camera or style.

Return only JSON matching the schema.
"""

GUIDANCE = {
    "very_popular": (
        "This is a very famous film. Use iconic_scene with a single piece, "
        "unless no one action is unmistakable."
    ),
    "popular": (
        "This is a well-known film. Use iconic_scene with one or two pieces "
        "if a clear action exists; otherwise title_words."
    ),
    "unlisted": (
        "This film is not widely known. Use title_words and mime every "
        "meaningful word as its own piece (up to the limit)."
    ),
}


def lookup_popularity(title):
    if MOVIES_FILE.exists():
        wanted = title.casefold().strip()
        for m in json.loads(MOVIES_FILE.read_text(encoding="utf-8")):
            if m["title"].casefold() == wanted:
                return m["popularity"]
    return "unlisted"


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "approach": {"type": "string", "enum": ["iconic_scene", "title_words"]},
        "pieces": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_PIECES,
            "items": {
                "type": "object",
                "properties": {"mime": {"type": "string"}},
                "required": ["mime"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["approach", "pieces"],
    "additionalProperties": False,
}


def plan_movie(title, thinking=False):
    popularity = lookup_popularity(title)
    plan = chat_json(
        PLANNER_PROMPT,
        f"Movie title: {title}\n{GUIDANCE[popularity]}",
        PLAN_SCHEMA, "plan", thinking,
    )
    pieces = [p["mime"].strip() for p in plan["pieces"] if p["mime"].strip()]
    if not pieces:
        raise SystemExit("Planner returned no pieces.")

    clips = []
    for kind, target in [("marker", MOVIE_MARKER)] + [("piece", m) for m in pieces]:
        clip = generate_clip(target, thinking)
        clips.append({"index": len(clips) + 1, "kind": kind, **clip})

    return {
        "title": title,
        "model": MODEL,
        "popularity": popularity,
        "approach": plan["approach"],
        "clips": clips,
        "total_frames": sum(c["frame_count"] for c in clips),
    }


if __name__ == "__main__":
    thinking = "--think" in sys.argv[1:]
    title = " ".join(a for a in sys.argv[1:] if a != "--think").strip()
    if not title:
        raise SystemExit('Usage: python charades_plan.py "Movie Title" [--think]')

    t0 = time.time()
    result = plan_movie(title, thinking)
    print(f"[total {time.time() - t0:.1f}s]")

    OUTPUT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{title}: {result['popularity']} -> {result['approach']}")
    for c in result["clips"]:
        print(f"  {c['index']}. [{c['kind']}] {c['target']}\n"
              f"     {c['action']} ({c['frame_count']}f)")
    print(f"\nSaved plan to: {OUTPUT_FILE.resolve()}")
