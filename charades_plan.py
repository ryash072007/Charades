"""Turn an English movie title into a sequence of charades clips.

Only the very most popular recent films get a single iconic-scene mime;
everything else is mimed word by word. Each clip goes through llama_runs.

Usage: python charades_plan.py "The Lion King" [--think]
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

from build_movie_list import MEGA_HITS, RECENT_YEARS
from llama_runs import MODEL, chat_json, generate_clip

OUTPUT_FILE = Path("charades_plan.json")

MAX_PIECES = 4
MOVIES_FILE = Path(__file__).parent / "movies.json"

PLANNER_PROMPT = f"""
You plan a silent charades performance for an English movie title.
The performer is one stickman who can only mime: no props, objects,
text, speech or sound. The audience already knows the answer is a movie,
so never mime "movie", a camera or filming.

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
        "This is one of the most famous films of all. Use iconic_scene with "
        "a single piece: its most unmistakable character or action."
    ),
    "popular": (
        "This is a well-known film. Use title_words and mime the meaningful "
        "words, keeping the pieces few. Use iconic_scene only if one action "
        "is truly unmistakable."
    ),
    "unlisted": (
        "This film is not widely known. Use title_words and mime every "
        "meaningful word as its own piece (up to the limit)."
    ),
}


JUDGE_PROMPT = """
You judge how famous an English-language movie is to a general audience.
- very_popular: a recent global blockbuster that almost anyone would
  recognise from its title or a single iconic scene (Avengers: Endgame,
  Barbie, Top Gun: Maverick).
- popular: well known to most movie-goers, but not a household name.
- obscure: many people would not know it, or you do not recognise the title.
Be strict: choose very_popular only for true household names, and obscure
whenever you are unsure the movie exists or is widely known.
Also give the year the movie was released (0 if unknown).
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "popularity": {
            "type": "string",
            "enum": ["very_popular", "popular", "obscure"],
        },
        "release_year": {"type": "integer"},
    },
    "required": ["popularity", "release_year"],
    "additionalProperties": False,
}


def judge_popularity(title, thinking=False):
    """Ask the LLM how famous a title is when it is not in movies.json."""
    result = chat_json(
        JUDGE_PROMPT, f"Movie title: {title}", JUDGE_SCHEMA, "judge", thinking
    )
    popularity = result["popularity"]
    if popularity == "obscure":
        return "unlisted"
    # The iconic-scene tier is only for recent films or the named mega-hits.
    recent = result["release_year"] >= date.today().year - RECENT_YEARS
    if popularity == "very_popular" and not (recent or title.casefold().strip() in MEGA_HITS):
        return "popular"
    return popularity


def lookup_popularity(title):
    if MOVIES_FILE.exists():
        wanted = title.casefold().strip()
        for m in json.loads(MOVIES_FILE.read_text(encoding="utf-8")):
            if m["title"].casefold() == wanted:
                return m["popularity"]
    return None


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


def plan_schema(popularity):
    """Only the very popular tier may use an iconic scene; the rest are title words."""
    schema = json.loads(json.dumps(PLAN_SCHEMA))
    if popularity != "very_popular":
        schema["properties"]["approach"]["enum"] = ["title_words"]
    return schema


def plan_movie(title, thinking=False):
    popularity = lookup_popularity(title)
    source = "catalog"
    if popularity is None:
        popularity = judge_popularity(title, thinking)
        source = "llm"
    plan = chat_json(
        PLANNER_PROMPT,
        f"Movie title: {title}\n{GUIDANCE[popularity]}",
        plan_schema(popularity), "plan", thinking,
    )
    pieces = [p["mime"].strip() for p in plan["pieces"] if p["mime"].strip()]
    if not pieces:
        raise SystemExit("Planner returned no pieces.")

    clips = []
    for mime in pieces:
        clips.append({"index": len(clips) + 1, **generate_clip(mime, thinking)})

    return {
        "title": title,
        "model": MODEL,
        "popularity": popularity,
        "popularity_source": source,
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
    print(f"{title}: {result['popularity']} ({result['popularity_source']}) ->{result['approach']}")
    for c in result["clips"]:
        print(f"  {c['index']}. {c['target']}\n"
              f"     {c['action']} ({c['frame_count']}f)")
    print(f"\nSaved plan to: {OUTPUT_FILE.resolve()}")
