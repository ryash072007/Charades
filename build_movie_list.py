"""Build movies.json: popular English-language movies that are easy to mime.

Source: IMDb's non-commercial datasets (https://datasets.imdbws.com), ranked by
vote count. The LLM then marks each title as English-language or not and rates
how guessable it is as a silent, prop-free stickman mime. Run once:

    python build_movie_list.py [--candidates 1500]
"""
import csv
import gzip
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests

from llama_runs import chat_json

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = Path(__file__).parent / "movies.json"
IMDB = "https://datasets.imdbws.com"
BATCH = 15
MIN_MIMEABILITY = 2

# "very_popular" gets a full iconic-scene mime, so it is kept tiny: the most
# popular films of the last decade, plus a few timeless mega-hits by name.
RECENT_YEARS = 10
VERY_POPULAR_RECENT = 40  # top N by votes among recent titles
MEGA_HITS = {"titanic", "iron man", "spider-man"}

RATE_PROMPT = """
You help build a charades game where a single stickman mimes a movie title
silently, with no props, text or speech. For every movie in the list decide:

- english: true if the film was originally made in English (Hollywood and
  British productions are English, even if the director is foreign).
- mimeability: 1-5, how likely a general audience is to guess the title
  from a short silent mime. 5 = famous and has an obvious iconic action or
  mimeable words (Titanic, Jaws, Spider-Man). 3 = guessable from the
  words. 2 = hard but possible. 1 = abstract, vague, or only a name/place that cannot be mimed.

Return one entry per movie, using its list number "n".
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "movies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "english": {"type": "boolean"},
                    "mimeability": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                },
                "required": ["n", "english", "mimeability"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["movies"],
    "additionalProperties": False,
}


def fetch(name):
    dest = DATA_DIR / name
    if not dest.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"downloading {name}")
        with requests.get(f"{IMDB}/{name}", stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
    return dest


def top_movies(count):
    votes = {}
    with gzip.open(fetch("title.ratings.tsv.gz"), "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            votes[row["tconst"]] = int(row["numVotes"])

    movies = []
    with gzip.open(fetch("title.basics.tsv.gz"), "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row["titleType"] != "movie" or row["isAdult"] == "1":
                continue
            if row["tconst"] in votes:
                movies.append({
                    "id": row["tconst"],
                    "title": row["primaryTitle"],
                    "year": row["startYear"],
                    "votes": votes[row["tconst"]],
                })
    movies.sort(key=lambda m: -m["votes"])
    return movies[:count]


def assign_popularity(movies):
    """Set rank and popularity tier on a vote-sorted list, in place."""
    cutoff = date.today().year - RECENT_YEARS
    recent_rank = 0
    for rank, m in enumerate(movies, start=1):
        m["rank"] = rank
        recent = m["year"].isdigit() and int(m["year"]) >= cutoff
        if recent:
            recent_rank += 1
        top = (recent and recent_rank <= VERY_POPULAR_RECENT) or (
            m["title"].casefold() in MEGA_HITS
        )
        m["popularity"] = "very_popular" if top else "popular"


def main():
    if "--retier" in sys.argv:  # recompute tiers without re-rating
        kept = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        assign_popularity(kept)
        OUTPUT_FILE.write_text(
            json.dumps(kept, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        print(sum(m["popularity"] == "very_popular" for m in kept), "very_popular")
        return

    count = 1500
    if "--candidates" in sys.argv:
        count = int(sys.argv[sys.argv.index("--candidates") + 1])

    candidates = top_movies(count)
    print(f"{len(candidates)} candidates")

    kept = []
    t0 = time.time()
    for i in range(0, len(candidates), BATCH):
        batch = candidates[i:i + BATCH]
        listing = "\n".join(
            f"{n}. {m['title']} ({m['year']})" for n, m in enumerate(batch, 1)
        )
        result = chat_json(
            RATE_PROMPT, listing, SCHEMA, "rate", max_tokens=1500
        )
        rated = {r["n"]: r for r in result["movies"]}
        for n, m in enumerate(batch, 1):
            r = rated.get(n)
            if r and r["english"] and r["mimeability"] >= MIN_MIMEABILITY:
                kept.append({**m, "mimeability": r["mimeability"]})
        print(f"\r{i + len(batch)}/{len(candidates)} rated, {len(kept)} kept "
              f"({time.time() - t0:.0f}s)", end="", flush=True)
    print()

    assign_popularity(kept)

    OUTPUT_FILE.write_text(
        json.dumps(kept, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(kept)} movies to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
