"""
The JWST corpus: 979 NASA Webb Flickr photos with weak ground-truth labels.

Ported from the TypeScript original so the agent under test is the same agent,
against the same data, as the Braintrust project this repo is a sequel to. That
matters: the argument here is that the *dataset* was too easy, not that the
agent or the retriever changed.

Search is plain token overlap. It is deliberately the least interesting code in
the repo — a deterministic retriever means a failure is always attributable to
the agent rather than to retrieval noise.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class Photo(TypedDict):
    photo_id: str
    title: str
    description: str
    tags: list[str]
    canonical_label: str
    date_taken: str | None
    image_url: str


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    return json.loads((DATA_DIR / "corpus.json").read_text())


def meta() -> dict[str, Any]:
    return _load()["meta"]


def photos() -> list[Photo]:
    return _load()["photos"]


@lru_cache(maxsize=1)
def _by_id() -> dict[str, Photo]:
    return {p["photo_id"]: p for p in photos()}


def get_photo(photo_id: str) -> Photo | None:
    return _by_id().get(photo_id)


def photo_exists(photo_id: str) -> bool:
    return photo_id in _by_id()


def count_by_label() -> dict[str, int]:
    return dict(meta()["label_counts"])


# ── Search ──────────────────────────────────────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "and", "or", "to", "for", "with", "is",
    "are", "was", "were", "by", "at", "from", "as", "its", "this", "that",
    "what", "which", "how", "many", "find", "photo", "photos", "image", "images",
}


def tokenize(text: str) -> list[str]:
    cleaned = "".join(c if (c.isalnum() or c in " -") else " " for c in text.lower())
    return [t for t in cleaned.split() if len(t) > 1 and t not in STOPWORDS]


@lru_cache(maxsize=1)
def _indexes() -> tuple[list[set[str]], list[set[str]]]:
    docs, titles = [], []
    for p in photos():
        docs.append(set(tokenize(f"{p['title']} {p['description']} {' '.join(p['tags'])}")))
        titles.append(set(tokenize(p["title"])))
    return docs, titles


def search_photos(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Rank by query-term overlap, weighting title matches double.

    Ties break on photo_id so results are stable across runs. Returns [] when
    nothing matches — which is the case the whole repo is about, because an
    agent that answers anyway has nowhere to have got the answer from.
    """
    terms = tokenize(query)
    if not terms:
        return []

    docs, titles = _indexes()
    scored = []
    for i, photo in enumerate(photos()):
        score = 0
        for term in terms:
            if term in titles[i]:
                score += 2
            elif term in docs[i]:
                score += 1
        if score > 0:
            scored.append((score, photo))

    scored.sort(key=lambda s: (-s[0], s[1]["photo_id"]))
    return [
        {
            "photo_id": p["photo_id"],
            "title": p["title"],
            "canonical_label": p["canonical_label"],
            "score": score,
        }
        for score, p in scored[:limit]
    ]
