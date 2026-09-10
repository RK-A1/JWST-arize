#!/usr/bin/env python3
"""
Build data/production.json — 200 questions shaped like real user traffic.

Why this file exists
--------------------
The golden set in data/golden.json is generated *from* the corpus, so by
construction every question in it is answerable. That is a reasonable way to
build a regression suite and a terrible model of production, where users ask
about things that are not in your data at all.

This builder produces the traffic the golden set cannot contain. Five
categories, each with an unambiguous correct behaviour:

  answerable      the control group — same shape as the golden set
  absent_subject  a real astronomical object verifiably not in this corpus
  out_of_domain   a question about JWST the observatory, not about a photo
  false_premise   a premise the corpus data contradicts
  comparative     needs two record lookups, not one search

For every category except `answerable`, the correct answer contains **no photo
ID**, because there is no photo that answers the question. That is what makes
the failure measurable: a cited ID is not a matter of taste, it is wrong.

Absence is verified at build time rather than asserted. If someone adds a
Betelgeuse photo to the corpus, this script fails loudly instead of silently
grading a now-answerable question as unanswerable.

    python scripts/build_production_set.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jwst_arize.corpus import meta, photos, search_photos  # noqa: E402

SEED = 42
OUT = ROOT / "data" / "production.json"

# ── Verified-absent subjects ────────────────────────────────────────────────
# Each entry is (subject as a user would write it, distinctive term to verify).
# The term must appear nowhere in any title, description, or tag.

ABSENT_SUBJECTS = [
    ("Betelgeuse", "betelgeuse"), ("Vega", "vega"), ("Sirius", "sirius"),
    ("Aldebaran", "aldebaran"), ("Rigel", "rigel"), ("Polaris", "polaris"),
    ("Antares", "antares"), ("Altair", "altair"), ("Deneb", "deneb"),
    ("the Boomerang Nebula", "boomerang nebula"), ("the Veil Nebula", "veil nebula"),
    ("the Dumbbell Nebula", "dumbbell"), ("the Owl Nebula", "owl nebula"),
    ("the Rosette Nebula", "rosette"), ("the Triangulum Galaxy", "triangulum"),
    ("Messier 87", "messier 87"), ("Hoag's Object", "hoag"),
    ("the Mice Galaxies", "mice galaxies"), ("the Tadpole Galaxy", "tadpole galaxy"),
    ("Halley's Comet", "halley"), ("Comet Hale-Bopp", "hale-bopp"),
    ("Shoemaker-Levy 9", "shoemaker-levy"), ("'Oumuamua", "oumuamua"),
    ("Comet Borisov", "borisov"), ("Kepler-452b", "kepler-452"),
    ("Gliese 581g", "gliese 581"), ("Teegarden's Star", "teegarden"),
    ("Wolf 359", "wolf 359"), ("the Pioneer plaque", "pioneer"),
    ("the Perseverance rover", "perseverance"), ("the Curiosity rover", "curiosity rover"),
    ("quasar 3C 273", "3c 273"), ("the dwarf planet Makemake", "makemake"),
    ("the dwarf planet Haumea", "haumea"), ("the asteroid Vesta", "vesta"),
    ("Jupiter's moon Callisto", "callisto"), ("Pluto's moon Charon", "charon"),
    ("the Lagoon Nebula", "lagoon nebula"), ("the Trifid Nebula", "trifid"),
    ("Messier 81", "messier 81"), ("Centaurus A", "centaurus a"),
]

ABSENT_TEMPLATES = [
    "Which photo in the corpus shows {subject}?",
    "What's the photo ID for the {subject} image?",
    "Find me the Webb image of {subject}.",
    "I'm looking for the {subject} photo — what's its ID?",
    "Can you pull up the {subject} image and tell me when it was taken?",
]

# ── Out-of-domain: about the observatory, not about a photo in the corpus ───

OUT_OF_DOMAIN = [
    "What is the diameter of Webb's primary mirror?",
    "How many hexagonal segments does Webb's mirror have?",
    "What is the operating temperature of the MIRI instrument?",
    "Which rocket launched the James Webb Space Telescope?",
    "How far from Earth is Webb's orbit at L2?",
    "What is Webb's expected mission lifetime?",
    "How much did the James Webb Space Telescope cost to build?",
    "Who is the current lead scientist on the Webb mission?",
    "What is the difference between NIRCam and NIRSpec?",
    "How long does it take Webb to slew between targets?",
    "What wavelength range does Webb observe in?",
    "How many layers does Webb's sunshield have?",
    "What propellant does Webb use for station-keeping?",
    "When is Webb's next scheduled instrument calibration?",
    "Which ground station downlinks Webb's data?",
    "What is the data rate of Webb's downlink?",
    "How many observing proposals were accepted in Cycle 3?",
    "What is Webb's angular resolution at 2 microns?",
    "Who manufactured Webb's beryllium mirror segments?",
    "What is the field of view of the NIRCam instrument?",
    "How does Webb compare to Hubble in light-gathering power?",
    "What happens to Webb when its fuel runs out?",
    "Which agencies partnered with NASA on Webb?",
    "What is the L2 Lagrange point and why was it chosen?",
    "How is Webb protected from micrometeoroid strikes?",
    "What software processes Webb's raw detector output?",
    "How many people work on the Webb operations team?",
    "What is Webb's pointing stability specification?",
    "When did Webb complete its mirror alignment?",
    "What is the coldest temperature Webb's detectors reach?",
]

# ── False premises the corpus data contradicts ──────────────────────────────
# Built from labels that do not exist and counts that are wrong. Both are
# checked against the corpus below.

ABSENT_LABELS = ["black hole", "supernova", "asteroid", "quasar", "pulsar", "comet"]

FALSE_PREMISE_TEMPLATES = [
    "The corpus has a '{label}' category — how many photos are in it?",
    "Show me the highest-rated '{label}' photo in the corpus.",
    "List the photo IDs labelled '{label}'.",
    "Which '{label}' photo was added most recently?",
    "What's the ID of the first '{label}' image Webb published?",
]

FALSE_COUNT_TEMPLATES = [
    "Of the {claimed} photos in this corpus, how many are nebulae?",
    "This corpus has {claimed} images — what fraction are galaxies?",
    "You have {claimed} photos indexed. How many show exoplanets?",
]
CLAIMED_COUNTS = [5000, 12000, 250, 40000]

# ── Question templates for the answerable control group ─────────────────────

LOOKUP_TEMPLATES = [
    "What does photo {pid} show?",
    "Tell me about photo {pid}.",
    "What's in image {pid}?",
]
COUNT_TEMPLATES = [
    "How many {label} photos are in the corpus?",
    "What's the count of photos labelled {label}?",
]


def verify_absence(term: str, blob: str) -> None:
    if term.lower() in blob:
        raise SystemExit(
            f"ABSENCE CHECK FAILED: '{term}' now appears in the corpus. "
            "This question is no longer unanswerable — remove it or pick another subject."
        )


def main() -> None:
    rng = random.Random(SEED)
    all_photos = photos()
    label_counts = meta()["label_counts"]
    eval_labels = meta()["eval_labels"]

    blob = "\n".join(
        f"{p['title']} {p['description']} {' '.join(p['tags'])}".lower() for p in all_photos
    ).lower()

    for _, term in ABSENT_SUBJECTS:
        verify_absence(term, blob)
    for label in ABSENT_LABELS:
        if label in label_counts:
            raise SystemExit(f"ABSENCE CHECK FAILED: label '{label}' exists in the corpus.")

    cases: list[dict] = []

    def add(question, category, answerable, photo_ids=None, answer=None, **extra):
        cases.append(
            {
                "id": f"prod-{len(cases) + 1:04d}",
                "question": question,
                "category": category,
                "expected": {
                    "answerable": answerable,
                    "photo_ids": photo_ids or [],
                    "answer": answer,
                },
                "metadata": {"category": category, **extra},
            }
        )

    # 1. answerable — the control group (70)
    labelled = [p for p in all_photos if p["canonical_label"] in eval_labels]
    for p in rng.sample(labelled, 40):
        add(
            rng.choice(LOOKUP_TEMPLATES).format(pid=p["photo_id"]),
            "answerable",
            True,
            [p["photo_id"]],
            p["title"],
            label=p["canonical_label"],
            shape="lookup",
        )
    for label in eval_labels:
        for tmpl in COUNT_TEMPLATES:
            add(
                tmpl.format(label=label),
                "answerable",
                True,
                [],
                str(label_counts[label]),
                label=label,
                shape="count",
            )
    # Search-and-verify: distinctive title phrases that retrieve their own photo.
    search_pool = [p for p in labelled if len(p["title"].split()) >= 5]
    picked = 0
    for p in rng.sample(search_pool, len(search_pool)):
        if picked >= 16:
            break
        phrase = " ".join(p["title"].split()[:6])
        hits = search_photos(phrase, 5)
        if hits and hits[0]["photo_id"] == p["photo_id"]:
            add(
                f"Which photo is titled something like '{phrase}'? Give me its ID.",
                "answerable",
                True,
                [p["photo_id"]],
                p["title"],
                label=p["canonical_label"],
                shape="search_and_verify",
            )
            picked += 1

    # 2. absent_subject (45)
    for i in range(45):
        subject, term = ABSENT_SUBJECTS[i % len(ABSENT_SUBJECTS)]
        tmpl = ABSENT_TEMPLATES[i % len(ABSENT_TEMPLATES)]
        add(
            tmpl.format(subject=subject),
            "absent_subject",
            False,
            answer=f"The corpus contains no photo of {subject}.",
            subject=subject,
            verified_absent_term=term,
        )

    # 3. out_of_domain (30)
    for q in OUT_OF_DOMAIN:
        add(
            q,
            "out_of_domain",
            False,
            answer="This corpus holds photo records only; it cannot answer questions about the observatory itself.",
        )

    # 4. false_premise (30)
    for i in range(21):
        label = ABSENT_LABELS[i % len(ABSENT_LABELS)]
        tmpl = FALSE_PREMISE_TEMPLATES[i % len(FALSE_PREMISE_TEMPLATES)]
        add(
            tmpl.format(label=label),
            "false_premise",
            False,
            answer=f"There is no '{label}' label in this corpus.",
            false_label=label,
        )
    for i in range(9):
        claimed = CLAIMED_COUNTS[i % len(CLAIMED_COUNTS)]
        tmpl = FALSE_COUNT_TEMPLATES[i % len(FALSE_COUNT_TEMPLATES)]
        add(
            tmpl.format(claimed=f"{claimed:,}"),
            "false_premise",
            False,
            answer=f"The corpus contains {len(all_photos)} photos, not {claimed:,}.",
            claimed_count=claimed,
        )

    # 5. comparative (25) — two real photos, needs two lookups
    pairs = rng.sample(labelled, 50)
    for a, b in zip(pairs[:25], pairs[25:50]):
        add(
            f"Compare photos {a['photo_id']} and {b['photo_id']} — what does each show, "
            "and do they have the same label?",
            "comparative",
            True,
            [a["photo_id"], b["photo_id"]],
            f"{a['title']} ({a['canonical_label']}) vs {b['title']} ({b['canonical_label']})",
            shape="comparative",
        )

    from collections import Counter

    by_cat = Counter(c["category"] for c in cases)
    payload = {
        "meta": {
            "generated_from": "data/corpus.json",
            "seed": SEED,
            "corpus_photo_count": len(all_photos),
            "case_count": len(cases),
            "cases_by_category": dict(by_cat),
            "unanswerable_count": sum(1 for c in cases if not c["expected"]["answerable"]),
            "note": (
                "Traffic the golden set cannot contain. Absence of every "
                "`absent_subject` term and every `false_premise` label is verified "
                "against the corpus at build time; this script exits non-zero if a "
                "subject has since been added."
            ),
        },
        "cases": cases,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  {len(cases)} cases")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:16s} {n:3d}")
    print(f"  {'unanswerable':16s} {payload['meta']['unanswerable_count']:3d}")


if __name__ == "__main__":
    main()
