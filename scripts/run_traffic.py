#!/usr/bin/env python3
"""
Replay data/production.json through the agent — the "production traffic" run.

This is not an experiment. There is no dataset object, no expected output
attached to the run, and no pass/fail at the end. It is the agent answering
questions and emitting traces, which is all you get in production.

Each turn carries OpenInference metadata (category, whether the corpus can
answer it) so the scoring pass can join a span back to what should have
happened. In a real deployment that join key is a user ID or a request ID and
ground truth arrives later — from a thumbs-down, a support ticket, or a human
review queue. The mechanism is the same; only the source of truth differs.

    python scripts/run_traffic.py                       # all 200
    python scripts/run_traffic.py --limit 40            # a slice
    python scripts/run_traffic.py --prompt-version v2-concise
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from openinference.instrumentation import using_attributes  # noqa: E402

from src.jwst_arize.agent import DEFAULT_MODEL, run_agent  # noqa: E402
from src.jwst_arize.tracing import destination, flush, setup_tracing  # noqa: E402

# What an ideal trajectory costs, by question shape. Used by
# trajectory_efficiency; an unanswerable question still needs one search to
# establish that there is nothing there.
OPTIMAL_CALLS = {
    "lookup": 1,
    "count": 1,
    "search_and_verify": 2,
    "comparative": 2,
    "absent_subject": 1,
    "out_of_domain": 1,
    "false_premise": 1,
}


def optimal_for(case: dict) -> int:
    shape = case["metadata"].get("shape") or case["category"]
    return OPTIMAL_CALLS.get(shape, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt-version", default="v2-concise",
                    help="The shipped prompt. v2-concise is what passed the gate.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--project", default="jwst-production")
    ap.add_argument("--out", default="runs/traffic.jsonl")
    args = ap.parse_args()

    payload = json.loads((ROOT / "data" / "production.json").read_text())
    cases = payload["cases"]
    if args.limit:
        # Spread across categories rather than taking the first N, which would
        # be all `answerable` and would report a flattering number.
        by_cat: dict[str, list] = {}
        for c in cases:
            by_cat.setdefault(c["category"], []).append(c)
        cases, i = [], 0
        while len(cases) < args.limit and any(len(v) > i for v in by_cat.values()):
            for bucket in by_cat.values():
                if i < len(bucket) and len(cases) < args.limit:
                    cases.append(bucket[i])
            i += 1

    setup_tracing(args.project)
    print(f"traffic → {destination()}  project={args.project}")
    print(f"{len(cases)} questions | prompt {args.prompt_version} | model {args.model}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def one(case: dict) -> dict:
        meta = {
            "case_id": case["id"],
            "category": case["category"],
            "expected_answerable": case["expected"]["answerable"],
            "prompt_version": args.prompt_version,
        }
        # Metadata and tags ride on every span in the turn, so the Phoenix / AX
        # UI can filter and group by them without any extra plumbing.
        with using_attributes(metadata=meta, tags=[case["category"], args.prompt_version]):
            result = run_agent(case["question"], args.prompt_version, args.model)
        return {
            "case_id": case["id"],
            "question": case["question"],
            "category": case["category"],
            "shape": case["metadata"].get("shape") or case["category"],
            "expected_answerable": case["expected"]["answerable"],
            "expected_photo_ids": case["expected"]["photo_ids"],
            "expected_answer": case["expected"]["answer"],
            "optimal_tool_calls": optimal_for(case),
            "answer": result.answer,
            "trajectory": [t.__dict__ for t in result.trajectory],
            "prompt_version": args.prompt_version,
            "model": args.model,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = []
        for i, row in enumerate(pool.map(one, cases), 1):
            rows.append(row)
            if i % 20 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)}")

    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    flush()
    print(f"\n{len(rows)} turns in {time.time() - started:.0f}s → {args.out}")
    print("Traces are in the UI now. Nothing has been scored yet — that is scripts/score_traffic.py.")


if __name__ == "__main__":
    main()
