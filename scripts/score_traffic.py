#!/usr/bin/env python3
"""
Score production traces — the online-evaluation pass.

This is deliberately written the hard way. The traffic runner already returned
every answer and trajectory in memory, and scoring that object would be two
lines. Instead this reads the spans back out of the platform, reconstructs each
turn from its span tree, scores it, and writes the results back onto the spans
as annotations.

That is the shape online scoring has to have, because in production there is no
return value. There is a trace, and a scorer sampling it after the fact. Writing
it this way means the same evaluators in evaluators.py serve both the offline
experiment and the live traffic, which is the property that makes an eval suite
worth maintaining.

Ground truth is joined from data/production.json by case_id rather than carried
on the span. In a real deployment that join is against whatever arrives later —
a thumbs-down, a support ticket, a labelling queue — and keeping it out of the
span is the honest version.

    python scripts/score_traffic.py
    python scripts/score_traffic.py --sample 0.5     # score half the traffic
    python scripts/score_traffic.py --no-annotate    # print only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from phoenix.client import Client  # noqa: E402
from phoenix.evals import evaluate_dataframe  # noqa: E402

from src.jwst_arize.spans import reconstruct  # noqa: E402
from src.jwst_arize.evaluators import (  # noqa: E402
    answer_quality_evaluator,
    appropriate_refusal_evaluator,
    citation_grounding,
    count_accuracy,
    required_citations,
    trajectory_efficiency,
    unsupported_citation,
)

def flatten_scores(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """phoenix.evals returns each Score as a dict in a `<name>_score` column.

    Unpack them into plain float and label columns so they can be averaged,
    grouped, and written back as annotations. A skipped score (score=None) stays
    NaN and drops out of every mean, which is the behaviour we want: a scorer
    that does not apply to a row must not be counted as a zero on it.
    """
    out = df.copy()
    for name in names:
        col = f"{name}_score"
        if col not in out.columns:
            continue
        payload = out[col]
        out[f"{name}__score"] = payload.apply(
            lambda v: v.get("score") if isinstance(v, dict) else None
        )
        out[f"{name}__label"] = payload.apply(
            lambda v: v.get("label") if isinstance(v, dict) else None
        )
    return out


def summarise(df: pd.DataFrame, scorer_names: list[str], group: str | None = None) -> str:
    lines = []
    groups = [(None, df)] if group is None else sorted(df.groupby(group), key=lambda g: str(g[0]))
    for key, sub in groups:
        if key is not None:
            lines.append(f"\n  {key}  (n={len(sub)})")
        for name in scorer_names:
            col = f"{name}__score"
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if vals.empty:
                continue
            pad = "    " if key is not None else "  "
            lines.append(f"{pad}{name:24s} {vals.mean() * 100:6.1f}%   n={len(vals)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.getenv("PRODUCTION_PROJECT", "jwst-production"))
    ap.add_argument("--endpoint", default=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"))
    ap.add_argument("--sample", type=float, default=1.0, help="Fraction of traces to score (0-1).")
    ap.add_argument("--max-spans", type=int, default=100_000,
                    help="Span fetch ceiling. The default page size is 1000, which silently "
                         "truncates a 200-turn run to a third of its traces.")
    ap.add_argument("--no-annotate", action="store_true")
    ap.add_argument("--out", default="runs/scored.csv")
    args = ap.parse_args()

    truth = {
        c["id"]: c for c in json.loads((ROOT / "data" / "production.json").read_text())["cases"]
    }

    client = Client(base_url=args.endpoint)
    spans = client.spans.get_spans_dataframe(
        project_identifier=args.project, limit=args.max_spans, timeout=120
    )
    print(f"read {len(spans)} spans from {args.project}")

    records = reconstruct(spans)
    print(f"reconstructed {len(records)} agent turns from their span trees")

    if args.sample < 1.0:
        rng = random.Random(42)
        records = [r for r in records if rng.random() < args.sample]
        print(f"sampled {len(records)} ({args.sample:.0%})")

    rows = []
    for r in records:
        case = truth.get(r["case_id"])
        if case is None:
            continue
        shape = case["metadata"].get("shape") or case["category"]
        rows.append(
            {
                **r,
                "question": case["question"],
                "category": case["category"],
                "shape": shape,
                "expected_answerable": case["expected"]["answerable"],
                "expected_photo_ids": case["expected"]["photo_ids"],
                "expected_answer": case["expected"]["answer"] or "",
                "optimal_tool_calls": {"lookup": 1, "count": 1, "search_and_verify": 2,
                                       "comparative": 2}.get(shape, 1),
            }
        )
    if not rows:
        raise SystemExit("No traces matched data/production.json. Run scripts/run_traffic.py first.")

    df = pd.DataFrame(rows)
    answerable = df[df["expected_answerable"]].copy()
    unanswerable = df[~df["expected_answerable"]].copy()
    print(f"{len(answerable)} answerable | {len(unanswerable)} unanswerable\n")

    code_evals = [citation_grounding, unsupported_citation, required_citations,
                  count_accuracy, trajectory_efficiency]

    # The judges are split by row type on purpose: asking "did it refuse?" about
    # an answerable question is meaningless, and it costs a model call to find
    # that out.
    scored_parts = []
    if len(answerable):
        scored_parts.append(
            evaluate_dataframe(dataframe=answerable, evaluators=code_evals + [answer_quality_evaluator()])
        )
    if len(unanswerable):
        scored_parts.append(
            evaluate_dataframe(dataframe=unanswerable, evaluators=code_evals + [appropriate_refusal_evaluator()])
        )
    scored = pd.concat(scored_parts, ignore_index=True)

    names = ["citation_grounding", "unsupported_citation", "appropriate_refusal",
             "required_citations", "count_accuracy", "trajectory_efficiency", "answer_quality"]
    scored = flatten_scores(scored, names)

    print("═" * 62)
    print("  PRODUCTION TRAFFIC")
    print("═" * 62)
    print(summarise(scored, names))
    print("\n" + "─" * 62)
    print("  BY CATEGORY")
    print("─" * 62)
    print(summarise(scored, names, group="category"))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    print(f"\nwrote {args.out}")

    if not args.no_annotate:
        wrote = 0
        for name in names:
            col = f"{name}__score"
            if col not in scored.columns:
                continue
            ann = scored[["span_id", col, f"{name}__label"]].copy()
            ann = ann[pd.to_numeric(ann[col], errors="coerce").notna()]
            if ann.empty:
                continue
            ann = ann.rename(columns={col: "score", f"{name}__label": "label"})
            ann["annotation_name"] = name
            client.spans.log_span_annotations_dataframe(
                dataframe=ann.set_index("span_id"),
                annotator_kind="LLM" if name in ("answer_quality", "appropriate_refusal") else "CODE",
            )
            wrote += len(ann)
        print(f"annotated {wrote} span scores — they are on the traces in the UI now")


if __name__ == "__main__":
    main()
