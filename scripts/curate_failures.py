#!/usr/bin/env python3
"""
Turn production failures into a regression test.

This is the beat the golden set could not reach. `unsupported_citation` cannot
be gated against data/golden.json, because every case in it is answerable and
the scorer skips every row. The only place it has anything to measure is
traffic — and traffic is not a test suite, because it is different every day.

So: read the production traces, find the turns where the agent cited a photo for
a question with no answer, and curate exactly those into a **Phoenix dataset**.
Each example keeps the `span_id` of the trace it came from, so a failing row in
the regression suite links back to the production conversation that produced it.

From that point on the failure is a fixture rather than an anecdote, and
`unsupported_citation` has a gate it can fail.

    python scripts/curate_failures.py                  # build the dataset
    python scripts/curate_failures.py --run-experiment # then grade against it
    python scripts/curate_failures.py --prompt-version v1-grounded --run-experiment
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from phoenix.client import Client  # noqa: E402

from src.jwst_arize.agent import DEFAULT_MODEL, run_agent  # noqa: E402
from src.jwst_arize.evaluators import unsupported_citation  # noqa: E402
from src.jwst_arize.experiment_evaluators import FAILURES  # noqa: E402
from src.jwst_arize.spans import reconstruct  # noqa: E402
from src.jwst_arize.tracing import destination, flush, setup_tracing  # noqa: E402

DATASET_NAME = "jwst-production-failures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=os.getenv("PRODUCTION_PROJECT", "jwst-production"))
    ap.add_argument("--endpoint", default=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"))
    ap.add_argument("--dataset", default=DATASET_NAME)
    ap.add_argument("--max-spans", type=int, default=100_000)
    ap.add_argument("--run-experiment", action="store_true",
                    help="After curating, run the agent against the dataset and grade it.")
    ap.add_argument("--prompt-version", default="v2-concise")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--gate", action="store_true",
                    help="Exit non-zero if failureGates in thresholds.json are breached. "
                         "Off by default: this suite fails on purpose until the fix lands.")
    args = ap.parse_args()

    truth = {
        c["id"]: c for c in json.loads((ROOT / "data" / "production.json").read_text())["cases"]
    }

    client = Client(base_url=args.endpoint)
    spans = client.spans.get_spans_dataframe(
        project_identifier=args.project, limit=args.max_spans, timeout=120
    )
    records = reconstruct(spans)
    print(f"read {len(spans)} spans → {len(records)} turns from '{args.project}'")

    # Only the code scorer runs here. Finding failures must be cheap and
    # deterministic: this is a filter over production, not an evaluation of it.
    failures: list[dict[str, Any]] = []
    for record in records:
        case = truth.get(record["case_id"])
        if case is None or case["expected"]["answerable"]:
            continue
        score = unsupported_citation.evaluate(
            {"answer": record["answer"], "expected_answerable": False}
        )[0]
        if score.score == 0.0:
            failures.append(
                {
                    "question": case["question"],
                    "why_unanswerable": case["expected"]["answer"],
                    "category": case["category"],
                    "case_id": case["id"],
                    "cited_ids": ",".join(score.metadata.get("cited_ids", [])),
                    "span_id": record["span_id"],
                }
            )

    if not failures:
        print("No failures found. Run scripts/run_traffic.py first.")
        return

    by_cat = Counter(f["category"] for f in failures)
    print(f"\n{len(failures)} turns cited a photo for a question with no answer:")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:16s} {n:3d}")

    df = pd.DataFrame(failures)
    dataset = client.datasets.create_dataset(
        name=args.dataset,
        dataframe=df,
        input_keys=["question"],
        output_keys=["why_unanswerable"],
        metadata_keys=["category", "case_id", "cited_ids"],
        # Each example remembers the production trace it was curated from.
        span_id_key="span_id",
        dataset_description=(
            "Production turns where the agent cited a photo ID for a question the "
            "corpus cannot answer. Curated from traces, not written by hand."
        ),
    )
    print(f"\ndataset '{args.dataset}' — {len(dataset.examples)} examples")
    try:
        print(f"  {args.endpoint}/datasets/{dataset.id}/examples")
    except Exception:
        pass

    if not args.run_experiment:
        print("\nRe-run with --run-experiment to grade the agent against it.")
        return

    setup_tracing(args.project)
    print(f"\nexperiment → {destination()}")

    def task(input: dict[str, Any]) -> dict[str, Any]:
        result = run_agent(input["question"], args.prompt_version, args.model)
        return {"answer": result.answer, "trajectory": [t.__dict__ for t in result.trajectory]}

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=FAILURES,
        experiment_name=f"failures-{args.prompt_version}",
        experiment_description=(
            f"Curated production failures — prompt {args.prompt_version}, model {args.model}."
        ),
        experiment_metadata={"prompt_version": args.prompt_version, "model": args.model},
        repetitions=args.repetitions,
        print_summary=False,
    )
    flush()

    from collections import defaultdict

    buckets: dict[str, list[float]] = defaultdict(list)
    for run in experiment.get("evaluation_runs", []) or []:
        result = getattr(run, "result", None) or {}
        name = getattr(run, "name", None) or result.get("name")
        if name and result.get("score") is not None:
            buckets[name].append(float(result["score"]))

    print("\n" + "═" * 58)
    print(f"  CURATED FAILURES — {args.prompt_version}")
    print("═" * 58)
    for name in sorted(buckets):
        vals = buckets[name]
        print(f"  {name:24s} {sum(vals) / len(vals):6.1%}   n={len(vals)}")
    try:
        print(f"\n  {client.experiments.get_experiment_url(experiment['dataset_id'], experiment['experiment_id'])}")
    except Exception:
        pass
    print(f"\n  These {len(dataset.examples)} rows were a production incident. They are now a fixture.")

    gates = json.loads((ROOT / "thresholds.json").read_text()).get("failureGates", {})
    breached = [
        f"{name}: {sum(buckets[name]) / len(buckets[name]):.1%} is below its floor of {gate['floor']:.0%}"
        for name, gate in gates.items()
        if name in buckets and "floor" in gate
        and sum(buckets[name]) / len(buckets[name]) < gate["floor"]
    ]
    if breached:
        print()
        for b in breached:
            print(f"  ✗ {b}")
        print("\n  These are targets, not the current state — see thresholds.json.")
        print("  The fix is a tool change: let the agent check whether a label exists.")
        if args.gate:
            sys.exit(1)


if __name__ == "__main__":
    main()
