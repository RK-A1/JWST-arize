#!/usr/bin/env python3
"""
Run the golden set as a Phoenix experiment, and gate the result.

The golden set is uploaded once as a **versioned Phoenix dataset**, and every
run is an **experiment** against it. That matters more than it sounds: the
platform then owns the dataset version, the per-example results, and the
comparison between runs, so "did this change help?" is a view rather than
something this script computes and prints.

An earlier version of this file scored a pandas DataFrame and diffed the means
against a JSON file it kept on disk. It worked, and it was the wrong shape: it
reimplemented experiment tracking badly, and the results lived nowhere anyone
else could see them.

    python scripts/run_golden.py                              # v2-concise, gated
    python scripts/run_golden.py --prompt-version v1-grounded
    python scripts/run_golden.py --limit 12 --repetitions 1   # PR smoke run

Exits non-zero when a limit in thresholds.json is breached.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from phoenix.client import Client  # noqa: E402

from src.jwst_arize.agent import DEFAULT_MODEL, run_agent  # noqa: E402
from src.jwst_arize.experiment_evaluators import GOLDEN  # noqa: E402
from src.jwst_arize.tracing import destination, flush, setup_tracing  # noqa: E402

GATES = json.loads((ROOT / "thresholds.json").read_text())["gates"]
DATASET_NAME = "jwst-golden-set"


def spread(cases: list[dict], n: int) -> list[dict]:
    """Take n cases across all case types, not the first n.

    golden.json is grouped by type, so slicing from the front returns only count
    questions — which skip the scorers that matter most. A smoke run that
    exercises neither is worse than no smoke run, because it reports green.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        buckets[c["metadata"]["case_type"]].append(c)
    picked, i = [], 0
    while len(picked) < n and any(len(b) > i for b in buckets.values()):
        for b in buckets.values():
            if i < len(b) and len(picked) < n:
                picked.append(b[i])
        i += 1
    return picked


def ensure_dataset(client: Client, cases: list[dict], name: str) -> Any:
    """Upload the golden set once; reuse it afterwards.

    Phoenix versions a dataset on write, so re-uploading identical content would
    produce a new version and silently break the comparison between this run and
    the last. Fetch first, create only if absent.
    """
    try:
        dataset = client.datasets.get_dataset(dataset=name)
        print(f"dataset '{name}' — {len(dataset.examples)} examples (existing)")
        return dataset
    except Exception:
        pass

    dataset = client.datasets.create_dataset(
        name=name,
        inputs=[{"question": c["input"]["question"]} for c in cases],
        outputs=[
            {
                "answer": c["expected"]["answer"],
                "photo_ids": c["expected"]["photo_ids"],
                "answerable": True,  # every golden case is answerable by construction
            }
            for c in cases
        ],
        metadata=[
            {
                "shape": "search_and_verify"
                if c["metadata"]["case_type"] == "disambiguate"
                else c["metadata"]["case_type"],
                "case_type": c["metadata"]["case_type"],
                "optimal_tool_calls": c["metadata"]["optimal_tool_calls"],
                "label": c["metadata"].get("label"),
            }
            for c in cases
        ],
        dataset_description=(
            "44 questions generated from the JWST photo corpus. Every one is "
            "answerable — which is the blind spot this repo is about."
        ),
    )
    print(f"dataset '{name}' — {len(dataset.examples)} examples (created)")
    return dataset


def scores_from(experiment: dict[str, Any]) -> dict[str, float]:
    """Mean score per evaluator, skipping rows the evaluator did not apply to."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for run in experiment.get("evaluation_runs", []) or []:
        # ExperimentEvaluationRun is an object with a dict `result`.
        result = getattr(run, "result", None) or {}
        name = getattr(run, "name", None) or result.get("name")
        score = result.get("score")
        if name and score is not None:
            buckets[name].append(float(score))
    return {name: sum(v) / len(v) for name, v in buckets.items() if v}


def previous_experiment(client: Client, dataset_id: str, current_id: str) -> dict[str, float]:
    """The most recent earlier experiment on this dataset, as scores.

    This is the comparison the gate's regression limits are measured against.
    Reading it back from the platform rather than from a file on disk means CI,
    a laptop, and a colleague's run all compare against the same history.
    """
    try:
        experiments = client.experiments.list(dataset_id=dataset_id)
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not list prior experiments: {exc})")
        return {}
    earlier = [e for e in experiments if e.get("id") != current_id]
    if not earlier:
        return {}
    prior = earlier[-1]
    try:
        return scores_from(client.experiments.get_experiment(experiment_id=prior["id"]))
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-version", default="v2-concise")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repetitions", type=int, default=2)
    ap.add_argument("--project", default=os.getenv("GOLDEN_PROJECT", "jwst-golden"))
    ap.add_argument("--endpoint", default=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"))
    ap.add_argument("--dataset", default=DATASET_NAME)
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    cases = json.loads((ROOT / "data" / "golden.json").read_text())["cases"]
    dataset_name = args.dataset
    if args.limit:
        cases = spread(cases, args.limit)
        # A subset is a different dataset, not a different version of the same
        # one — otherwise a 12-case PR run becomes the baseline for the next
        # 44-case run on main.
        dataset_name = f"{args.dataset}-{len(cases)}"

    setup_tracing(args.project)
    client = Client(base_url=args.endpoint)
    print(f"golden set → {destination()}  project={args.project}")

    dataset = ensure_dataset(client, cases, dataset_name)

    def task(input: dict[str, Any]) -> dict[str, Any]:
        result = run_agent(input["question"], args.prompt_version, args.model)
        return {"answer": result.answer, "trajectory": [t.__dict__ for t in result.trajectory]}

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=GOLDEN,
        experiment_name=args.prompt_version,
        experiment_description=(
            f"JWST research agent — prompt {args.prompt_version}, model {args.model}, "
            f"{len(cases)} cases x {args.repetitions} repetitions."
        ),
        experiment_metadata={
            "prompt_version": args.prompt_version,
            "model": args.model,
            "repetitions": args.repetitions,
        },
        repetitions=args.repetitions,
        print_summary=False,
    )
    flush()

    current = scores_from(experiment)
    baseline = previous_experiment(client, experiment["dataset_id"], experiment["experiment_id"])

    print("\n" + "═" * 58)
    print(f"  GOLDEN SET — {args.prompt_version}")
    print("═" * 58)

    failures: list[str] = []
    if not current:
        failures.append("no scores reported — nothing to check")

    for name in sorted(current):
        score = current[name]
        gate = GATES.get(name, {})
        delta = score - baseline[name] if name in baseline else None
        bad = []
        if "floor" in gate and score < gate["floor"]:
            bad.append(f"{name}: {score:.1%} is below its floor of {gate['floor']:.0%}")
        if "maxRegression" in gate and delta is not None and -delta > gate["maxRegression"]:
            bad.append(f"{name}: {delta * 100:+.1f} exceeds its {gate['maxRegression']:.0%} drop limit")
        failures.extend(bad)
        shown = f"  ({delta * 100:+.1f})" if delta is not None else ""
        print(f"  {'FAIL' if bad else 'ok  '}  {name:24s} {score:6.1%}{shown}")

    try:
        url = client.experiments.get_experiment_url(experiment["dataset_id"], experiment["experiment_id"])
        print(f"\n  {url}")
    except Exception:
        pass
    if not baseline:
        print("  (first experiment on this dataset — nothing to compare against yet)")

    print()
    if failures and not args.no_gate:
        for f in failures:
            print(f"  ✗ {f}")
        print("\nLimits are in thresholds.json. Changing one is a reviewable diff.")
        sys.exit(1)
    print(f"  All gates passed. ({len(current)} scorers checked against thresholds.json)")
    print("\n  Note what is NOT in this table: unsupported_citation. Every case in")
    print("  this dataset is answerable, so it has nothing to measure. See")
    print("  scripts/curate_failures.py for the dataset where it does.")


if __name__ == "__main__":
    main()
