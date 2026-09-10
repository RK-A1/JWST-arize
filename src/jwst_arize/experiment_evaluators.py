"""
Binding the scorers to Phoenix's experiment runner.

There is one implementation of each check, in evaluators.py. It gets used in two
places that call it very differently:

  * `scripts/score_traffic.py` scores production traffic. It rebuilds a turn
    from its span tree and passes named fields.
  * Phoenix's `run_experiment` scores a dataset. It calls an evaluator with
    `output`, `expected`, `metadata` and friends, drawn from the dataset example
    and the task's return value.

Rather than write the logic twice, this module is a thin adapter: unpack the
experiment runner's calling convention, delegate to the real scorer, and return
a `(score, label, explanation)` tuple, which is one of the shapes the runner
accepts.

Keeping the adapters this thin is deliberate. If a scorer's behaviour can drift
between the offline experiment and online production scoring, the whole claim
that one suite covers both is false.
"""

from __future__ import annotations

from typing import Any

from . import evaluators as ev

Tuple3 = tuple[float | None, str | None, str | None]


def _unpack(score: Any) -> Tuple3:
    """phoenix.evals returns a Score object; the experiment runner wants a tuple."""
    return (score.score, score.label, score.explanation)


def _run(evaluator: Any, **kwargs: Any) -> Tuple3:
    return _unpack(evaluator.evaluate(kwargs)[0])


def citation_grounding(output: dict[str, Any], **_: Any) -> Tuple3:
    return _run(
        ev.citation_grounding,
        answer=output.get("answer", ""),
        trajectory=output.get("trajectory", []),
    )


def unsupported_citation(output: dict[str, Any], expected: dict[str, Any], **_: Any) -> Tuple3:
    return _run(
        ev.unsupported_citation,
        answer=output.get("answer", ""),
        expected_answerable=bool(expected.get("answerable", True)),
    )


def required_citations(output: dict[str, Any], expected: dict[str, Any], **_: Any) -> Tuple3:
    return _run(
        ev.required_citations,
        answer=output.get("answer", ""),
        expected_photo_ids=expected.get("photo_ids") or [],
    )


def count_accuracy(
    output: dict[str, Any], expected: dict[str, Any], metadata: dict[str, Any], **_: Any
) -> Tuple3:
    return _run(
        ev.count_accuracy,
        answer=output.get("answer", ""),
        expected_answer=expected.get("answer"),
        shape=metadata.get("shape", ""),
    )


def tool_selection(output: dict[str, Any], metadata: dict[str, Any], **_: Any) -> Tuple3:
    return _run(
        ev.tool_selection,
        trajectory=output.get("trajectory", []),
        shape=metadata.get("shape", ""),
    )


def trajectory_efficiency(output: dict[str, Any], metadata: dict[str, Any], **_: Any) -> Tuple3:
    return _run(
        ev.trajectory_efficiency,
        trajectory=output.get("trajectory", []),
        optimal_tool_calls=int(metadata.get("optimal_tool_calls", 1)),
    )


def answer_quality(
    input: dict[str, Any], output: dict[str, Any], expected: dict[str, Any], **_: Any
) -> Tuple3:
    """The LLM judge. Built lazily so importing this module needs no API key."""
    judge = ev.answer_quality_evaluator()
    return _unpack(
        judge.evaluate(
            {
                "question": input.get("question", ""),
                "answer": output.get("answer", ""),
                "expected_answer": expected.get("answer") or "",
            }
        )[0]
    )


def appropriate_refusal(
    input: dict[str, Any], output: dict[str, Any], expected: dict[str, Any], **_: Any
) -> Tuple3:
    judge = ev.appropriate_refusal_evaluator()
    return _unpack(
        judge.evaluate(
            {
                "question": input.get("question", ""),
                "answer": output.get("answer", ""),
                "expected_answer": expected.get("answer") or "",
            }
        )[0]
    )


# The suite for a dataset of answerable questions (the golden set).
GOLDEN = {
    "citation_grounding": citation_grounding,
    "required_citations": required_citations,
    "tool_selection": tool_selection,
    "count_accuracy": count_accuracy,
    "trajectory_efficiency": trajectory_efficiency,
    "answer_quality": answer_quality,
}

# ── The curated-failures dataset ────────────────────────────────────────────
#
# Every example there is unanswerable by construction — it was curated from
# production turns where the agent cited a photo for a question with no answer.
# These adapters encode that rather than reading an `answerable` flag, and they
# read the reference from `why_unanswerable`, which is the dataset's output key.
#
# They are written as explicit functions rather than as a wrapper around the
# ones above, because Phoenix binds evaluator arguments by inspecting the
# signature: a `**kwargs`-only wrapper looks like a single positional parameter
# and the bind fails at runtime.


def unsupported_citation_strict(output: dict[str, Any], **_: Any) -> Tuple3:
    return _run(
        ev.unsupported_citation,
        answer=output.get("answer", ""),
        expected_answerable=False,
    )


def appropriate_refusal_strict(
    input: dict[str, Any], output: dict[str, Any], expected: dict[str, Any], **_: Any
) -> Tuple3:
    judge = ev.appropriate_refusal_evaluator()
    return _unpack(
        judge.evaluate(
            {
                "question": input.get("question", ""),
                "answer": output.get("answer", ""),
                "expected_answer": expected.get("why_unanswerable")
                or expected.get("answer")
                or "",
            }
        )[0]
    )


def trajectory_efficiency_strict(output: dict[str, Any], **_: Any) -> Tuple3:
    # One search is enough to establish that there is nothing there.
    return _run(
        ev.trajectory_efficiency,
        trajectory=output.get("trajectory", []),
        optimal_tool_calls=1,
    )


FAILURES = {
    "citation_grounding": citation_grounding,
    "unsupported_citation": unsupported_citation_strict,
    "appropriate_refusal": appropriate_refusal_strict,
    "trajectory_efficiency": trajectory_efficiency_strict,
}
