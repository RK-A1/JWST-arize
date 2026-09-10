"""
Evaluators, in two sets.

`GOLDEN_EVALUATORS` is the suite carried over from the Braintrust project — the
one that gates CI and that passes. `PRODUCTION_EVALUATORS` is what the same
agent looks like when it is measured against traffic the golden set could not
contain.

The interesting design point is that these two sets *overlap*. `citation_grounding`
appears in both and stays high in both, because the agent really does cite IDs
that a tool really did return. It is a correct scorer reporting a true fact. The
scorer that collapses is `unsupported_citation`, which asks a question the golden
set could never ask: was there anything in the corpus to cite at all?

An eval suite is only as good as the hardest question in its dataset. This
module is that argument in code.
"""

from __future__ import annotations

import os
import re
from typing import Any

from phoenix.evals import LLM, Score, create_classifier, create_evaluator

from .corpus import photo_exists

# Flickr photo IDs are 10-12 digit numbers; nothing else in these answers is.
PHOTO_ID_RE = re.compile(r"\b\d{10,12}\b")


def cited_ids(answer: str) -> list[str]:
    return list(dict.fromkeys(PHOTO_ID_RE.findall(answer or "")))


def _retrieved_ids(trajectory: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for call in trajectory or []:
        out.update(call.get("returned_photo_ids") or [])
    return out


# ── 1. Citation grounding — carried over, and the one that does not fire ────


@create_evaluator(name="citation_grounding", kind="code")
def citation_grounding(answer: str, trajectory: list[dict[str, Any]]) -> Score:
    """Every photo ID in the answer must have come back from a tool call.

    Note what this does *not* check: whether the ID has anything to do with the
    question. On the production set the agent is handed near-miss search hits
    for subjects that are not in the corpus, cites one, and scores 1.0 here —
    correctly, because a tool did return it. That is the whole point. This
    scorer was never wrong; it was answering a narrower question than anyone
    reading the dashboard assumed.
    """
    cited = cited_ids(answer)
    if not cited:
        return Score(name="citation_grounding", score=None, label="skipped",
                     explanation="answer cited no photo IDs")

    retrieved = _retrieved_ids(trajectory)
    grounded = [i for i in cited if i in retrieved]
    return Score(
        name="citation_grounding",
        score=len(grounded) / len(cited),
        label="grounded" if len(grounded) == len(cited) else "partial",
        metadata={
            "cited": len(cited),
            "grounded": len(grounded),
            "ungrounded_ids": [i for i in cited if i not in retrieved],
            "nonexistent_ids": [i for i in cited if not photo_exists(i)],
        },
    )


# ── 2. Unsupported citation — the production-only scorer ────────────────────


@create_evaluator(name="unsupported_citation", kind="code")
def unsupported_citation(answer: str, expected_answerable: bool) -> Score:
    """Did the agent cite a photo ID for a question the corpus cannot answer?

    Scoped to the unanswerable categories, where the correct answer provably
    contains no photo ID: the subject is verifiably absent from the corpus, or
    the question is about the observatory rather than about a photo, or the
    premise is contradicted by the data. On those rows a citation is not a
    matter of taste. There is no right ID, so any ID is wrong.

    Scored 1.0 for a clean answer and 0.0 for any citation, so it reads the same
    way round as every other scorer: higher is better.
    """
    if expected_answerable:
        return Score(name="unsupported_citation", score=None, label="skipped",
                     explanation="question is answerable from the corpus")

    cited = cited_ids(answer)
    return Score(
        name="unsupported_citation",
        score=0.0 if cited else 1.0,
        label="fabricated" if cited else "clean",
        metadata={
            "cited_ids": cited,
            # An ID that exists is a real photo pressed into answering a
            # question it has nothing to do with; one that does not exist is
            # outright invention. Both are wrong, and they are different bugs.
            "ids_that_exist": [i for i in cited if photo_exists(i)],
            "ids_invented": [i for i in cited if not photo_exists(i)],
        },
    )


# ── 3. Required citations — did it name the right photo? ────────────────────


@create_evaluator(name="required_citations", kind="code")
def required_citations(answer: str, expected_photo_ids: list[str]) -> Score:
    if not expected_photo_ids:
        return Score(name="required_citations", score=None, label="skipped",
                     explanation="no specific photo required")
    cited = set(cited_ids(answer))
    hit = [i for i in expected_photo_ids if i in cited]
    return Score(
        name="required_citations",
        score=len(hit) / len(expected_photo_ids),
        metadata={"required": expected_photo_ids, "missing": [i for i in expected_photo_ids if i not in cited]},
    )


# ── 4. Count accuracy ───────────────────────────────────────────────────────


@create_evaluator(name="count_accuracy", kind="code")
def count_accuracy(answer: str, expected_answer: str | None, shape: str) -> Score:
    if shape != "count" or not expected_answer:
        return Score(name="count_accuracy", score=None, label="skipped",
                     explanation="not a count question")
    try:
        target = int(str(expected_answer).strip())
    except ValueError:
        return Score(name="count_accuracy", score=None, label="skipped",
                     explanation="expected answer is not an integer")
    # 1-6 digits only, so a cited photo ID cannot be misread as the count.
    numbers = [int(n) for n in re.findall(r"\b\d{1,6}\b", answer or "")]
    return Score(
        name="count_accuracy",
        score=1.0 if target in numbers else 0.0,
        metadata={"expected": target, "numbers_in_answer": numbers},
    )


# ── 5. Trajectory efficiency ────────────────────────────────────────────────


@create_evaluator(name="trajectory_efficiency", kind="code")
def trajectory_efficiency(trajectory: list[dict[str, Any]], optimal_tool_calls: int) -> Score:
    """How close the agent came to the optimal number of tool calls.

    Rewards *fewer* calls, so on its own it would endorse an agent that stops
    looking things up. That is exactly why it is never read alone.
    """
    actual = len(trajectory or [])
    if actual == 0:
        return Score(name="trajectory_efficiency", score=0.0, label="no_tools",
                     explanation="agent used no tools",
                     metadata={"optimal": optimal_tool_calls, "actual": 0})
    if optimal_tool_calls <= 0:
        return Score(name="trajectory_efficiency", score=None, label="skipped",
                     explanation="no optimal call count for this shape")
    return Score(
        name="trajectory_efficiency",
        score=min(1.0, optimal_tool_calls / actual),
        metadata={"optimal": optimal_tool_calls, "actual": actual,
                  "excess_calls": max(0, actual - optimal_tool_calls)},
    )


# ── 6. Tool selection ───────────────────────────────────────────────────────

REQUIRED_TOOLS = {
    "lookup": ["get_photo"],
    "count": ["count_by_label"],
    "search_and_verify": ["search_photos", "get_photo"],
    "comparative": ["get_photo"],
}


@create_evaluator(name="tool_selection", kind="code")
def tool_selection(trajectory: list[dict[str, Any]], shape: str) -> Score:
    required = REQUIRED_TOOLS.get(shape, [])
    if not required:
        return Score(name="tool_selection", score=None, label="skipped",
                     explanation="no required tools for this shape")
    used = {c.get("name") for c in trajectory or []}
    hit = [t for t in required if t in used]
    return Score(
        name="tool_selection",
        score=len(hit) / len(required),
        metadata={"required": required, "used": sorted(x for x in used if x),
                  "missing": [t for t in required if t not in used]},
    )


# ── LLM judges ──────────────────────────────────────────────────────────────


def judge_llm() -> LLM:
    """The judge model, reached through whatever OpenAI-compatible endpoint the
    agent uses. Same gateway, same auth, different model name if you want one."""
    kwargs: dict[str, Any] = {}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    return LLM(
        provider="openai",
        model=os.getenv("JUDGE_MODEL", "claude-haiku-4-5"),
        client="openai",
        sync_client_kwargs=kwargs,
        async_client_kwargs=kwargs,
    )


ANSWER_QUALITY_TEMPLATE = """You are grading a research assistant that answers questions about a
corpus of James Webb Space Telescope photographs.

Question: {question}
Reference answer: {expected_answer}
Assistant's answer: {answer}

Does the assistant's answer agree with the reference answer on the facts that matter?
Ignore differences in wording, length, and formatting. Extra correct detail is fine.

Respond with exactly one word:
correct   - the answer agrees with the reference on the substantive facts
partial   - the answer is partly right but omits or misstates something material
incorrect - the answer contradicts the reference, or answers a different question"""


REFUSAL_TEMPLATE = """A research assistant can only answer from a fixed corpus of 979 James Webb
Space Telescope photographs. It has been asked a question that corpus cannot answer.

Question: {question}
Why it cannot be answered: {expected_answer}
Assistant's answer: {answer}

Did the assistant make clear that it could not answer from the corpus?

Respond with exactly one word:
refused  - it stated it could not find or confirm this in the corpus
hedged   - it gestured at uncertainty but still supplied a substantive answer
complied - it answered as though the corpus supported the answer"""


def answer_quality_evaluator():
    return create_classifier(
        name="answer_quality",
        prompt_template=ANSWER_QUALITY_TEMPLATE,
        llm=judge_llm(),
        choices={"correct": 1.0, "partial": 0.5, "incorrect": 0.0},
    )


def appropriate_refusal_evaluator():
    """Only meaningful on unanswerable rows — the runner filters to those."""
    return create_classifier(
        name="appropriate_refusal",
        prompt_template=REFUSAL_TEMPLATE,
        llm=judge_llm(),
        choices={"refused": 1.0, "hedged": 0.5, "complied": 0.0},
    )


# Deterministic sets. The judges are constructed lazily by the runners, because
# building an LLM client at import time would break the unit tests.
GOLDEN_CODE_EVALUATORS = [
    citation_grounding,
    required_citations,
    count_accuracy,
    tool_selection,
    trajectory_efficiency,
]

PRODUCTION_CODE_EVALUATORS = [
    citation_grounding,
    unsupported_citation,
    required_citations,
    count_accuracy,
    trajectory_efficiency,
]
