"""
Unit tests for the deterministic evaluators.

These gate first in CI, before anything that costs money. A scorer that is
wrong produces experiments that are wrong in a way that looks like a real
result — the most expensive kind of bug in an eval suite, because it is
invisible until someone acts on it.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.jwst_arize.evaluators import (  # noqa: E402
    citation_grounding,
    count_accuracy,
    required_citations,
    tool_selection,
    trajectory_efficiency,
    unsupported_citation,
)

REAL_ID = "51842916663"       # in the corpus
OTHER_REAL_ID = "52211883534"  # also in the corpus
FAKE_ID = "99999999999"        # not in the corpus


def score(evaluator, **kwargs):
    return evaluator.evaluate(kwargs)[0]


# ── citation_grounding ──────────────────────────────────────────────────────


def test_grounding_cited_id_that_was_retrieved_scores_one():
    s = score(citation_grounding, answer=f"See photo {REAL_ID}.",
              trajectory=[{"name": "get_photo", "returned_photo_ids": [REAL_ID]}])
    assert s.score == 1.0


def test_grounding_real_id_never_retrieved_is_not_grounded():
    s = score(citation_grounding, answer=f"See photo {REAL_ID}.",
              trajectory=[{"name": "search_photos", "returned_photo_ids": [OTHER_REAL_ID]}])
    assert s.score == 0.0
    assert s.metadata["ungrounded_ids"] == [REAL_ID]
    # It exists, so it is ungrounded rather than invented — a different bug.
    assert s.metadata["nonexistent_ids"] == []


def test_grounding_flags_a_fabricated_id_as_nonexistent():
    s = score(citation_grounding, answer=f"See photo {FAKE_ID}.",
              trajectory=[{"name": "search_photos", "returned_photo_ids": []}])
    assert s.metadata["nonexistent_ids"] == [FAKE_ID]


def test_grounding_no_citations_is_skipped_not_passed():
    s = score(citation_grounding, answer="I could not find it.", trajectory=[])
    assert s.score is None, "an answer citing nothing must not score 1.0"


def test_grounding_partial_credit():
    s = score(citation_grounding, answer=f"Photos {REAL_ID} and {FAKE_ID}.",
              trajectory=[{"name": "get_photo", "returned_photo_ids": [REAL_ID]}])
    assert s.score == 0.5


# ── unsupported_citation — the production-only scorer ───────────────────────


def test_unsupported_citation_skips_answerable_rows():
    s = score(unsupported_citation, answer=f"Photo {REAL_ID}.", expected_answerable=True)
    assert s.score is None


def test_unsupported_citation_fails_any_citation_on_an_unanswerable_question():
    s = score(unsupported_citation, answer=f"That is photo {REAL_ID}.", expected_answerable=False)
    assert s.score == 0.0
    assert s.label == "fabricated"
    assert s.metadata["ids_that_exist"] == [REAL_ID]


def test_unsupported_citation_passes_a_clean_refusal():
    s = score(unsupported_citation, answer="The corpus has no photo of Betelgeuse.",
              expected_answerable=False)
    assert s.score == 1.0
    assert s.label == "clean"


def test_the_two_citation_scorers_disagree_on_the_same_answer():
    """The finding this repo exists to demonstrate, as an assertion.

    An agent asked an unanswerable question, handed a near-miss by search, and
    citing it: grounded (a tool really did return that ID) and unsupported
    (there was nothing to cite in the first place).
    """
    answer = f"The supernova photo is {REAL_ID}."
    trajectory = [{"name": "search_photos", "returned_photo_ids": [REAL_ID]}]
    grounded = score(citation_grounding, answer=answer, trajectory=trajectory)
    unsupported = score(unsupported_citation, answer=answer, expected_answerable=False)
    assert grounded.score == 1.0
    assert unsupported.score == 0.0


# ── the rest ────────────────────────────────────────────────────────────────


def test_required_citations_catches_the_wrong_photo():
    s = score(required_citations, answer=f"Photo {OTHER_REAL_ID}.", expected_photo_ids=[REAL_ID])
    assert s.score == 0.0
    assert s.metadata["missing"] == [REAL_ID]


def test_count_accuracy_finds_the_integer_in_prose():
    s = score(count_accuracy, answer="There are 42 nebula photos.", expected_answer="42", shape="count")
    assert s.score == 1.0


def test_count_accuracy_does_not_mistake_a_photo_id_for_the_count():
    s = score(count_accuracy, answer=f"Photo {REAL_ID} is one of them.",
              expected_answer="42", shape="count")
    assert s.score == 0.0


def test_tool_selection_catches_the_skipped_verification_step():
    s = score(tool_selection, trajectory=[{"name": "search_photos"}], shape="search_and_verify")
    assert s.score == 0.5
    assert s.metadata["missing"] == ["get_photo"]


def test_efficiency_penalises_redundant_calls():
    s = score(trajectory_efficiency,
              trajectory=[{"name": "get_photo"}, {"name": "get_photo"}], optimal_tool_calls=1)
    assert s.score == 0.5


def test_efficiency_using_no_tools_scores_zero_not_infinity():
    s = score(trajectory_efficiency, trajectory=[], optimal_tool_calls=1)
    assert s.score == 0.0


def test_efficiency_never_exceeds_one():
    s = score(trajectory_efficiency, trajectory=[{"name": "get_photo"}], optimal_tool_calls=3)
    assert s.score == 1.0


# ── The adapter layer ───────────────────────────────────────────────────────
#
# evaluators.py holds the logic; experiment_evaluators.py binds it to Phoenix's
# experiment runner, which calls evaluators with `output`/`expected`/`metadata`
# rather than named fields. These tests assert the two paths agree — if they
# could drift, the claim that one suite covers both offline experiments and
# production traffic would be false.

import inspect  # noqa: E402

from src.jwst_arize import experiment_evaluators as xe  # noqa: E402


def test_adapter_agrees_with_the_scorer_it_wraps():
    answer = f"See photo {REAL_ID}."
    trajectory = [{"name": "get_photo", "returned_photo_ids": [REAL_ID]}]

    direct = score(citation_grounding, answer=answer, trajectory=trajectory)
    adapted = xe.citation_grounding(output={"answer": answer, "trajectory": trajectory})

    assert adapted == (direct.score, direct.label, direct.explanation)


def test_strict_adapter_treats_every_row_as_unanswerable():
    """The curated-failures dataset has no `answerable` flag — every example in
    it is unanswerable by construction, and the adapter must encode that."""
    out = {"answer": f"That is photo {REAL_ID}."}
    assert xe.unsupported_citation_strict(output=out)[0] == 0.0
    assert xe.unsupported_citation_strict(output={"answer": "Not in the corpus."})[0] == 1.0


def test_every_adapter_binds_to_phoenix_named_parameters():
    """Phoenix binds evaluator arguments by inspecting the signature.

    A function whose only parameter is `**kwargs` looks like a single positional
    argument to that binder and blows up at runtime, after the task runs and the
    money is spent. Catch it here instead.
    """
    allowed = {"input", "output", "expected", "reference", "metadata", "example", "trace_id"}
    for suite in (xe.GOLDEN, xe.FAILURES):
        for name, fn in suite.items():
            params = inspect.signature(fn).parameters
            named = [
                p for p in params.values()
                if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            ]
            assert named, f"{name} has no named parameters — Phoenix will bind it positionally"
            for p in named:
                assert p.name in allowed, f"{name} takes '{p.name}', which Phoenix cannot supply"


def test_both_suites_cover_the_scorers_their_dataset_can_measure():
    assert "unsupported_citation" not in xe.GOLDEN, (
        "every golden case is answerable, so this scorer would skip every row"
    )
    assert "unsupported_citation" in xe.FAILURES
    assert "citation_grounding" in xe.FAILURES, (
        "the failures suite must keep the scorer that reads 100% — the "
        "disagreement between the two is the finding"
    )
