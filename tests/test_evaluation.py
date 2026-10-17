"""Hand-calculated ranking examples and independent answer/retrieval regressions."""

import math

import pytest

from app.evaluation import (
    ABSTENTION,
    ROOT,
    answer_scores,
    evaluate,
    gate,
    retrieval_scores,
)


def test_hand_calculated_recall_mrr_and_graded_ndcg() -> None:
    scores = retrieval_scores(["noise", "second", "first"], {"first": 2, "second": 1})
    assert scores is not None
    assert scores["recall_at_5"] == 1.0
    assert scores["mrr_at_5"] == 0.5
    expected = (1 / math.log2(3) + 3 / math.log2(4)) / (3 + 1 / math.log2(3))
    assert scores["ndcg_at_5"] == pytest.approx(expected)
    assert retrieval_scores([], {"first": 1}) == {
        "recall_at_5": 0.0,
        "mrr_at_5": 0.0,
        "ndcg_at_5": 0.0,
    }
    assert retrieval_scores(["noise"], {}) is None


def test_truncation_and_duplicate_rankings() -> None:
    scores = retrieval_scores(["a", "b", "c", "d", "e", "relevant"], {"relevant": 1})
    assert scores is not None and scores["recall_at_5"] == 0.0
    with pytest.raises(ValueError, match="unique"):
        retrieval_scores(["relevant", "relevant"], {"relevant": 1})
    with pytest.raises(ValueError, match="positive integers"):
        retrieval_scores([], {"relevant": -1})


def test_citation_text_and_id_must_both_match_retrieved_context() -> None:
    retrieved = {"source": "Revenue was 42 pounds."}
    assert answer_scores('"Revenue was 42 pounds." [source]', retrieved, True)[
        "answer_contract_passed"
    ]
    for answer in [
        '"Revenue was 999 pounds." [source]',
        '"Revenue was 42 pounds." [invented]',
        "Revenue was 42 pounds.",
        '"Revenue was 42 pounds." [source] And profits doubled.',
    ]:
        assert not answer_scores(answer, retrieved, True)["answer_contract_passed"]


def test_abstention_does_not_inflate_citation_support() -> None:
    scores = answer_scores(ABSTENTION, {}, False)
    assert scores["answer_contract_passed"] and scores["abstention_correct"]
    assert scores["citation_support"] is None
    assert not answer_scores(ABSTENTION, {}, True)["answer_contract_passed"]
    assert not answer_scores('"Unknown fact." [x]', {}, False)["abstention_correct"]


@pytest.mark.parametrize("fault", ["retrieval", "citation", "abstention"])
def test_observed_regressions_block_gate_separately(fault: str) -> None:
    baseline = evaluate(ROOT / "eval/development.json", ROOT / "eval/policy.json")
    failed = evaluate(ROOT / "eval/development.json", ROOT / "eval/policy.json", fault)
    assert baseline["gate"]["passed"]
    assert not failed["gate"]["passed"]
    assert failed["sample_size"] == 10 and failed["retrieval_sample_size"] == 8
    assert failed["execution_mode"] == "offline_local_bm25_extractive"
    assert failed["external_provider_calls"] == 0
    if fault in {"citation", "abstention"}:
        assert failed["summary"]["retrieval"] == baseline["summary"]["retrieval"]
    if fault == "retrieval":
        assert failed["summary"]["retrieval"]["recall_at_5"] == 0.0


def test_empty_report_cannot_pass() -> None:
    assert gate([], {}) == {"passed": False, "failures": ["empty_report"]}
