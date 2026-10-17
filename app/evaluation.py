"""Separate retrieval and quoted-answer contract evaluation on a tiny authored corpus.

The local BM25/extractive adapter is not Azure Search or an LLM. These development
fixtures exercise the scorer and regression gate, not financial research quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ABSTENTION = "The provided documents do not contain information about this."
STOPWORDS = {"what", "was", "is", "the", "for", "how", "much", "were", "did", "a", "of"}


def retrieval_scores(
    ranked_ids: list[str], relevance: dict[str, int], k: int = 5
) -> dict[str, float] | None:
    if k < 1 or len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Positive k and unique ranked document IDs required")
    if any(type(grade) is not int or grade < 1 for grade in relevance.values()):
        raise ValueError("Relevance grades must be positive integers")
    if not relevance:
        return None  # Unanswerable cases do not inflate retrieval averages.
    found = ranked_ids[:k]
    recall = len(set(found) & relevance.keys()) / len(relevance)
    reciprocal_rank = next((1 / rank for rank, doc in enumerate(found, 1) if doc in relevance), 0.0)
    dcg = sum((2 ** relevance.get(doc, 0) - 1) / math.log2(i + 2) for i, doc in enumerate(found))
    ideal = sum(
        (2**grade - 1) / math.log2(i + 2)
        for i, grade in enumerate(sorted(relevance.values(), reverse=True)[:k])
    )
    return {
        f"recall_at_{k}": recall,
        f"mrr_at_{k}": reciprocal_rank,
        f"ndcg_at_{k}": dcg / ideal,
    }


def answer_scores(answer: str, retrieved: dict[str, str], answerable: bool) -> dict[str, Any]:
    """Exact-quote support, NOT semantic entailment or financial correctness."""
    abstained = answer.strip() == ABSTENTION
    quote_matches = list(re.finditer(r'"([^"\n]+)" \[([A-Za-z0-9_-]+)\]', answer))
    residue = re.sub(r'"([^"\n]+)" \[([A-Za-z0-9_-]+)\]', "", answer).strip()
    supported = sum(
        match.group(2) in retrieved and match.group(1) in retrieved[match.group(2)]
        for match in quote_matches
    )
    support = supported / len(quote_matches) if quote_matches else None
    contract_passed = bool(quote_matches) and not residue and support == 1.0
    return {
        "abstained": abstained,
        "abstention_correct": abstained == (not answerable),
        "quoted_claims": len(quote_matches),
        "supported_quoted_claims": supported,
        "citation_support": support,
        "answer_contract_passed": abstained if not answerable else contract_passed,
        "unscored_free_text": bool(residue) and not abstained,
    }


def tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def local_retrieve(question: str, documents: dict[str, str]) -> list[str]:
    """An actual local BM25 reference adapter, separate from Azure SDK retrieval."""
    terms = {doc: Counter(tokens(text)) for doc, text in documents.items()}
    average_length = fmean(sum(freq.values()) for freq in terms.values())
    query = set(tokens(question))
    scores = {}
    for doc, frequency in terms.items():
        score = 0.0
        for word in query:
            containing = sum(word in other for other in terms.values())
            idf = math.log(1 + (len(terms) - containing + 0.5) / (containing + 0.5))
            count = frequency[word]
            score += (
                idf
                * count
                * 2.2
                / (count + 1.2 * (0.25 + 0.75 * sum(frequency.values()) / average_length))
            )
        if score > 0:
            scores[doc] = score
    return sorted(scores, key=lambda doc: (-scores[doc], doc))[:5]


def local_answer(question: str, retrieved: dict[str, str]) -> str:
    """Quote a complete matching passage; abstain unless all query terms occur."""
    query = set(tokens(question))
    for doc, content in retrieved.items():
        if query and query <= set(tokens(content)):
            return f'"{content}" [{doc}]'
    return ABSTENTION


def load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("scope") != "self-authored-development-fixture":
        raise ValueError("An explicitly labelled authored fixture is required")
    documents = payload.get("documents")
    cases = payload.get("cases")
    if not isinstance(documents, dict) or not documents or not isinstance(cases, list) or not cases:
        raise ValueError("Nonempty corpus and cases required")
    if any(not isinstance(text, str) or not text for text in documents.values()):
        raise ValueError("Nonempty document text required")
    ids = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("question"), str):
            raise ValueError("Invalid question")
        relevance = case.get("relevance")
        if not isinstance(relevance, dict) or not set(relevance) <= documents.keys():
            raise ValueError("Relevance must refer to known corpus document IDs")
        retrieval_scores([], relevance)
        ids.append(case["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs")
    return payload


def gate(rows: list[dict[str, Any]], policy: dict[str, float]) -> dict[str, Any]:
    if not rows:
        return {"passed": False, "failures": ["empty_report"]}
    retrieval = [row["retrieval"] for row in rows if row["retrieval"] is not None]
    failures = []
    for metric in ("recall_at_5", "mrr_at_5", "ndcg_at_5"):
        if not retrieval or fmean(row[metric] for row in retrieval) < policy[metric]:
            failures.append(metric)
    for metric in ("answer_contract_passed", "abstention_correct"):
        if fmean(row["answer"][metric] for row in rows) < policy[metric]:
            failures.append(metric)
    if not any(not row["expected_answerable"] for row in rows):
        failures.append("missing_abstention_cases")
    return {"passed": not failures, "failures": failures}


def evaluate(fixture: Path, policy_path: Path, fault: str = "none") -> dict[str, Any]:
    data = load_fixture(fixture)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if set(policy) != {
        "recall_at_5",
        "mrr_at_5",
        "ndcg_at_5",
        "answer_contract_passed",
        "abstention_correct",
    } or any(type(value) not in {int, float} or not 0 <= value <= 1 for value in policy.values()):
        raise ValueError("Invalid gate policy")
    if fault not in {"none", "retrieval", "citation", "abstention"}:
        raise ValueError("Unknown fault")
    rows = []
    for case in data["cases"]:
        ranked = local_retrieve(case["question"], data["documents"])
        if fault == "retrieval":
            ranked = [doc for doc in ranked if doc not in case["relevance"]]
        retrieved = {doc: data["documents"][doc] for doc in ranked}
        answer = local_answer(case["question"], retrieved)
        if fault == "citation" and case["relevance"]:
            answer = '"Revenue was fabricated at 999 million." [missing-document]'
        if fault == "abstention" and not case["relevance"]:
            answer = '"The dividend was 999 pounds." [missing-document]'
        rows.append(
            {
                "case_id": case["id"],
                "question": case["question"],
                "expected_answerable": bool(case["relevance"]),
                "ranked_ids": ranked,
                "generated_answer": answer,
                "retrieval": retrieval_scores(ranked, case["relevance"]),
                "answer": answer_scores(answer, retrieved, bool(case["relevance"])),
            }
        )
    retrieval = [row["retrieval"] for row in rows if row["retrieval"] is not None]
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "execution_mode": "offline_local_bm25_extractive",
        "candidate_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scope": "self-authored development checks, not a benchmark or Azure/LLM measurement",
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "fault_injection": fault,
        "sample_size": len(rows),
        "retrieval_sample_size": len(retrieval),
        "unanswerable_sample_size": len(rows) - len(retrieval),
        "summary": {
            "retrieval": {
                metric: fmean(row[metric] for row in retrieval)
                for metric in ("recall_at_5", "mrr_at_5", "ndcg_at_5")
            },
            "answer_contract_pass_rate": fmean(
                row["answer"]["answer_contract_passed"] for row in rows
            ),
            "abstention_accuracy": fmean(row["answer"]["abstention_correct"] for row in rows),
        },
        "policy": policy,
        "gate": gate(rows, policy),
        "rows": rows,
        "external_provider_calls": 0,
        "limitations": [
            "Local BM25/extractive adapter; not QueryPipeline or Azure Search.",
            "Quote matching only; no semantic or financial correctness claim.",
            "Tiny authored fixture; thresholds are chosen contracts, not calibrated targets.",
            "No live EDGAR answer quality or held-out benchmark; B01 remains deferred.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "eval/development.json")
    parser.add_argument("--policy", type=Path, default=ROOT / "eval/policy.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/S5-report.json")
    parser.add_argument(
        "--fault", choices=["none", "retrieval", "citation", "abstention"], default="none"
    )
    args = parser.parse_args()
    report = evaluate(args.fixture, args.policy, args.fault)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"mode": report["execution_mode"], "n": report["sample_size"], "gate": report["gate"]}
        )
    )
    raise SystemExit(0 if report["gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
