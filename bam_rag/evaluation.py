from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bam_rag.retrieval import RetrievalIndex, SearchHit


@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    question: str
    expected_article_id: str
    split: str


def load_questions(path: Path) -> list[EvaluationQuestion]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"id", "question", "expected_article_id", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Colonnes manquantes dans evaluation.csv : " + ", ".join(sorted(missing))
            )
        questions = []
        for row_number, row in enumerate(reader, start=2):
            if not row.get("question", "").strip():
                continue
            if row.get("split", "").strip().lower() not in {"train", "test"}:
                raise ValueError(f"Split invalide à la ligne {row_number} : utiliser train ou test.")
            if not row.get("expected_article_id", "").strip():
                raise ValueError(f"Article attendu manquant à la ligne {row_number}.")
            questions.append(
                EvaluationQuestion(
                    row.get("id", str(row_number)).strip(),
                    row["question"].strip(),
                    row["expected_article_id"].strip(),
                    row["split"].strip().lower(),
                )
            )
    return questions


def ranking_metrics(
    ranked_ids: list[str], relevant_id: str, k_values: tuple[int, ...] = (1, 3, 5, 10)
) -> dict[str, float]:
    try:
        rank = ranked_ids.index(relevant_id) + 1
    except ValueError:
        rank = 0
    results = {"reciprocal_rank": 1.0 / rank if rank else 0.0}
    for k in k_values:
        results[f"hit_rate@{k}"] = float(rank != 0 and rank <= k)
        results[f"recall@{k}"] = float(rank != 0 and rank <= k)
        results[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if rank and rank <= k else 0.0
    return results


def evaluate(
    questions: list[EvaluationQuestion],
    index: RetrievalIndex,
    method: str,
    top_k: int = 10,
) -> list[dict[str, object]]:
    unknown_ids = {q.expected_article_id for q in questions} - set(index.article_by_id)
    if unknown_ids:
        examples = ", ".join(sorted(unknown_ids)[:3])
        raise ValueError(
            f"{len(unknown_ids)} identifiant(s) d’article attendu(s) introuvable(s) "
            f"dans le catalogue, par exemple : {examples}"
        )
    results: list[dict[str, object]] = []
    search: Callable[..., list[SearchHit]] = {
        "sémantique": index.semantic,
        "hiérarchique": index.hierarchical,
        "avancée": index.advanced,
    }[method.lower()]
    for question in questions:
        hits = search(question.question, top_k=top_k)
        metrics = ranking_metrics(
            [hit.article.article_id for hit in hits], question.expected_article_id
        )
        results.append(
            {
                "id": question.question_id,
                "question": question.question,
                "split": question.split,
                "expected_article_id": question.expected_article_id,
                "method": method,
                "rank": (
                    [hit.article.article_id for hit in hits].index(question.expected_article_id) + 1
                    if question.expected_article_id in [hit.article.article_id for hit in hits]
                    else None
                ),
                **metrics,
            }
        )
    return results


def summarize(results: list[dict[str, object]]) -> dict[str, float]:
    if not results:
        return {}
    metric_names = [
        key for key in results[0] if key.startswith(("hit_rate@", "recall@", "ndcg@"))
    ] + ["reciprocal_rank"]
    return {
        metric: sum(float(row[metric]) for row in results) / len(results)
        for metric in metric_names
    }
