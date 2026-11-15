from bam_rag.evaluation import ranking_metrics
from bam_rag.retrieval import detect_references, tokenize


def test_ranking_metrics_are_reciprocal_and_cut_off() -> None:
    metrics = ranking_metrics(["a", "target", "c"], "target")

    assert metrics["reciprocal_rank"] == 0.5
    assert metrics["hit_rate@1"] == 0.0
    assert metrics["hit_rate@3"] == 1.0
    assert metrics["ndcg@3"] > 0.0


def test_french_tokenization_and_reference_detection() -> None:
    assert "établissements" in tokenize("Établissements de crédit")
    article_numbers, instrument_ids = detect_references(
        "Que prévoit l’article 218-4-1 de la loi n° 103-12 ?"
    )

    assert article_numbers == {"218-4-1"}
    assert "103-12" in instrument_ids
