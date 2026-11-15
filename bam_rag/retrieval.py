from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from bam_rag.chunking import TextChunk, hierarchical_chunks, semantic_chunks
from bam_rag.corpus import Article
from bam_rag.config import settings

LOGGER = logging.getLogger(__name__)
INDEX_VERSION = 1


@dataclass(frozen=True)
class SearchHit:
    article: Article
    score: float
    method: str
    matched_passage: str = ""


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())


def detect_references(query: str) -> tuple[set[str], set[str]]:
    article_numbers = {
        re.sub(r"\s+", " ", match.group(1).lower()).replace("premier", "1er")
        for match in re.finditer(
            r"\barticle\s+(premier|1er|\d{1,3}(?:[-.]\d{1,3})*"
            r"(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|undecies))?)\b",
            query,
            re.IGNORECASE,
        )
    }
    instrument_ids = {
        re.sub(r"[\s–—]", "", match.group(1).lower()).replace("n°", "")
        for match in re.finditer(
            r"\b(?:loi|circulaire|directive|dahir|décret|decret)\s+n[°o]?\s*"
            r"([\w/-]+(?:\s*/\s*[\w-]+)?)",
            query,
            re.IGNORECASE,
        )
    }
    return article_numbers, instrument_ids


class RetrievalIndex:
    def __init__(
        self,
        articles: list[Article],
        model: SentenceTransformer,
        embedding_cache: Path,
    ) -> None:
        if not articles:
            raise ValueError("Le corpus ne contient aucun article indexable.")
        self.articles = articles
        self.article_by_id = {article.article_id: article for article in articles}
        self.model = model
        self.semantic_units = semantic_chunks(articles)
        self.passage_units = hierarchical_chunks(articles)
        embeddings = self._load_embeddings(embedding_cache)
        if embeddings is None:
            embeddings = {
                "article": self._encode([self._article_text(article) for article in articles]),
                "semantic": self._encode([chunk.text for chunk in self.semantic_units]),
                "passage": self._encode([chunk.text for chunk in self.passage_units]),
            }
            self._save_embeddings(embedding_cache, embeddings)
        self._article_embeddings = embeddings["article"]
        self._semantic_embeddings = embeddings["semantic"]
        self._passage_embeddings = embeddings["passage"]
        self._bm25 = BM25Okapi([tokenize(self._article_text(article)) for article in articles])
        self._reranker: CrossEncoder | None = None

    def _load_embeddings(self, path: Path) -> dict[str, np.ndarray] | None:
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as cache:
                embeddings = {
                    "article": np.asarray(cache["article"], dtype=np.float32),
                    "semantic": np.asarray(cache["semantic"], dtype=np.float32),
                    "passage": np.asarray(cache["passage"], dtype=np.float32),
                }
        except (OSError, ValueError, KeyError) as error:
            LOGGER.warning("Cache d’embeddings illisible (%s), il sera reconstruit.", error)
            return None

        expected = {
            "article": len(self.articles),
            "semantic": len(self.semantic_units),
            "passage": len(self.passage_units),
        }
        if any(
            embeddings[name].ndim != 2 or embeddings[name].shape[0] != count
            for name, count in expected.items()
        ):
            LOGGER.warning("Tailles du cache d’embeddings incohérentes, il sera reconstruit.")
            return None
        return embeddings

    @staticmethod
    def _save_embeddings(path: Path, embeddings: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(temporary_path, **embeddings)
        temporary_path.replace(path)

    @staticmethod
    def _article_text(article: Article) -> str:
        context = f"{article.section_context}\n" if article.section_context else ""
        return (
            f"{article.document_title} {context}"
            f"Article {article.article_number}\n{article.text}"
        )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )

    def _dense_ranks(
        self, query: str, units: list[TextChunk], embeddings: np.ndarray, limit: int
    ) -> list[tuple[str, float, str]]:
        query_embedding = self._encode([query])[0]
        scores = embeddings @ query_embedding
        best: dict[str, tuple[float, str]] = {}
        for index in np.argsort(scores)[::-1]:
            unit = units[int(index)]
            score = float(scores[index])
            previous = best.get(unit.article_id)
            if previous is None or score > previous[0]:
                best[unit.article_id] = (score, unit.text)
        ranked = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
        return [(article_id, value[0], value[1]) for article_id, value in ranked[:limit]]

    def _article_dense_ranks(self, query: str, limit: int) -> list[tuple[str, float, str]]:
        query_embedding = self._encode([query])[0]
        scores = self._article_embeddings @ query_embedding
        indexes = np.argsort(scores)[::-1][:limit]
        return [
            (
                self.articles[int(index)].article_id,
                float(scores[int(index)]),
                self.articles[int(index)].text[:1200],
            )
            for index in indexes
        ]

    def semantic(self, query: str, top_k: int = 5) -> list[SearchHit]:
        ranked = self._dense_ranks(
            query, self.semantic_units, self._semantic_embeddings, max(top_k * 4, 30)
        )
        return [
            SearchHit(self.article_by_id[key], score, "Sémantique", passage)
            for key, score, passage in ranked[:top_k]
        ]

    def hierarchical(self, query: str, top_k: int = 5) -> list[SearchHit]:
        ranked = self._dense_ranks(
            query, self.passage_units, self._passage_embeddings, max(top_k * 4, 30)
        )
        return [
            SearchHit(self.article_by_id[key], score, "Hiérarchique", passage)
            for key, score, passage in ranked[:top_k]
        ]

    def advanced(self, query: str, top_k: int = 5, candidate_k: int = 30) -> list[SearchHit]:
        dense = self._article_dense_ranks(query, max(candidate_k * 3, 60))
        bm25_scores = self._bm25.get_scores(tokenize(query))
        bm25_order = np.argsort(bm25_scores)[::-1][: max(candidate_k * 3, 60)]
        bm25 = [
            (self.articles[int(index)].article_id, float(bm25_scores[int(index)]), "")
            for index in bm25_order
        ]
        fused: dict[str, float] = {}
        passages: dict[str, str] = {}
        for ranked in (dense, bm25):
            for rank, (article_id, _score, passage) in enumerate(ranked, start=1):
                fused[article_id] = fused.get(article_id, 0.0) + 1.0 / (60 + rank)
                passages.setdefault(article_id, passage)

        requested_articles, instrument_ids = detect_references(query)
        for article_id in fused:
            article = self.article_by_id[article_id]
            article_number = article.article_number.lower()
            title = re.sub(r"[^a-z0-9]", "", article.document_title.lower())
            instrument_match = any(
                re.sub(r"[^a-z0-9]", "", identifier) in title
                for identifier in instrument_ids
            )
            if article_number in requested_articles and (
                not instrument_ids or instrument_match
            ):
                fused[article_id] += 0.08
            if instrument_match:
                fused[article_id] += 0.12

        candidates = sorted(fused, key=fused.__getitem__, reverse=True)[:candidate_k]
        if not candidates:
            return []
        if self._reranker is None:
            self._reranker = CrossEncoder(settings.reranker_model)
        pairs = [
            (query, self._article_text(self.article_by_id[article_id])[:5000])
            for article_id in candidates
        ]
        reranked_scores = self._reranker.predict(pairs, show_progress_bar=False)
        ordered = sorted(
            zip(candidates, reranked_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            SearchHit(
                self.article_by_id[article_id],
                float(score),
                "Avancée",
                passages.get(article_id, self.article_by_id[article_id].text[:1200]),
            )
            for article_id, score in ordered[:top_k]
        ]

    def search(self, query: str, method: str, top_k: int = 5) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("La question ne peut pas être vide.")
        methods = {
            "sémantique": self.semantic,
            "hiérarchique": self.hierarchical,
            "avancée": self.advanced,
        }
        try:
            search_method = methods[method.strip().lower()]
        except KeyError as error:
            raise ValueError(f"Méthode de recherche inconnue : {method}") from error
        return search_method(query, top_k)


def create_index(articles: list[Article]) -> RetrievalIndex:
    model = SentenceTransformer(settings.embedding_model)
    digest = hashlib.sha256()
    digest.update(f"{INDEX_VERSION}:{settings.embedding_model}:1400:800:120".encode())
    for article in articles:
        digest.update(article.article_id.encode("utf-8"))
        digest.update(article.text.encode("utf-8"))
        digest.update(article.section_context.encode("utf-8"))
    cache_path = settings.cache_dir / f"embeddings-{digest.hexdigest()[:20]}.npz"
    return RetrievalIndex(articles, model, cache_path)
