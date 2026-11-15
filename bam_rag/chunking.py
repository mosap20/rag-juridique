from __future__ import annotations

import re
from dataclasses import dataclass

from bam_rag.corpus import Article


@dataclass(frozen=True)
class TextChunk:
    article_id: str
    text: str
    page_start: int
    page_end: int


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n+", text) if part.strip()]


def semantic_chunks(articles: list[Article], max_chars: int = 1400) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for article in articles:
        context = f"{article.section_context}\n" if article.section_context else ""
        paragraphs = _paragraphs(context + article.text)
        current: list[str] = []
        size = 0
        for paragraph in paragraphs:
            if current and size + len(paragraph) > max_chars:
                chunks.append(
                    TextChunk(article.article_id, "\n".join(current), article.page_start, article.page_end)
                )
                current, size = [], 0
            current.append(paragraph)
            size += len(paragraph)
        if current:
            chunks.append(
                TextChunk(article.article_id, "\n".join(current), article.page_start, article.page_end)
            )
    return chunks


def hierarchical_chunks(
    articles: list[Article], max_chars: int = 800, overlap: int = 120
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for article in articles:
        text = re.sub(
            r"\s+",
            " ",
            f"{article.section_context} {article.text}" if article.section_context else article.text,
        ).strip()
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + max_chars // 2, end)
                if boundary > start:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(
                    TextChunk(article.article_id, chunk, article.page_start, article.page_end)
                )
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
