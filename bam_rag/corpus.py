from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from pypdf import PdfReader

from bam_rag.config import settings

ARTICLE_PREFIX = re.compile(
    r"^\s*Article\s+(?P<number>premier|1er|deuxième|deux|troisième|trois|"
    r"quatrième|quatre|cinquième|cinq|sixième|six|septième|sept|huitième|huit|"
    r"neuvième|neuf|dixième|dix|[0-9]{1,3}(?:[-.][0-9]{1,3})*"
    r"(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|undecies))?)",
    re.IGNORECASE,
)
PARSER_VERSION = 7
SECTION_HEADER = re.compile(
    r"^(?:TITRE\b|Chapitre\b|Section\b|Sous-section\b|Paragraphe\b)", re.IGNORECASE
)
INSTRUMENT_HEADER = re.compile(
    r"^\s*(?:Dahir|Loi|Décret|Decret|Arrêté|Arreté|Circulaire|Directive|"
    r"Décision|Decision|Lettre circulaire|Charte|Code|Recommandation)\b",
    re.IGNORECASE,
)
SKIP_LINE = re.compile(
    r"^(?:PUBLIC|Mis à jour à fin \d{4}|\d{1,4}|"
    r"RECUEIL DES TEXTES LEGISLATIFS.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Article:
    article_id: str
    document_title: str
    article_number: str
    text: str
    page_start: int
    page_end: int
    section_context: str = ""

    @property
    def label(self) -> str:
        return f"{self.document_title} — Article {self.article_number}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_number(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    names = {
        "premier": "1er",
        "deux": "2",
        "deuxième": "2",
        "trois": "3",
        "troisième": "3",
        "quatre": "4",
        "quatrième": "4",
        "cinq": "5",
        "cinquième": "5",
        "six": "6",
        "sixième": "6",
        "sept": "7",
        "septième": "7",
        "huit": "8",
        "huitième": "8",
        "neuf": "9",
        "neuvième": "9",
        "dix": "10",
        "dixième": "10",
    }
    return names.get(normalized, normalized)


def _article_heading(line: str, page_number: int | None = None) -> tuple[str, str] | None:
    candidate = line.strip()
    # A complete heading can legitimately end with the current page number.
    if (
        page_number is not None
        and candidate.endswith(str(page_number))
        and not ARTICLE_PREFIX.fullmatch(candidate)
    ):
        without_page = candidate[: -len(str(page_number))].rstrip()
        if ARTICLE_PREFIX.fullmatch(without_page):
            candidate = without_page

    merged_number = re.fullmatch(r"\s*Article\s+(\d{4,5})\s*", candidate, re.IGNORECASE)
    if merged_number:
        return merged_number.group(1)[:-2], ""

    match = ARTICLE_PREFIX.match(candidate)
    if not match:
        return None

    number = _normalize_number(match.group("number"))
    remainder = candidate[match.end() :].strip()

    if remainder.startswith((":", "-", "–", "—")):
        suffix = remainder[1:].strip()
        return number, suffix
    if re.fullmatch(r"\d{1,3}", remainder):
        return number, ""
    if not remainder or remainder in {".", ":", ";"}:
        return number, ""
    if remainder.startswith("("):
        return number, remainder
    return None


def _slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:72] or 'texte'}-{digest}"


def _clean_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    if not line or SKIP_LINE.match(line):
        return ""
    return line


def _read_pages(pdf_path: Path) -> Iterator[tuple[int, str]]:
    try:
        import pymupdf
    except ImportError:
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            yield page_number, page.extract_text() or ""
        return

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            yield page_number, page.get_text("text")


def _instrument_title(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if not INSTRUMENT_HEADER.match(line):
            continue
        if index > 10 or not line[:1].isupper() or len(line.strip()) < 16:
            continue
        if re.match(r"^\s*Code\b", line, re.IGNORECASE) and not re.search(
            r"\b(?:déontologie|de place|bonne conduite|conduite bancaire)\b",
            line,
            re.IGNORECASE,
        ):
            continue
        title_lines = [line]
        for following in lines[index + 1 :]:
            cleaned = _clean_line(following)
            if not cleaned:
                continue
            if _article_heading(cleaned) or len(title_lines) >= 5:
                break
            if re.match(r"^(?:LOUANGE|Vu\b|Que\b|Est promulguée|La présente|Le Wali)\b", cleaned):
                break
            if INSTRUMENT_HEADER.match(cleaned):
                break
            title_lines.append(cleaned)
        return re.sub(r"\s+", " ", " ".join(title_lines)).strip()
    return None


def parse_article_pages(pages: Iterable[tuple[int, str]]) -> list[Article]:
    articles: list[Article] = []
    current_title = "Texte réglementaire non identifié"
    active: dict[str, object] | None = None
    section_context: list[str] = []
    expecting_section_title = False
    occurrences: dict[tuple[str, str], int] = {}

    def flush() -> None:
        nonlocal active
        if active is None:
            return
        body = "\n".join(active["lines"]).strip()  # type: ignore[arg-type]
        if len(body) >= 20:
            title = str(active["document_title"])
            number = str(active["article_number"])
            key = (title, number)
            occurrence = occurrences.get(key, 0)
            occurrences[key] = occurrence + 1
            articles.append(
                Article(
                    article_id=f"{_slug(title)}::{number}::{occurrence}",
                    document_title=title,
                    article_number=number,
                    text=body,
                    page_start=int(active["page_start"]),
                    page_end=int(active["page_end"]),
                    section_context=str(active["section_context"]),
                )
            )
        active = None

    for page_number, page_text in pages:
        if page_number < 20:
            continue
        lines = (page_text or "").splitlines()
        article_positions = [
            index
            for index, line in enumerate(lines)
            if _article_heading(line.strip(), page_number)
        ]
        if not article_positions:
            for line in lines:
                cleaned = _clean_line(line)
                if not cleaned:
                    continue
                if SECTION_HEADER.match(cleaned):
                    section_context.append(cleaned)
                    section_context[:] = section_context[-4:]
                    expecting_section_title = True
                elif expecting_section_title:
                    section_context.append(cleaned)
                    section_context[:] = section_context[-4:]
                    expecting_section_title = False
                elif active is not None:
                    active["lines"].append(cleaned)
                    active["page_end"] = page_number
            continue

        title = _instrument_title(lines[: article_positions[0]])
        if title and title != current_title:
            flush()
            current_title = title
            section_context.clear()
            expecting_section_title = False

        for index, line in enumerate(lines):
            heading = _article_heading(line.strip(), page_number)
            if heading:
                flush()
                expecting_section_title = False
                number, heading_text = heading
                active = {
                    "document_title": current_title,
                    "article_number": number,
                    "page_start": page_number,
                    "page_end": page_number,
                    "section_context": " — ".join(section_context[-4:]),
                    "lines": [f"Article {number}"] + ([heading_text] if heading_text else []),
                }
                continue
            cleaned = _clean_line(line)
            if not cleaned:
                continue
            if SECTION_HEADER.match(cleaned):
                section_context.append(cleaned)
                section_context[:] = section_context[-4:]
                expecting_section_title = True
            elif expecting_section_title:
                section_context.append(cleaned)
                section_context[:] = section_context[-4:]
                expecting_section_title = False
            elif active is not None:
                if not re.match(r"^\d+\s+Publié au Bulletin officiel", cleaned, re.I):
                    active["lines"].append(cleaned)
                active["page_end"] = page_number

    flush()
    return articles


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_articles(pdf_path: Path | None = None) -> list[Article]:
    path = pdf_path or settings.corpus_pdf
    if not path.is_file():
        raise FileNotFoundError(
            f"Recueil PDF introuvable : {path}. Déposez le PDF à la racine du projet."
        )

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = settings.cache_dir / "articles.json"
    fingerprint = _file_fingerprint(path)
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if (
            cached.get("fingerprint") == fingerprint
            and cached.get("parser_version") == PARSER_VERSION
        ):
            articles = [Article(**record) for record in cached["articles"]]
            catalog_file = settings.data_dir / "article_catalog.csv"
            if not catalog_file.exists():
                _write_catalog(articles, catalog_file)
            return articles

    articles = parse_article_pages(_read_pages(path))
    if not articles:
        raise ValueError(
            "Aucun article n’a été extrait du PDF. Vérifiez que le document contient du texte "
            "sélectionnable plutôt que des pages uniquement scannées."
        )
    payload = {
        "fingerprint": fingerprint,
        "parser_version": PARSER_VERSION,
        "articles": [article.to_dict() for article in articles],
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _write_catalog(articles, settings.data_dir / "article_catalog.csv")
    return articles


def _write_catalog(articles: list[Article], destination: Path) -> None:
    import csv

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "article_id",
                "document_title",
                "article_number",
                "section_context",
                "page_start",
                "page_end",
            ],
        )
        writer.writeheader()
        for article in articles:
            writer.writerow(
                {
                    "article_id": article.article_id,
                    "document_title": article.document_title,
                    "article_number": article.article_number,
                    "section_context": article.section_context,
                    "page_start": article.page_start,
                    "page_end": article.page_end,
                }
            )
