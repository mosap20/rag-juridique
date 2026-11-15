from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    corpus_pdf: Path = ROOT_DIR / "Recueil des textes législatifs et réglementaires de BAM _ VF.pdf"
    data_dir: Path = ROOT_DIR / "data"
    cache_dir: Path = ROOT_DIR / "data" / ".cache"
    evaluation_file: Path = ROOT_DIR / "data" / "evaluation.csv"
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")


settings = Settings()
