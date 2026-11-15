from __future__ import annotations

import html

import streamlit as st

from bam_rag.config import settings
from bam_rag.corpus import Article, load_articles
from bam_rag.retrieval import RetrievalIndex, create_index


@st.cache_resource(show_spinner="Lecture et préparation du recueil…")
def get_articles() -> list[Article]:
    return load_articles()


@st.cache_resource(show_spinner="Chargement du modèle d’embeddings et indexation…")
def get_index() -> RetrievalIndex:
    return create_index(get_articles())


def show_setup_error(error: Exception) -> None:
    st.error(str(error))
    st.caption(
        "Vérifiez les étapes d’installation du README.md. Le téléchargement des modèles "
        "multilingues nécessite une connexion Internet lors de leur première utilisation."
    )


def render_page_band(articles: list[Article], total_pages: int = 1268) -> None:
    if not articles:
        return
    segments: list[str] = []
    for article in articles:
        start = max(1, min(total_pages, article.page_start))
        end = max(start, min(total_pages, article.page_end))
        left = (start - 1) / total_pages * 100
        width = max((end - start + 1) / total_pages * 100, 0.18)
        title = html.escape(f"{article.label}, p. {start}–{end}")
        segments.append(
            f'<span title="{title}" style="left:{left:.4f}%;width:{width:.4f}%"></span>'
        )
    st.markdown(
        '<div class="page-band"><div class="page-band-track">'
        + "".join(segments)
        + '</div><div class="page-band-labels"><span>p. 1</span>'
        + f'<span>p. {total_pages}</span></div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Repères colorés : pages du PDF associées aux articles retrouvés.")


def render_hits(hits, show_passage: bool = True) -> None:
    for rank, hit in enumerate(hits, start=1):
        article = hit.article
        page_label = (
            f"p. {article.page_start}"
            if article.page_start == article.page_end
            else f"p. {article.page_start}–{article.page_end}"
        )
        with st.container(border=True):
            st.markdown(f"**[{rank}] {article.document_title} — Article {article.article_number}**")
            st.caption(
                f"PDF {page_label} · score {hit.score:.4f} · identifiant `{article.article_id}`"
            )
            if article.section_context:
                st.caption(article.section_context)
            st.write(article.text[:1800] + ("…" if len(article.text) > 1800 else ""))
            if show_passage and hit.matched_passage:
                with st.expander("Passage ayant déclenché la recherche"):
                    st.write(hit.matched_passage[:1400])


def ensure_corpus_exists() -> bool:
    if settings.corpus_pdf.is_file():
        return True
    st.error(
        "Le recueil PDF est introuvable. Déposez le PDF à la racine du projet, "
        "à côté de app.py."
    )
    return False


def render_branding() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1180px; padding-top: 2rem; }
        .page-band { margin: 0.8rem 0 1.5rem; }
        .page-band-track { position: relative; height: 18px; border-radius: 9px;
          background: #e7ebf2; overflow: hidden; }
        .page-band-track span { position: absolute; top: 0; height: 100%;
          background: #c45132; opacity: .9; border-radius: 4px; }
        .page-band-labels { display: flex; justify-content: space-between;
          color: #64748b; font-size: .75rem; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
