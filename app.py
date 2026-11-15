from __future__ import annotations

import streamlit as st

from bam_rag.answer import GeminiError, generate_answer
from bam_rag.config import settings
from bam_rag.ui import (
    ensure_corpus_exists,
    get_index,
    render_branding,
    render_hits,
    render_page_band,
    show_setup_error,
)

st.set_page_config(page_title="RAG juridique BAM", page_icon="⚖️", layout="wide")
render_branding()
st.title("Assistant juridique — Bank Al-Maghrib")
st.caption(
    "Interrogez le Recueil des textes législatifs et réglementaires. "
    "Les réponses Gemini s’appuient sur les articles affichés et les citent."
)

with st.sidebar:
    st.header("Recherche")
    method = st.selectbox("Approche", ["Avancée", "Sémantique", "Hiérarchique"])
    top_k = st.slider("Articles à retrouver", min_value=3, max_value=10, value=5)
    use_gemini = st.checkbox(
        "Générer une réponse avec Gemini",
        value=bool(settings.gemini_api_key),
        help="La clé est lue depuis GEMINI_API_KEY dans le fichier local .env.",
    )

with st.form("question_form"):
    question = st.text_area(
        "Votre question en français",
        placeholder="Ex. Quelles sont les missions fondamentales de Bank Al-Maghrib ?",
        height=100,
    )
    submitted = st.form_submit_button("Rechercher", type="primary")

if submitted:
    if not question.strip():
        st.warning("Saisissez une question avant de lancer la recherche.")
    elif ensure_corpus_exists():
        st.session_state.pop("last_hits", None)
        st.session_state.pop("last_answer", None)
        try:
            index = get_index()
            st.session_state["last_hits"] = index.search(question, method, top_k)
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
            show_setup_error(error)
        else:
            hits = st.session_state["last_hits"]
            if not hits:
                st.info("Aucun article n’a été retrouvé.")
            elif use_gemini:
                with st.spinner("Gemini rédige une réponse à partir des sources…"):
                    try:
                        st.session_state["last_answer"] = generate_answer(question, hits)
                    except GeminiError as error:
                        st.warning(str(error))
                    except ImportError:
                        st.warning("Le module Gemini est indisponible. Vérifiez les dépendances du projet.")

hits = st.session_state.get("last_hits", [])
if hits:
    st.subheader("Sources retrouvées")
    render_page_band([hit.article for hit in hits])
    if answer := st.session_state.get("last_answer"):
        st.subheader("Réponse")
        st.markdown(answer)
    render_hits(hits)
