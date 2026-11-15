from __future__ import annotations

import streamlit as st

from bam_rag.ui import (
    ensure_corpus_exists,
    get_index,
    render_branding,
    render_hits,
    show_setup_error,
)

st.set_page_config(page_title="Comparaison des approches", page_icon="⚖️", layout="wide")
render_branding()
st.title("Comparaison des approches")
st.write("La même question est lancée sur les trois méthodes, avec le même corpus.")

with st.form("compare_form"):
    question = st.text_input("Question", placeholder="Ex. Comment la politique monétaire est-elle conduite ?")
    top_k = st.slider("Articles affichés par méthode", 3, 10, 5)
    submitted = st.form_submit_button("Comparer", type="primary")

if submitted:
    if not question.strip():
        st.warning("Saisissez une question.")
    elif ensure_corpus_exists():
        try:
            index = get_index()
            st.session_state["comparison_question"] = question
            st.session_state["comparison_results"] = {
                method: index.search(question, method, top_k)
                for method in ["Sémantique", "Hiérarchique", "Avancée"]
            }
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
            show_setup_error(error)

results = st.session_state.get("comparison_results")
if results:
    columns = st.columns(3)
    for column, method in zip(columns, ["Sémantique", "Hiérarchique", "Avancée"]):
        with column:
            st.subheader(method)
            render_hits(results[method], show_passage=False)
