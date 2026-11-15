from __future__ import annotations

import pandas as pd
import streamlit as st

from bam_rag.config import settings
from bam_rag.evaluation import evaluate, load_questions, summarize
from bam_rag.ui import ensure_corpus_exists, get_articles, get_index, render_branding, show_setup_error

st.set_page_config(page_title="Résultats d’évaluation", page_icon="📊", layout="wide")
render_branding()
st.title("Évaluation des méthodes")
st.write(
    "Les métriques sont calculées uniquement à partir des questions et articles attendus "
    "présents dans `data/evaluation.csv`. Aucun score n’est simulé."
)

try:
    get_articles()
    catalog = (settings.data_dir / "article_catalog.csv").read_bytes()
except (OSError, ValueError) as error:
    st.warning(f"Le catalogue des articles est indisponible. {error}")
else:
    st.download_button(
        "Télécharger le catalogue des articles (CSV)",
        data=catalog,
        file_name="article_catalog.csv",
        mime="text/csv",
    )

try:
    questions = load_questions(settings.evaluation_file)
except (OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

if not questions:
    st.info(
        "Aucune question annotée n’est chargée. Copiez `data/evaluation_template.csv` vers "
        "`data/evaluation.csv`, ajoutez une question par ligne et choisissez `train` ou `test`."
    )
    st.stop()

split = st.selectbox("Jeu à évaluer", ["test", "train", "tous"])
selected_questions = (
    questions if split == "tous" else [question for question in questions if question.split == split]
)
if not selected_questions:
    st.warning(f"Aucune question n’est annotée pour le jeu « {split} ».")
    st.stop()

st.caption(f"{len(selected_questions)} question(s) sélectionnée(s).")
if st.button("Calculer les métriques", type="primary"):
    if not ensure_corpus_exists():
        st.stop()
    try:
        index = get_index()
        all_results = []
        progress = st.progress(0.0)
        methods = ["Sémantique", "Hiérarchique", "Avancée"]
        total = len(methods)
        for position, method in enumerate(methods, start=1):
            all_results.extend(evaluate(selected_questions, index, method, top_k=10))
            progress.progress(position / total)
        frame = pd.DataFrame(all_results)
        st.session_state["evaluation_results"] = frame
        st.session_state["evaluation_split"] = split
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        show_setup_error(error)

frame = st.session_state.get("evaluation_results")
evaluated_split = st.session_state.get("evaluation_split")
if frame is not None and evaluated_split == split:
    summaries = []
    for method, method_frame in frame.groupby("method"):
        summary = summarize(method_frame.to_dict(orient="records"))
        summaries.append({"Méthode": method, **summary})
    summary_frame = pd.DataFrame(summaries).set_index("Méthode")
    st.subheader("Métriques agrégées")
    st.dataframe(summary_frame.style.format("{:.3f}"), use_container_width=True)
    st.bar_chart(summary_frame[["hit_rate@1", "hit_rate@5", "reciprocal_rank"]])
    st.subheader("Résultat par question")
    st.dataframe(
        frame[["id", "question", "split", "method", "rank", "reciprocal_rank"]],
        use_container_width=True,
        hide_index=True,
    )
elif frame is not None:
    st.info("Le jeu sélectionné a changé. Cliquez sur « Calculer les métriques » pour l’évaluer.")
