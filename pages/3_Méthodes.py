from __future__ import annotations

import streamlit as st

from bam_rag.ui import render_branding

st.set_page_config(page_title="Méthodes", page_icon="📚", layout="wide")
render_branding()
st.title("Comment fonctionne le système ?")

st.header("1. Recherche sémantique")
st.write(
    "Chaque article est découpé en blocs cohérents à partir de ses paragraphes. "
    "Un modèle multilingue transforme les blocs et la question en vecteurs. "
    "La similarité cosinus classe les blocs ; le meilleur bloc représente ensuite son article."
)
st.markdown(
    "**Atout :** retrouve des formulations proches par le sens, même si les mots diffèrent.  \n"
    "**Limite :** peut négliger un numéro d’article ou un terme juridique précis."
)

st.header("2. Recherche hiérarchique")
st.write(
    "Chaque article est découpé en passages courts avec chevauchement. Le système recherche "
    "les passages par embeddings, puis remonte à l’article complet ayant le passage le mieux "
    "classé. Le contexte rendu à l’utilisateur reste l’article entier."
)
st.markdown(
    "**Atout :** les passages ciblent les paragraphes pertinents dans les articles longs.  \n"
    "**Limite :** un passage isolé peut perdre le contexte ou la structure juridique."
)

st.header("3. Recherche avancée : hybride et rerankée")
st.write(
    "La méthode combine le classement sémantique des articles et BM25, qui compte les termes "
    "rares et les correspondances lexicales. Les listes sont fusionnées par Reciprocal Rank "
    "Fusion (RRF). Les références explicites (numéro d’article ou de texte) ajoutent un signal, "
    "puis un cross-encoder réordonne les 30 premiers candidats en lisant la question et l’article."
)
st.markdown(
    "**Atout :** combine sens, mots-clés et références juridiques explicites.  \n"
    "**Limite :** le reranker ajoute un coût de calcul et sa qualité dépend du modèle choisi."
)

st.header("Génération avec sources")
st.write(
    "Gemini reçoit seulement les articles retrouvés. Le prompt lui demande de répondre en français, "
    "de ne pas compléter les lacunes avec des connaissances externes et de citer chaque affirmation "
    "juridique avec `[1]`, `[2]`, etc. Les pages du PDF sont également affichées."
)

st.header("Protocole d’évaluation")
st.write(
    "Chaque question doit être associée à un `article_id` exact du catalogue et à un split "
    "`train` ou `test`. Le jeu de test doit rester hors du réglage des méthodes. Le tableau Résultats "
    "calcule Hit Rate / Recall@k, MRR et NDCG@k sur les seules annotations fournies."
)
st.warning(
    "Le jeu des 41 questions annotées n’étant pas fourni dans ce dossier, aucun résultat "
    "de performance n’est présenté comme reproduit ou validé."
)
