# RAG juridique — Recueil de Bank Al-Maghrib

Application de démonstration en français pour interroger le recueil PDF de Bank Al-Maghrib. Elle
extrait les articles, compare trois méthodes de recherche et peut demander à Gemini de rédiger une
réponse avec des citations vers les articles et les pages du PDF.

> **Document source non inclus :** le PDF réglementaire n’est pas distribué dans ce dépôt.
> Procurez-vous une copie depuis une source autorisée et placez-la à la racine du projet sous le
> nom `Recueil des textes législatifs et réglementaires de BAM _ VF.pdf` avant le premier lancement.
> Le fichier est exclu de Git. La clé Gemini est également locale et ne doit jamais être publiée.

## Contenu

- **Assistant** : recherche, articles sources, pages du recueil consultées et réponse Gemini facultative.
- **Comparaison** : mêmes questions lancées sur les trois méthodes.
- **Résultats** : calcul de métriques à partir d’annotations réelles.
- **Méthodes** : explication des algorithmes et du protocole.

## Prérequis et lancement (Windows PowerShell)

Python 3.10 ou plus récent est nécessaire.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
streamlit run app.py
```

La recherche nécessite le PDF source indiqué ci-dessus, à la racine à côté de `app.py`. La clé
Gemini est facultative pour la recherche et ne doit être renseignée que dans le fichier local
`.env` (`GEMINI_API_KEY=...`). Ne la commitez pas et ne la publiez pas.

Lors du premier lancement, l’application extrait les articles du PDF, construit les embeddings
et télécharge les modèles multilingues Sentence Transformers configurés dans `.env`. Une connexion
Internet est nécessaire pour ce téléchargement initial. La première indexation peut prendre
plusieurs minutes sur CPU. Les articles et embeddings sont mis en cache dans `data/.cache/`;
les fichiers du cache sont régénérables.

## Les trois approches

1. **Sémantique** — embeddings multilingues de blocs formés à partir des paragraphes de chaque
   article ; un bloc pertinent fait remonter l’article entier.
2. **Hiérarchique** — embeddings de passages courts qui se chevauchent ; le meilleur passage
   détermine l’article retourné.
3. **Avancée** — classement dense des articles + BM25, fusion Reciprocal Rank Fusion, signal
   supplémentaire pour les numéros de textes/articles cités explicitement et reranking
   cross-encoder des 30 meilleurs candidats.

Les modèles peuvent être remplacés via `EMBEDDING_MODEL` et `RERANKER_MODEL` dans `.env`.
Le reranker ne se charge que lorsque la recherche avancée est utilisée.

## Évaluation reproductible

Le jeu d’évaluation de 41 questions annotées n’a pas été fourni avec le PDF. L’application
n’invente ni ces questions ni leurs résultats ; les scores annoncés oralement ne sont donc pas
reproduits dans cette version.

1. Après une première recherche, consulter `data/article_catalog.csv` : il contient l’identifiant
   stable (`article_id`), le texte réglementaire, le numéro d’article, le contexte de section et
   les pages PDF.
2. Copier le modèle et ajouter les annotations :

   ```powershell
   Copy-Item data/evaluation_template.csv data/evaluation.csv
   ```

3. Ajouter une ligne par question dans `data/evaluation.csv` avec les colonnes :
   `id,question,expected_article_id,split`. `expected_article_id` doit correspondre exactement à
   l’identifiant du catalogue ; `split` vaut `train` ou `test`.
4. Utiliser le jeu `train` pour régler les méthodes, puis le jeu `test` une seule fois pour le
   résultat final. Ne réutilisez pas les questions de test pour choisir les paramètres.

L’écran **Résultats** calcule Hit Rate/Recall@k, MRR (rang réciproque moyen) et NDCG@k pour les
trois méthodes. Avec un seul article attendu par question, Recall@k et Hit Rate@k ont la même
valeur. Les scores affichés sont calculés par l’application sur le fichier chargé.

## Limites à vérifier avant la soutenance

- Le parseur a extrait 3 818 articles du PDF fourni, contre environ 4 500 annoncés dans le brief.
  Ce décalage doit être audité avec l’équipe et quelques pages du recueil avant de présenter
  l’index comme exhaustif.
- L’extraction s’appuie sur les lignes `Article …` du PDF et sur le texte précédant le premier
  article pour repérer le titre du texte. Il faut relire quelques entrées de `article_catalog.csv`
  (articles longs, textes avec amendements et articles bis/ter notamment).
- Les PDF scannés sans couche texte ne sont pas OCRisés et ne seront pas indexables.
- Gemini peut produire des citations mal placées : les articles retrouvés sont montrés à côté de
  la réponse afin de permettre la vérification humaine.
- Les performances dépendent du modèle téléchargé, des paramètres et des annotations. Indiquez
  ces éléments lors de la comparaison des résultats.

## Organisation du code

- [app.py](app.py) et [pages/](pages/) contiennent les écrans Streamlit.
- [corpus.py](bam_rag/corpus.py) extrait les articles et leurs références de page. [chunking.py](bam_rag/chunking.py) produit les passages à indexer.
- [retrieval.py](bam_rag/retrieval.py) implémente les trois méthodes et [answer.py](bam_rag/answer.py) prépare la réponse Gemini.
- [evaluation.py](bam_rag/evaluation.py) calcule les métriques de classement à partir des annotations.

## Tests

Les tests couvrent la segmentation des articles, le suivi des pages sources et le calcul des métriques sur des exemples synthétiques. Ils ne nécessitent ni le recueil ni une clé Gemini.

```powershell
python -m pytest
```
