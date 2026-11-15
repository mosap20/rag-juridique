from __future__ import annotations

import re

import httpx

from bam_rag.config import settings
from bam_rag.retrieval import SearchHit


class GeminiError(RuntimeError):
    pass


def _format_source(index: int, hit: SearchHit) -> str:
    article = hit.article
    page_range = (
        f"–{article.page_end}" if article.page_end != article.page_start else ""
    )
    section = f"Contexte : {article.section_context}\n" if article.section_context else ""
    return (
        f"[{index}] {article.label} (PDF, p. {article.page_start}{page_range})\n"
        f"{section}{article.text[:6000]}"
    )


def generate_answer(question: str, hits: list[SearchHit]) -> str:
    if not settings.gemini_api_key:
        raise GeminiError("Renseignez GEMINI_API_KEY dans le fichier local .env.")
    if not hits:
        raise ValueError("Aucun article n’a été trouvé pour préparer une réponse.")

    from google import genai
    from google.genai.errors import APIError

    sources = "\n\n".join(
        _format_source(index, hit) for index, hit in enumerate(hits, start=1)
    )
    prompt = f"""Tu es un assistant documentaire qui répond en français sur la réglementation bancaire marocaine.
Réponds uniquement à partir des extraits ci-dessous. Si les extraits ne permettent pas de répondre,
dis-le clairement et ne complète pas avec des connaissances externes. Pour chaque affirmation juridique,
ajoute une citation de source au format [1], [2], etc. N'invente pas de citation.

Question : {question}

Extraits autorisés :
{sources}
"""
    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(model=settings.gemini_model, contents=prompt)
    except APIError as error:
        if error.code == 429:
            raise GeminiError("Quota Gemini atteint. Réessayez plus tard.") from error
        if error.code in {401, 403}:
            raise GeminiError("Accès à Gemini refusé. Vérifiez la clé et ses autorisations.") from error
        raise GeminiError(
            f"Gemini a renvoyé une erreur ({error.code}). Vérifiez la clé et le modèle configurés."
        ) from error
    except httpx.RequestError as error:
        raise GeminiError("Connexion à Gemini impossible. Vérifiez votre connexion et réessayez.") from error

    answer = response.text
    if not answer:
        raise GeminiError("Gemini n’a renvoyé aucun texte de réponse.")

    invalid = {
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer)
        if int(number) < 1 or int(number) > len(hits)
    }
    if invalid:
        raise GeminiError(
            "Gemini a renvoyé des références qui ne correspondent pas aux sources fournies : "
            + ", ".join(f"[{number}]" for number in sorted(invalid))
        )
    return answer
