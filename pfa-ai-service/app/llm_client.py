"""
Appel à un LLM externe (offre gratuite) pour générer une réponse naturelle
quand l'intention détectée localement est ambiguë ou générale ("general").

Fournisseur par défaut : Google Gemini (tier gratuit généreux, endpoint REST
simple, pas de SDK requis). Clé lue depuis la variable d'environnement
GEMINI_API_KEY -- jamais committée. Si la clé est absente ou l'appel échoue,
on retombe sur un message générique (le chatbot reste utilisable hors-ligne).
"""
import os
from typing import List

import httpx

from app.schemas import PlaceCandidate

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

FALLBACK_REPLY = {
    "fr": "Je peux vous suggérer des lieux, des restaurants, la météo ou créer un circuit. Que préférez-vous ?",
    "en": "I can suggest places, restaurants, weather, or create a route. What would you like?",
}


def _build_prompt(message: str, lang: str, context_places: List[PlaceCandidate]) -> str:
    places_text = ""
    if context_places:
        listed = "\n".join(f"- {p.name} ({p.category}, {p.budget} MAD)" for p in context_places[:5])
        places_text = f"\nLieux réels disponibles à mentionner si pertinent :\n{listed}\n"

    lang_instruction = "Réponds en français." if lang == "fr" else "Answer in English."
    return (
        "Tu es l'assistant de voyage de l'application Rihla, spécialisée dans le tourisme à Rabat. "
        "Réponds de façon brève (2-3 phrases), amicale et utile. "
        f"{lang_instruction}"
        f"{places_text}\n"
        f"Message de l'utilisateur : {message}"
    )


async def generate_reply(message: str, lang: str, context_places: List[PlaceCandidate]) -> str:
    if not GEMINI_API_KEY:
        return FALLBACK_REPLY.get(lang, FALLBACK_REPLY["fr"])

    prompt = _build_prompt(message, lang, context_places)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(
                GEMINI_URL, params={"key": GEMINI_API_KEY}, json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return FALLBACK_REPLY.get(lang, FALLBACK_REPLY["fr"])
