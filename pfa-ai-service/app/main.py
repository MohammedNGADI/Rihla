from fastapi import FastAPI

from app.intent_classifier import intent_classifier
from app.llm_client import generate_reply
from app.recommender import recommend
from app.schemas import ChatRequest, ChatResponse, RecommendRequest, RecommendResponse

app = FastAPI(title="pfa-ai-service", version="0.1.0")

RULE_TEMPLATES = {
    "find_place": {
        "fr": "Voici quelques lieux à découvrir : {places}.",
        "en": "Here are some places to explore: {places}.",
    },
    "find_restaurant": {
        "fr": "Pour bien manger, je vous suggère : {places}.",
        "en": "For a good meal, I'd suggest: {places}.",
    },
    "create_circuit": {
        "fr": "Je peux générer un circuit personnalisé selon votre budget et vos préférences. On commence ?",
        "en": "I can generate a custom route based on your budget and preferences. Shall we start?",
    },
    "ask_weather": {
        "fr": "Consultez l'onglet météo de l'application pour la température en temps réel à Rabat.",
        "en": "Check the weather tab in the app for the real-time temperature in Rabat.",
    },
}

NO_CONTEXT_FALLBACK = {
    "fr": "Je n'ai pas encore de lieux à vous montrer pour cette demande, mais je peux vous aider à en trouver.",
    "en": "I don't have places to show you for this yet, but I can help you find some.",
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def load_models():
    intent_classifier.load()


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(request: RecommendRequest):
    items = recommend(request.places, request.user, request.top_k)
    return RecommendResponse(recommendations=items)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    lang = request.lang if request.lang in ("fr", "en") else "fr"
    intent, confidence = intent_classifier.predict(request.message)

    if intent in RULE_TEMPLATES:
        template = RULE_TEMPLATES[intent][lang]
        if "{places}" in template:
            if request.context_places:
                names = ", ".join(p.name for p in request.context_places[:3])
                reply = template.format(places=names)
            else:
                reply = NO_CONTEXT_FALLBACK[lang]
        else:
            reply = template
        return ChatResponse(reply=reply, intent=intent, confidence=confidence, source="rules")

    reply = await generate_reply(request.message, lang, request.context_places)
    return ChatResponse(reply=reply, intent=intent, confidence=confidence, source="llm")
