# pfa-ai-service

Microservice Python (FastAPI) hébergeant les briques IA du projet Rihla :
- `/recommend` : moteur de recommandation par filtrage basé sur le contenu (similarité cosinus).
- `/chat` : chatbot hybride (classification d'intentions locale entraînée + LLM externe gratuit en secours).

Ce service est **sans état** : il ne se connecte pas à la base MySQL. Le backend Spring Boot
(`pfa-tourisme-backend`) lui envoie les données nécessaires (lieux candidats, contexte utilisateur)
en JSON et reste la seule source de vérité.

## Installation

```bash
cd pfa-ai-service
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Entraîner et évaluer les modèles

```bash
python train_intent_model.py       # entraîne le classifieur d'intentions, affiche accuracy/F1/matrice de confusion
python evaluate_recommender.py     # compare le recommandeur (cosine) à l'ancienne formule linéaire, Précision@k
```

## Lancer le service

```bash
uvicorn app.main:app --reload --port 8001
```

## Clé LLM (optionnelle mais recommandée pour /chat)

Pour des réponses naturelles sur les questions générales/ambiguës, définir une clé API
Google Gemini (offre gratuite) avant de lancer le service :

```bash
set GEMINI_API_KEY=votre_cle          # Windows cmd
$env:GEMINI_API_KEY="votre_cle"       # PowerShell
```

Sans clé, `/chat` retombe automatiquement sur un message générique — le service reste
utilisable hors-ligne (dégradation gracieuse).

## Endpoints

- `GET /health` — vérification de disponibilité.
- `POST /recommend` — voir `app/schemas.py` (`RecommendRequest`/`RecommendResponse`).
- `POST /chat` — voir `app/schemas.py` (`ChatRequest`/`ChatResponse`).
