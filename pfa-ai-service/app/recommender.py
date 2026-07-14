"""
Moteur de recommandation basé sur le contenu (content-based filtering).

Remplace la formule linéaire fixe (distanceScore*70 + budgetScore*30) du backend
Java par un score de similarité cosinus entre :
  - un vecteur de caractéristiques par lieu candidat (catégorie one-hot, proximité,
    adéquation budget, popularité) ;
  - un vecteur "profil utilisateur" construit soit à partir de son historique de
    catégories favorites (Preference/Favorite), soit d'un profil par défaut.

Chaque dimension étant explicite, le score est justifiable/traçable (voir
`breakdown` dans la réponse), ce qui répond au reproche de "justification
insuffisante des choix techniques".
"""
from math import radians, sin, cos, asin, sqrt
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas import CATEGORIES, PlaceCandidate, RecommendationItem, UserContext

EARTH_RADIUS_KM = 6371


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def _category_one_hot(category: str) -> np.ndarray:
    vec = np.zeros(len(CATEGORIES))
    if category in CATEGORIES:
        vec[CATEGORIES.index(category)] = 1.0
    return vec


def _place_vector(place: PlaceCandidate, user: UserContext, max_popularity: float) -> np.ndarray:
    distance_km = haversine_km(user.lat, user.lon, place.lat, place.lon)
    distance_score = 1.0 / (1.0 + distance_km)

    if place.budget <= user.budget_max:
        budget_score = 1.0 - (place.budget / user.budget_max) * 0.5  # reste entre 0.5 et 1
    else:
        # Dépassement de budget : pénalité progressive plutôt qu'exclusion brutale
        overshoot = (place.budget - user.budget_max) / user.budget_max
        budget_score = max(0.0, 0.5 - overshoot)

    popularity = place.review_count + place.favorite_count
    popularity_score = popularity / max_popularity if max_popularity > 0 else 0.0

    return np.concatenate([
        _category_one_hot(place.category),
        [distance_score, budget_score, popularity_score],
    ])


def _user_profile_vector(user: UserContext) -> np.ndarray:
    category_pref = np.zeros(len(CATEGORIES))
    if user.category_history:
        for cat in user.category_history:
            if cat in CATEGORIES:
                category_pref[CATEGORIES.index(cat)] += 1.0
        total = category_pref.sum()
        if total > 0:
            category_pref = category_pref / total
    else:
        # Cold start : aucune préférence connue -> profil neutre (pas de catégorie privilégiée)
        category_pref[:] = 1.0 / len(CATEGORIES)

    # Utilisateur idéal : proche, dans son budget, populaire (scores maximaux)
    ideal_numeric = [1.0, 1.0, 1.0]
    return np.concatenate([category_pref, ideal_numeric])


def recommend(places: List[PlaceCandidate], user: UserContext, top_k: int) -> List[RecommendationItem]:
    if not places:
        return []

    max_popularity = max((p.review_count + p.favorite_count for p in places), default=0)
    user_vector = _user_profile_vector(user).reshape(1, -1)

    scored = []
    for place in places:
        place_vector = _place_vector(place, user, max_popularity)
        similarity = float(cosine_similarity(place_vector.reshape(1, -1), user_vector)[0][0])

        distance_km = haversine_km(user.lat, user.lon, place.lat, place.lon)
        breakdown = {
            "distance_km": round(distance_km, 2),
            "category_match": place.category in (user.category_history or []),
            "within_budget": place.budget <= user.budget_max,
            "popularity": place.review_count + place.favorite_count,
            "cosine_similarity": round(similarity, 4),
        }
        scored.append(RecommendationItem(
            place_id=place.id, name=place.name, score=round(similarity, 4), breakdown=breakdown,
        ))

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def baseline_linear_score(place: PlaceCandidate, user: UserContext) -> float:
    """Réimplémentation fidèle de l'ancienne formule Java (CircuitService.score),
    utilisée uniquement par evaluate_recommender.py pour comparer les deux approches."""
    distance_km = haversine_km(user.lat, user.lon, place.lat, place.lon)
    distance_score = 1.0 / (1.0 + distance_km)
    budget_score = 1.0 / (1.0 + place.budget / 100.0)
    return distance_score * 70 + budget_score * 30
