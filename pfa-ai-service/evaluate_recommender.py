"""
Compare le nouveau recommandeur (filtrage basé contenu, similarité cosinus)
à l'ancienne formule linéaire du backend Java (distanceScore*70 + budgetScore*30),
via une métrique de Précision@k.

IMPORTANT (honnêteté méthodologique, à rappeler dans le rapport) : le modèle de
données actuel (entité Review) ne contient pas de note numérique explicite, donc
aucune "vérité terrain" d'interactions utilisateur réelles n'est disponible pour
l'instant. Cette évaluation utilise donc un jeu de données SYNTHÉTIQUE : des
utilisateurs et lieux générés aléatoirement, avec une pertinence de référence
définie par une règle de bon sens indépendante des deux algorithmes comparés
(catégorie préférée + budget respecté + à proximité). Cela permet de comparer
objectivement les deux approches sur un terrain contrôlé, mais ne remplace pas
une évaluation sur données réelles (perspective d'amélioration : ajouter une
note explicite aux avis pour permettre une évaluation en conditions réelles).

Usage: python evaluate_recommender.py
"""
import random

from app.recommender import baseline_linear_score, haversine_km, recommend
from app.schemas import CATEGORIES, PlaceCandidate, UserContext

RABAT_LAT, RABAT_LON = 34.020882, -6.841650
TOP_K = 5
N_PLACES = 150
N_USERS = 40


def make_places(seed: int) -> list[PlaceCandidate]:
    rng = random.Random(seed)
    places = []
    for i in range(N_PLACES):
        places.append(PlaceCandidate(
            id=i,
            name=f"Place-{i}",
            category=rng.choice(CATEGORIES),
            budget=rng.uniform(0, 400),
            lat=RABAT_LAT + rng.uniform(-0.08, 0.08),
            lon=RABAT_LON + rng.uniform(-0.08, 0.08),
            review_count=rng.randint(0, 50),
            favorite_count=rng.randint(0, 30),
        ))
    return places


def make_users(seed: int) -> list[UserContext]:
    rng = random.Random(seed)
    users = []
    for _ in range(N_USERS):
        preferred = rng.choice(CATEGORIES)
        users.append(UserContext(
            lat=RABAT_LAT + rng.uniform(-0.03, 0.03),
            lon=RABAT_LON + rng.uniform(-0.03, 0.03),
            budget_max=rng.uniform(80, 300),
            category_history=[preferred] * 4 + [rng.choice(CATEGORIES)],
        ))
    return users


def ground_truth_relevant(place: PlaceCandidate, user: UserContext) -> bool:
    preferred_category = user.category_history[0] if user.category_history else None
    within_budget = place.budget <= user.budget_max
    distance_km = haversine_km(user.lat, user.lon, place.lat, place.lon)
    return place.category == preferred_category and within_budget and distance_km <= 6.0


def precision_at_k(ranked_ids: list[int], places_by_id: dict, user: UserContext, k: int) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    relevant = sum(1 for pid in top if ground_truth_relevant(places_by_id[pid], user))
    return relevant / len(top)


def main():
    places = make_places(seed=1)
    users = make_users(seed=2)
    places_by_id = {p.id: p for p in places}

    baseline_scores = []
    content_scores = []

    for user in users:
        baseline_ranked = sorted(places, key=lambda p: baseline_linear_score(p, user), reverse=True)
        baseline_ids = [p.id for p in baseline_ranked]
        baseline_scores.append(precision_at_k(baseline_ids, places_by_id, user, TOP_K))

        content_ranked = recommend(places, user, TOP_K)
        content_ids = [item.place_id for item in content_ranked]
        content_scores.append(precision_at_k(content_ids, places_by_id, user, TOP_K))

    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    content_avg = sum(content_scores) / len(content_scores)
    improvement = (content_avg - baseline_avg) / baseline_avg * 100 if baseline_avg > 0 else float("inf")

    print(f"Évaluation synthétique : {N_USERS} utilisateurs, {N_PLACES} lieux, Précision@{TOP_K}\n")
    print(f"{'Méthode':35s} | Précision@{TOP_K}")
    print("-" * 55)
    print(f"{'Ancienne formule (Java, distance+budget)':35s} | {baseline_avg:.3f}")
    print(f"{'Nouveau recommandeur (cosine, contenu)':35s} | {content_avg:.3f}")
    print(f"\nAmélioration relative : {improvement:+.1f}%")


if __name__ == "__main__":
    main()
