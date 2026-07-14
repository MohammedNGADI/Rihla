"""
Classification d'intentions pour le chatbot (find_place, find_restaurant,
create_circuit, ask_weather, general) : pipeline TF-IDF + régression logistique,
entraîné sur data/intents_train.csv (voir train_intent_model.py pour les métriques
d'évaluation : accuracy, F1, matrice de confusion).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "intents_train.csv"
MODEL_PATH = BASE_DIR / "models" / "intent_model.joblib"

CONFIDENCE_THRESHOLD = 0.35  


def load_training_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear", C=5.0)),
    ])


class IntentClassifier:
    def __init__(self):
        self._pipeline: Pipeline | None = None

    def load(self):
        if MODEL_PATH.exists():
            self._pipeline = joblib.load(MODEL_PATH)
        else:
            # Pas de modèle entraîné trouvé : on entraîne à la volée sur le jeu de données
            # complet pour que le service reste utilisable (voir train_intent_model.py
            # pour l'entraînement "officiel" avec évaluation train/test).
            df = load_training_data()
            self._pipeline = build_pipeline()
            self._pipeline.fit(df["text"], df["intent"])
        return self

    def predict(self, text: str) -> tuple[str, float]:
        if self._pipeline is None:
            self.load()
        proba = self._pipeline.predict_proba([text])[0]
        classes = self._pipeline.classes_
        best_idx = proba.argmax()
        intent, confidence = classes[best_idx], float(proba[best_idx])
        if confidence < CONFIDENCE_THRESHOLD:
            return "general", confidence
        return intent, confidence


intent_classifier = IntentClassifier()
