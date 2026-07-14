"""
Entraîne le classifieur d'intentions du chatbot et affiche les métriques
d'évaluation (accuracy, précision/rappel/F1 par classe, matrice de confusion).

Ces chiffres sont ceux à reporter dans le rapport comme indicateur de
performance du "modèle IA" du chatbot.

Usage: python train_intent_model.py
"""
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from app.intent_classifier import MODEL_PATH, build_pipeline, load_training_data


def main():
    df = load_training_data()
    print(f"Jeu de données : {len(df)} exemples, {df['intent'].nunique()} intentions "
          f"({', '.join(sorted(df['intent'].unique()))})\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"], test_size=0.25, random_state=42, stratify=df["intent"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=== Évaluation sur le jeu de test (25%) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(df["intent"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Matrice de confusion (lignes = réel, colonnes = prédit) :")
    print(f"{'':16s}" + "".join(f"{label[:10]:12s}" for label in labels))
    for label, row in zip(labels, cm):
        print(f"{label[:16]:16s}" + "".join(f"{value:<12d}" for value in row))

    final_pipeline = build_pipeline()
    final_pipeline.fit(df["text"], df["intent"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"\nModèle final (entraîné sur 100% des données) sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
