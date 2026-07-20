import os
import sys
import time
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.utils.landmark_normalizer import normalize_dataframe


# ==========================================================
# Configuration
# ==========================================================

if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
else:
    DATASET_PATH = "dataset/asl_dataset.csv"

MODEL_PATH = "models/asl_sign_model.pkl"


# ==========================================================


def main():

    if not os.path.exists(DATASET_PATH):
        print("Dataset not found!")
        return

    print("\nLoading dataset...")

    data = pd.read_csv(DATASET_PATH)

    if data.empty:
        print("Dataset is empty!")
        return

    # --------------------------------------------------

    X = data.drop(columns=["label"])
    y = data["label"]

    print("\nNormalizing landmarks...")

    X = normalize_dataframe(X)

    print("Normalization complete.\n")

    # --------------------------------------------------

    print("========== DATASET SUMMARY ==========\n")

    print(y.value_counts())

    print("\n=====================================\n")

    print(f"Total Samples : {len(data)}")
    print(f"Classes       : {len(y.unique())}")

    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    # --------------------------------------------------

    print("\nTraining Random Forest...\n")

    start = time.time()

    model = RandomForestClassifier(

        n_estimators=200,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features="sqrt",

        bootstrap=True,

        oob_score=True,

        random_state=42,

        n_jobs=-1,

        verbose=1,
    )

    model.fit(X_train, y_train)

    end = time.time()

    print(f"\nTraining Time : {end-start:.2f} seconds")

    print(f"Out-of-Bag Accuracy : {model.oob_score_ * 100:.2f}%")

    # --------------------------------------------------

    print("\nEvaluating...\n")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print(f"Validation Accuracy : {accuracy * 100:.2f}%\n")

    print(classification_report(
        y_test,
        predictions,
    ))

    # --------------------------------------------------

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()