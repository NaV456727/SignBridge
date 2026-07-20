import joblib
from matplotlib.pylab import indices
import pandas as pd

from src.utils.landmark_normalizer import normalize_features
from src.config import MODEL_FILE


class GestureClassifier:

    def __init__(self):

        self.model = joblib.load(MODEL_FILE)

        self.columns = []

        for i in range(21):
            self.columns.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}",
            ])

    def predict(self, hand):

        features = []

        for landmark in hand.landmarks:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z,
            ])

        features = normalize_features(features)

        X = pd.DataFrame(
            [features],
            columns=self.columns,
)

        probabilities = self.model.predict_proba(X)[0]

        prediction_index = probabilities.argmax()

        prediction = self.model.classes_[prediction_index]

        confidence = probabilities[prediction_index] * 100

        indices = probabilities.argsort()[::-1][:3]

        top_predictions = [
            (
                self.model.classes_[i],
                probabilities[i] * 100,
            )
            for i in indices
        ]

        return prediction, confidence, top_predictions