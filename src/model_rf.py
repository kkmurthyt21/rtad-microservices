from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_rf(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest classifier for anomaly detection.

    y is assumed to be binary: 1 for anomaly, 0 for normal.
    """
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


def score_rf(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Return anomaly probabilities (class 1) from RF."""
    probs = model.predict_proba(X)
    # assume class 1 is anomaly
    return probs[:, 1]
