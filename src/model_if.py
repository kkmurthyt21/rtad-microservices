import numpy as np
from sklearn.ensemble import IsolationForest


def train_if(X: np.ndarray) -> IsolationForest:
    """Train Isolation Forest for unsupervised anomaly detection."""
    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42,
    )
    model.fit(X)
    return model


def score_if(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Return anomaly scores from Isolation Forest.

    Higher scores should indicate more anomalous points, so we invert
    the decision function.
    """
    raw = model.decision_function(X)
    return -raw
