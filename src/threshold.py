import numpy as np


def adaptive_threshold(scores: np.ndarray, k: float = 3.0) -> float:
    """Return μ + kσ adaptive threshold for anomaly scores."""
    mu = np.mean(scores)
    sigma = np.std(scores)
    return mu + k * sigma
