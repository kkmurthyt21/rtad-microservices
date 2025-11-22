import numpy as np
from typing import Tuple


def ensemble_score(
    rf_scores: np.ndarray,
    ae_scores: np.ndarray,
    if_scores: np.ndarray,
    weights: Tuple[float, float, float] = (0.33, 0.33, 0.34),
) -> np.ndarray:
    """Compute weighted ensemble anomaly score."""
    w1, w2, w3 = weights
    return w1 * rf_scores + w2 * ae_scores + w3 * if_scores
