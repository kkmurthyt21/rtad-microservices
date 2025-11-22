from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_ae(input_dim: int) -> tf.keras.Model:
    """Build a simple fully connected autoencoder."""
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(input_dim, activation=None),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_ae(model: tf.keras.Model, X: np.ndarray, epochs: int = 20) -> tf.keras.Model:
    """Train the autoencoder to reconstruct X."""
    model.fit(X, X, epochs=epochs, batch_size=32, verbose=0)
    return model


def score_ae(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """Return reconstruction error as anomaly score."""
    recon = model.predict(X, verbose=0)
    errors = np.mean((X - recon) ** 2, axis=1)
    return errors
