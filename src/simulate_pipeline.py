import numpy as np

from .collector import collect_metrics
from .preprocessor import preprocess
from .model_rf import train_rf, score_rf
from .model_if import train_if, score_if
from .model_ae import build_ae, train_ae, score_ae
from .ensemble import ensemble_score
from .threshold import adaptive_threshold


def run_pipeline(n_samples: int = 500) -> None:
    # 1. Collect synthetic metrics
    df = collect_metrics(n_samples=n_samples)

    # 2. Preprocess (z-score)
    X_df = preprocess(df)
    X = X_df.values

    # 3. Create synthetic binary labels (for RF) – 5% anomalies
    rng = np.random.default_rng(42)
    y = (rng.random(n_samples) < 0.05).astype(int)

    # 4. Train models
    rf = train_rf(X, y)
    iforest = train_if(X)
    ae = build_ae(X.shape[1])
    ae = train_ae(ae, X, epochs=20)

    # 5. Score models
    rf_s = score_rf(rf, X)
    if_s = score_if(iforest, X)
    ae_s = score_ae(ae, X)

    # 6. Ensemble score
    final_scores = ensemble_score(rf_s, ae_s, if_s)

    # 7. Adaptive threshold
    th = adaptive_threshold(final_scores, k=3.0)
    anomalies = final_scores > th

    total_anomalies = int(anomalies.sum())
    print(f"Total points: {len(final_scores)}")        
    print(f"Adaptive threshold: {th:.4f}")
    print(f"Detected anomalies: {total_anomalies}")


if __name__ == "__main__":
    run_pipeline()
