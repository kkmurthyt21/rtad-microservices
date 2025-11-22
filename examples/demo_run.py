"""Example script to run the full anomaly detection pipeline."""

from src.simulate_pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(n_samples=500)
