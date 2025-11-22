# Real-Time Anomaly Detection in Microservices

This repository contains a reproducible implementation of the framework described in the paper:

**"Real-Time Anomaly Detection in Microservices" – Krishna Kandi**

The goal of this project is to provide a lightweight, interpretable, and production-oriented anomaly
detection pipeline for microservice environments using an ensemble of:

- Random Forest (RF)
- Autoencoder (AE)
- Isolation Forest (IF)

The framework supports:
- Synthetic metrics collection (latency, CPU, queue depth, throughput)
- Z-score based preprocessing
- Training and scoring of RF, AE, IF
- Weighted ensemble scoring
- Adaptive μ + 3σ thresholding for anomaly flags

## Repository Structure

```text
RealTime-Anomaly-Detection-Microservices/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── collector.py
│   ├── preprocessor.py
│   ├── model_rf.py
│   ├── model_if.py
│   ├── model_ae.py
│   ├── ensemble.py
│   ├── threshold.py
│   └── simulate_pipeline.py
├── data/
│   └── README.md
└── examples/
    └── demo_run.py
```

## Quickstart

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main simulation pipeline:

   ```bash
   python -m src.simulate_pipeline
   ```

You should see output indicating how many anomalies were detected by the ensemble.

## Notes

- This implementation uses purely synthetic data to mimic microservice metrics.
- You can replace `collect_metrics()` in `collector.py` with real Prometheus / OpenTelemetry data.
- The code is designed as a clear, educational baseline rather than a full production system.
