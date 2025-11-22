import pandas as pd
import numpy as np


def collect_metrics(n_samples: int = 500) -> pd.DataFrame:
    """Generate synthetic microservice metrics.

    Columns:
    - latency_p99: 99th percentile latency in ms
    - cpu: CPU utilization in percent
    - queue_depth: pending messages in queue
    - throughput: requests per second
    """
    rng = np.random.default_rng()

    data = {
        "latency_p99": rng.normal(120.0, 25.0, n_samples),
        "cpu": rng.uniform(20.0, 95.0, n_samples),
        "queue_depth": rng.integers(0, 120, n_samples),
        "throughput": rng.uniform(100.0, 400.0, n_samples),
    }
    return pd.DataFrame(data)
