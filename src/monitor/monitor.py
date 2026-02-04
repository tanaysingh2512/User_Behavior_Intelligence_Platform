from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from src.config import settings
from src.utils.db import read_df, write_df
from src.utils.metrics import ks_statistic, psi_from_bins


def main() -> None:
    if not settings.baseline_stats_path.exists():
        raise FileNotFoundError(
            f"Missing baseline stats at {settings.baseline_stats_path}. Train the model first."
        )

    with settings.baseline_stats_path.open("r", encoding="utf-8") as f:
        baseline_stats = json.load(f)

    df = read_df("SELECT * FROM features")
    if df.empty:
        raise RuntimeError("No features found in database. Load features first.")

    drift_rows = []
    registry = CollectorRegistry()
    psi_gauge = Gauge("ubip_psi", "Population Stability Index", ["feature"], registry=registry)
    ks_gauge = Gauge("ubip_ks", "KS statistic", ["feature"], registry=registry)

    for feature, stats in baseline_stats.items():
        actual = df[feature].values.astype(float)
        edges = np.array(stats["edges"], dtype=float)
        expected_pct = np.array(stats["expected_pct"], dtype=float)

        actual_counts, _ = np.histogram(actual, bins=edges)
        actual_pct = actual_counts / max(actual_counts.sum(), 1)

        psi_value = psi_from_bins(expected_pct, actual_pct)
        ks_value = ks_statistic(np.array(stats["sample_values"], dtype=float), actual)

        psi_gauge.labels(feature=feature).set(psi_value)
        ks_gauge.labels(feature=feature).set(ks_value)

        drift_rows.append(
            {
                "feature_name": feature,
                "psi": float(psi_value),
                "ks": float(ks_value),
                "computed_at": datetime.utcnow(),
            }
        )

    drift_df = pd.DataFrame(drift_rows)
    write_df("drift_metrics", drift_df, if_exists="append", index=False)

    push_to_gateway(settings.pushgateway_url, job="ubip_monitor", registry=registry)
    print("Drift metrics computed and pushed to Prometheus")


if __name__ == "__main__":
    main()
