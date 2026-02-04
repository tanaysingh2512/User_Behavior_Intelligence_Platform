from __future__ import annotations

import json
from datetime import datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import settings


def main() -> None:
    dataset_path = settings.processed_dir / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {dataset_path}. Run build_features.py first."
        )

    with settings.feature_spec_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    df = pd.read_parquet(dataset_path)
    features = spec["features"]
    label = spec["label"]

    X = df[features]
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= settings.prediction_threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "brier": brier_score_loss(y_test, proba),
    }

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("ubip-churn")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "label_window_days": settings.label_window_days,
                "feature_window_days": settings.feature_window_days,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.model_path)

    baseline_stats: dict[str, dict[str, list[float] | float]] = {}
    for feature in features:
        values = X_train[feature].values.astype(float)
        edges = np.quantile(values, np.linspace(0, 1, 11)).tolist()
        edges[0] -= 1e-6
        edges[-1] += 1e-6
        expected_counts, _ = np.histogram(values, bins=edges)
        expected_pct = (expected_counts / max(expected_counts.sum(), 1)).tolist()

        if len(values) <= 1000:
            sample_values = values
        else:
            sample_values = np.random.choice(values, size=1000, replace=False)

        baseline_stats[feature] = {
            "edges": edges,
            "expected_pct": expected_pct,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "sample_values": [float(v) for v in sample_values],
        }

    with settings.baseline_stats_path.open("w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, indent=2)

    model_card = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
        "features": features,
        "label_window_days": settings.label_window_days,
        "feature_window_days": settings.feature_window_days,
    }
    with (settings.models_dir / "model_card.json").open("w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    print(f"Saved model to {settings.model_path}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
