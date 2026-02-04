from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import settings


def main() -> None:
    dataset_path = settings.processed_dir / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {dataset_path}. Run build_features.py first."
        )

    with settings.feature_spec_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    features = spec["features"]

    df = pd.read_parquet(dataset_path)
    sample = df[features].sample(n=min(1000, len(df)), random_state=42)

    model = joblib.load(settings.model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    mean_abs = np.abs(shap_values).mean(axis=0)
    global_importance = (
        pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .to_dict(orient="records")
    )

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[-1]

    payload = {
        "expected_value": float(expected_value),
        "global_importance": global_importance,
    }

    output_path = settings.models_dir / "shap_global.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved SHAP summary to {output_path}")


if __name__ == "__main__":
    main()
