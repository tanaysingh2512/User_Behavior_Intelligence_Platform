from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    models_dir: Path = project_root / "models"
    mlruns_dir: Path = project_root / "mlruns"

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/ubip",
    )
    mlflow_tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI", f"file://{mlruns_dir}"
    )
    model_path: Path = Path(
        os.getenv("MODEL_PATH", str(models_dir / "churn_model.pkl"))
    )
    feature_spec_path: Path = Path(
        os.getenv("FEATURE_SPEC_PATH", str(models_dir / "feature_spec.json"))
    )
    baseline_stats_path: Path = Path(
        os.getenv("BASELINE_STATS_PATH", str(models_dir / "baseline_stats.json"))
    )

    label_window_days: int = int(os.getenv("LABEL_WINDOW_DAYS", "90"))
    feature_window_days: int = int(os.getenv("FEATURE_WINDOW_DAYS", "180"))

    drift_psi_threshold: float = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))
    drift_ks_threshold: float = float(os.getenv("DRIFT_KS_THRESHOLD", "0.1"))

    prediction_threshold: float = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

    pushgateway_url: str = os.getenv(
        "PUSHGATEWAY_URL", "http://localhost:9091"
    )


settings = Settings()
