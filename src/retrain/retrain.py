from __future__ import annotations

from src.config import settings
from src.train.explain import main as explain_main
from src.train.train_model import main as train_main
from src.utils.db import read_df


def should_retrain() -> bool:
    drift = read_df(
        """
        SELECT feature_name, psi, ks, computed_at
        FROM drift_metrics
        ORDER BY computed_at DESC
        LIMIT 50
        """
    )

    if drift.empty:
        return False

    psi_trigger = drift["psi"].max() > settings.drift_psi_threshold
    ks_trigger = drift["ks"].max() > settings.drift_ks_threshold
    return bool(psi_trigger or ks_trigger)


def main() -> None:
    if should_retrain():
        print("Drift threshold exceeded. Retraining model...")
        train_main()
        explain_main()
    else:
        print("No retraining needed.")


if __name__ == "__main__":
    main()
