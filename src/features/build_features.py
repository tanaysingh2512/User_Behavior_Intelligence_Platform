from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    dataset_end = df["invoice_date"].max()
    cutoff = dataset_end - pd.Timedelta(days=settings.label_window_days)
    window_start = cutoff - pd.Timedelta(days=settings.feature_window_days)

    history = df[df["invoice_date"] <= cutoff]
    label_period = df[(df["invoice_date"] > cutoff) & (df["invoice_date"] <= dataset_end)]
    window_df = history[history["invoice_date"] > window_start].copy()

    window_df["invoice_day"] = window_df["invoice_date"].dt.date

    agg = window_df.groupby("customer_id").agg(
        last_invoice_date=("invoice_date", "max"),
        first_invoice_date=("invoice_date", "min"),
        invoices_count=("invoice_no", "nunique"),
        total_items=("quantity", "sum"),
        total_spend=("total_price", "sum"),
        unique_items=("stock_code", "nunique"),
        avg_unit_price=("unit_price", "mean"),
        active_days=("invoice_day", "nunique"),
    )

    agg["recency_days"] = (cutoff - agg["last_invoice_date"]).dt.days
    agg["tenure_days"] = (cutoff - agg["first_invoice_date"]).dt.days
    agg["avg_basket_size"] = agg["total_items"] / agg["invoices_count"].replace(0, np.nan)
    agg["spend_per_invoice"] = agg["total_spend"] / agg["invoices_count"].replace(0, np.nan)
    agg["frequency_per_day"] = agg["invoices_count"] / agg["tenure_days"].replace(0, np.nan)

    agg = agg.replace([np.inf, -np.inf], np.nan).fillna(0)

    label_users = set(label_period["customer_id"].unique())
    agg["churn_label"] = agg.index.to_series().apply(lambda cid: 0 if cid in label_users else 1)

    agg["cutoff_date"] = cutoff
    agg = agg.reset_index()

    return agg


def main() -> None:
    processed_path = settings.processed_dir / "transactions.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset at {processed_path}. Run clean_retail.py first."
        )

    df = pd.read_parquet(processed_path)
    features = build_features(df)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = settings.processed_dir / "dataset.parquet"
    features.to_parquet(dataset_path, index=False)

    feature_spec = {
        "features": [
            "invoices_count",
            "total_items",
            "total_spend",
            "unique_items",
            "avg_unit_price",
            "active_days",
            "recency_days",
            "tenure_days",
            "avg_basket_size",
            "spend_per_invoice",
            "frequency_per_day",
        ],
        "label": "churn_label",
        "cutoff_date": str(features["cutoff_date"].iloc[0]),
        "label_window_days": settings.label_window_days,
        "feature_window_days": settings.feature_window_days,
    }

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    with settings.feature_spec_path.open("w", encoding="utf-8") as f:
        json.dump(feature_spec, f, indent=2)

    print(f"Saved dataset to {dataset_path}")
    print(f"Saved feature spec to {settings.feature_spec_path}")


if __name__ == "__main__":
    main()
