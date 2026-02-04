from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings


COLUMN_MAP = {
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}


def main() -> None:
    raw_path = settings.raw_dir / "online_retail_ii.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing raw dataset at {raw_path}. Run fetch_uci_online_retail.py first."
        )

    df = pd.read_parquet(raw_path)
    df = df.rename(columns=COLUMN_MAP)

    df = df.dropna(subset=["customer_id", "invoice_date"])  # remove anonymous rows
    df["customer_id"] = df["customer_id"].astype(int)

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df = df.dropna(subset=["invoice_date"])

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    df = df.dropna(subset=["quantity", "unit_price"])

    df["is_cancelled"] = df["invoice_no"].astype(str).str.startswith("C")
    df = df[df["quantity"] > 0]
    df = df[df["unit_price"] > 0]

    df["total_price"] = df["quantity"] * df["unit_price"]

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = settings.processed_dir / "transactions.parquet"
    df.to_parquet(processed_path, index=False)

    print(f"Saved cleaned transactions to {processed_path}")


if __name__ == "__main__":
    main()
