from __future__ import annotations

import pandas as pd

from src.config import settings
from src.utils.db import execute, write_df


def main() -> None:
    processed_path = settings.processed_dir / "transactions.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset at {processed_path}. Run clean_retail.py first."
        )

    df = pd.read_parquet(processed_path)
    execute("TRUNCATE TABLE transactions")
    write_df("transactions", df, if_exists="append", index=False)
    print("Loaded transactions into PostgreSQL")


if __name__ == "__main__":
    main()
