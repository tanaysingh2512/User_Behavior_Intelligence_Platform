from __future__ import annotations

import pandas as pd

from src.config import settings
from src.utils.db import execute, write_df


def main() -> None:
    dataset_path = settings.processed_dir / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing dataset at {dataset_path}. Run build_features.py first."
        )

    df = pd.read_parquet(dataset_path)

    execute("TRUNCATE TABLE features")
    write_df("features", df, if_exists="append", index=False)
    print("Loaded features into PostgreSQL")


if __name__ == "__main__":
    main()
