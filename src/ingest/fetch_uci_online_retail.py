from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import settings


def main() -> None:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=502)  # Online Retail II
    data = getattr(dataset.data, "original", None)
    if data is None:
        data = dataset.data.features

    df = pd.DataFrame(data)

    raw_path = settings.raw_dir / "online_retail_ii.parquet"
    df.to_parquet(raw_path, index=False)

    meta_path = settings.raw_dir / "online_retail_ii_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "id": dataset.metadata.get("id"),
                "name": dataset.metadata.get("name"),
                "url": dataset.metadata.get("url"),
                "num_instances": dataset.metadata.get("num_instances"),
                "num_features": dataset.metadata.get("num_features"),
            },
            f,
            indent=2,
        )

    print(f"Saved raw dataset to {raw_path}")


if __name__ == "__main__":
    main()
