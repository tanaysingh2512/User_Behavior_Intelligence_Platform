from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import settings

_ENGINE: Engine | None = None


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(settings.database_url, pool_pre_ping=True)
    return _ENGINE


def execute(sql: str, params: dict[str, Any] | None = None) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


def read_df(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(text(sql), engine, params=params)


def write_df(
    table: str,
    df: pd.DataFrame,
    if_exists: str = "append",
    index: bool = False,
    chunksize: int = 10_000,
) -> None:
    engine = get_engine()
    df.to_sql(
        table,
        engine,
        if_exists=if_exists,
        index=index,
        chunksize=chunksize,
        method="multi",
    )


def upsert_predictions(
    rows: Iterable[dict[str, Any]],
) -> None:
    if not rows:
        return

    sql = """
        INSERT INTO predictions (customer_id, churn_proba, churn_label, model_version)
        VALUES (:customer_id, :churn_proba, :churn_label, :model_version)
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql), list(rows))
