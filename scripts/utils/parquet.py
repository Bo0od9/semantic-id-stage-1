"""Shared parquet helpers for the data-preparation pipeline.

Owns the canonical listens schema and the two operations every stage that
writes ``listens``-shaped parquet needs: schema/null validation and atomic
write. Kept in one place so stage 5, stage 6 and any later stage that
materialises user/item slices stay byte-identical on dtype and layout.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

__all__ = [
    "REQUIRED_SCHEMA",
    "REQUIRED_COLUMNS",
    "validate_listens_schema",
    "atomic_write_parquet",
    "counts",
]

REQUIRED_SCHEMA: dict[str, pl.DataType] = {
    "uid": pl.UInt32,
    "timestamp": pl.UInt32,
    "item_id": pl.UInt32,
    "is_organic": pl.UInt8,
    "played_ratio_pct": pl.UInt16,
    "track_length_seconds": pl.UInt32,
}
REQUIRED_COLUMNS: tuple[str, ...] = tuple(REQUIRED_SCHEMA.keys())


def validate_listens_schema(df: pl.DataFrame, origin: str) -> None:
    """Assert columns, dtypes and null-freeness of a listens-shaped frame.

    ``origin`` is a human-readable label (e.g. file path or split name) used
    in error messages so multi-stage failures are easy to attribute.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{origin}: missing columns {missing}")
    bad_dtypes = {
        col: str(df.schema[col])
        for col, want in REQUIRED_SCHEMA.items()
        if df.schema[col] != want
    }
    if bad_dtypes:
        expected = {col: str(t) for col, t in REQUIRED_SCHEMA.items()}
        raise RuntimeError(
            f"{origin}: dtype regression {bad_dtypes} (expected {expected})"
        )
    null_row = df.null_count().row(0)
    if any(n != 0 for n in null_row):
        raise RuntimeError(
            f"{origin}: nulls per column {dict(zip(df.columns, null_row))}"
        )


def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` via a tmp file + rename, zstd-compressed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)


def counts(df: pl.DataFrame) -> dict[str, int]:
    """Standard (interactions, users, items) triple used by every stage."""
    return {
        "num_interactions": int(df.height),
        "num_users": int(df.get_column("uid").n_unique()),
        "num_items": int(df.get_column("item_id").n_unique()),
    }
