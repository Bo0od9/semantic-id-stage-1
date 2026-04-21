from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .io import load_metrics


def _flatten_metrics(nested: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in nested.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, path))
        elif isinstance(value, (int, float)):
            flat[path] = float(value)
    return flat


def collect_runs(run_dirs: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        metrics_path = Path(run_dir) / "metrics.json"
        if not metrics_path.exists():
            continue
        data = load_metrics(metrics_path)
        row = {"run": Path(run_dir).name}
        row.update(_flatten_metrics(data))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_seeds(
    run_dirs: Iterable[Path],
    model_name: str,
) -> pd.DataFrame:
    df = collect_runs(run_dirs)
    if df.empty:
        return df
    metric_cols = [c for c in df.columns if c != "run"]
    summary = {"model": model_name, "n_runs": len(df)}
    for col in metric_cols:
        summary[f"{col}_mean"] = float(df[col].mean())
        std_val = df[col].std(ddof=1)
        summary[f"{col}_std"] = 0.0 if pd.isna(std_val) else float(std_val)
    return pd.DataFrame([summary])


def write_results_csv(
    path: str | Path,
    frames: Iterable[pd.DataFrame],
) -> None:
    frames_list = [f for f in frames if not f.empty]
    if not frames_list:
        raise ValueError("No non-empty frames to write")
    merged = pd.concat(frames_list, ignore_index=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
