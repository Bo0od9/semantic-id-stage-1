"""Train MostPop baseline and save metrics + per-user primary scores.

Hydra entry point. Usage:

    uv run python scripts/train_mostpop.py                        # full split
    uv run python scripts/train_mostpop.py data.split_set=subsample_1pct
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hydra  # noqa: E402
import polars as pl  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.baselines.popularity import rank_mostpop  # noqa: E402
from src.data import (  # noqa: E402
    ITEM_EMBEDDINGS_PARQUET,
    SAVED_DIR,
    build_targets,
    load_item_id_map,
    load_popularity,
    load_user_ids,
)
from src.metrics.metrics import calc_metrics, per_user_primary  # noqa: E402
from src.utils.io import save_metrics  # noqa: E402
from src.utils.seed import resolve_device, set_seed  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="mostpop")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.trainer.seed)
    device = resolve_device(cfg.trainer.device)
    print(f"device: {device}, split_set: {cfg.data.split_set}")

    k_values = list(cfg.data.k_values)
    max_k = max(k_values)
    metric_names = [f"{m}@{k}" for m in cfg.data.metrics for k in k_values]

    item_map = load_item_id_map()
    popularity = load_popularity(ITEM_EMBEDDINGS_PARQUET, item_map, device=device)

    run_dir = SAVED_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, dict[int, float]]] = {}
    per_user_rows: list[dict[str, float | int | str]] = []

    for split in ("val", "test"):
        user_ids = load_user_ids(cfg.data.split_set, split, device=device)
        targets = build_targets(cfg.data.split_set, split, item_map, device=device)
        ranked = rank_mostpop(popularity, user_ids, k=max_k)

        results[split] = calc_metrics(ranked, targets, metric_names, show_progress=False)

        pu = per_user_primary(ranked, targets, k=int(cfg.data.primary_k))
        uids = pu["user_ids"].cpu().numpy()
        recall = pu["recall"].cpu().numpy()
        ndcg = pu["ndcg"].cpu().numpy()
        for uid, r, n in zip(uids.tolist(), recall.tolist(), ndcg.tolist(), strict=True):
            per_user_rows.append({"split": split, "uid": int(uid), "recall": float(r), "ndcg": float(n)})

    save_metrics(run_dir / "metrics.json", results)
    pl.DataFrame(per_user_rows).write_parquet(run_dir / "per_user.parquet")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    print(f"wrote {run_dir}/metrics.json and per_user.parquet")


if __name__ == "__main__":
    main()
