"""Train Random baseline (seeds 42/43/44) and save metrics + per-user primary scores.

Hydra entry point. Usage:

    uv run python scripts/train_random.py -m trainer.seed=42,43,44
    uv run python scripts/train_random.py trainer.seed=42 data.split_set=subsample_1pct
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hydra  # noqa: E402
import polars as pl  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.baselines.random_rec import rank_random  # noqa: E402
from src.data import (  # noqa: E402
    SAVED_DIR,
    build_targets,
    load_item_id_map,
    load_user_ids,
)
from src.metrics.metrics import calc_metrics, per_user_primary  # noqa: E402
from src.utils.io import save_metrics  # noqa: E402
from src.utils.seed import resolve_device, set_seed  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="random")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.trainer.seed)
    device = resolve_device(cfg.trainer.device)
    print(f"device: {device}, seed: {cfg.trainer.seed}, split_set: {cfg.data.split_set}")

    k_values = list(cfg.data.k_values)
    max_k = max(k_values)
    metric_names = [f"{m}@{k}" for m in cfg.data.metrics for k in k_values]

    item_map = load_item_id_map()

    run_dir = SAVED_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, dict[int, float]]] = {}
    per_user_rows: list[dict[str, float | int | str]] = []

    for split in ("val", "test"):
        user_ids = load_user_ids(cfg.data.split_set, split, device=device)
        targets = build_targets(cfg.data.split_set, split, item_map, device=device)
        ranked = rank_random(
            user_ids=user_ids,
            n_items=item_map.n_items,
            k=max_k,
            seed=int(cfg.trainer.seed),
        )

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
