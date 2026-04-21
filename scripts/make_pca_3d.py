"""One-off visualisation: interactive 3D PCA of item embeddings.

Reads the cached ``embeddings.parquet`` (already on disk after stage 7) and
``data/processed/train.parquet``, fits PCA(n_components=3) on
``normalized_embed`` of all train items, then saves a standalone Plotly HTML
scatter with rich hover tooltips (item_id, popularity, top-5 cosine neighbours
within the subsample).

Output: ``reports/figures/fig_09_embeddings_pca_3d.html``

This script is NOT a numbered pipeline stage — it produces a visualisation
artifact only and does NOT modify ``filter_stats.json`` or
``splits_metadata.json``. It is safe to re-run at any time.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from huggingface_hub import hf_hub_download
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    EMBEDDING_PCA_HTML_SAMPLE_SIZE,
    EMBEDDING_PCA_HTML_TOP_K,
    EMBEDDINGS_PARQUET_REL,
    EMBEDDINGS_PCA_3D_HTML_PATH,
    FIGURES_DIR,
    RAW_CACHE_DIR,
    SPLITS_METADATA_PATH,
    TRAIN_PARQUET_PATH,
    YAMBDA_REPO,
)
from utils.io import load_json, setup_logging  # noqa: E402
from utils.parquet import validate_listens_schema  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders (same pattern as stage 07)
# ---------------------------------------------------------------------------


def download_parquet(repo_filename: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ensuring %s is cached (repo=%s)", repo_filename, YAMBDA_REPO)
    local = hf_hub_download(
        repo_id=YAMBDA_REPO,
        filename=repo_filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(local)


def load_seed(meta_path: Path) -> int:
    payload = load_json(meta_path)
    return int(payload["random_seed"])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_normalized_embeddings(
    emb_path: Path, train_item_ids: pl.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Return (item_ids_sorted, normalized_arr) for train items only."""
    wanted = pl.DataFrame({"item_id": train_item_ids}).unique().sort("item_id")
    logger.info("reading embeddings parquet (normalized_embed only)...")
    emb_df = (
        pl.read_parquet(emb_path, columns=["item_id", "normalized_embed"])
        .join(wanted, on="item_id", how="inner")
        .sort("item_id")
    )
    logger.info("%d train items loaded", emb_df.height)
    item_ids = emb_df.get_column("item_id").to_numpy().astype(np.uint32, copy=False)
    normalized = np.asarray(
        emb_df.get_column("normalized_embed").to_list(), dtype=np.float32
    )
    return item_ids, normalized


def compute_popularity(train: pl.DataFrame) -> np.ndarray:
    """Return popularity aligned with sorted item_ids from the embedding join."""
    agg = train.group_by("item_id").agg(pl.len().alias("count")).sort("item_id")
    return agg.get_column("count").to_numpy().astype(np.int64)


def popularity_lookup(item_ids: np.ndarray, pop_map: dict[int, int]) -> np.ndarray:
    out = np.empty(item_ids.shape[0], dtype=np.int64)
    for i, iid in enumerate(item_ids.tolist()):
        out[i] = pop_map[iid]
    return out


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _hover_text(
    item_id: int,
    popularity: int,
    neighbours: list[tuple[int, float, int]],
) -> str:
    lines = [
        f"item_id: {item_id}",
        f"popularity: {popularity}",
        f"top-{len(neighbours)} neighbours (cosine, within subsample):",
    ]
    for rank, (nb_id, sim, nb_pop) in enumerate(neighbours, 1):
        lines.append(f"  {rank}. id={nb_id}  cos={sim:.3f}  pop={nb_pop}")
    return "<br>".join(lines)


def make_3d_html(
    coords: np.ndarray,
    item_ids: np.ndarray,
    popularity: np.ndarray,
    normalized_sample: np.ndarray,
    top_k: int,
    explained: list[float],
    html_path: Path,
) -> None:
    n = coords.shape[0]
    logger.info("computing intra-sample NN for %d items (top-%d)...", n, top_k)
    nn = NearestNeighbors(n_neighbors=top_k + 1, metric="cosine", algorithm="brute")
    nn.fit(normalized_sample)
    distances, indices = nn.kneighbors(normalized_sample)
    nb_idx = indices[:, 1 : top_k + 1]
    nb_sim = 1.0 - distances[:, 1 : top_k + 1]

    hover_texts: list[str] = []
    for i in range(n):
        nbs = [
            (int(item_ids[nb_idx[i, r]]), float(nb_sim[i, r]), int(popularity[nb_idx[i, r]]))
            for r in range(top_k)
        ]
        hover_texts.append(_hover_text(int(item_ids[i]), int(popularity[i]), nbs))

    color = np.log1p(popularity.astype(np.float64))
    cumulative = sum(explained)

    fig = go.Figure(
        data=go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=2.5,
                color=color,
                colorscale="Viridis",
                opacity=0.7,
                colorbar=dict(title="log(1+pop)", thickness=14),
                showscale=True,
            ),
            text=hover_texts,
            hoverinfo="text",
        )
    )
    var_str = (
        f"PC1={explained[0]:.2%}, PC2={explained[1]:.2%}, "
        f"PC3={explained[2]:.2%} — cumulative {cumulative:.2%}"
    )
    fig.update_layout(
        title=(
            f"Item embeddings — 3D PCA (normalized_embed, "
            f"{n:,}-item subsample)<br>"
            f"<sup>{var_str}</sup>"
        ),
        scene=dict(
            xaxis_title=f"PC1 ({explained[0]:.2%})",
            yaxis_title=f"PC2 ({explained[1]:.2%})",
            zaxis_title=f"PC3 ({explained[2]:.2%})",
            bgcolor="rgb(248,248,252)",
        ),
        template="plotly_white",
        width=1050,
        height=800,
        margin=dict(l=0, r=0, b=0, t=80),
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    logger.info("wrote 3D PCA HTML → %s", html_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", type=Path, default=TRAIN_PARQUET_PATH)
    p.add_argument("--splits-metadata", type=Path, default=SPLITS_METADATA_PATH)
    p.add_argument("--raw-cache", type=Path, default=RAW_CACHE_DIR)
    p.add_argument("--output", type=Path, default=EMBEDDINGS_PCA_3D_HTML_PATH)
    p.add_argument(
        "--sample-size",
        type=int,
        default=EMBEDDING_PCA_HTML_SAMPLE_SIZE,
        help="Number of items in the HTML subsample (default: %(default)s)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=EMBEDDING_PCA_HTML_TOP_K,
        help="Neighbours per hover tooltip (default: %(default)s)",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    seed = load_seed(args.splits_metadata)
    logger.info("seed=%d", seed)

    logger.info("reading train.parquet...")
    train = pl.read_parquet(args.train_parquet)
    validate_listens_schema(train, origin="train.parquet")
    train_item_ids = train.get_column("item_id").unique()
    n_items = int(train_item_ids.len())
    logger.info("train: %d unique items", n_items)

    emb_path = download_parquet(EMBEDDINGS_PARQUET_REL, args.raw_cache)
    item_ids, normalized = load_normalized_embeddings(emb_path, train_item_ids)

    # Popularity aligned with item_ids (sorted)
    agg = (
        train.group_by("item_id")
        .agg(pl.len().alias("count"))
        .sort("item_id")
    )
    pop_map = {int(i): int(c) for i, c in zip(
        agg.get_column("item_id").to_list(),
        agg.get_column("count").to_list(),
    )}
    popularity = popularity_lookup(item_ids, pop_map)

    # 3D PCA
    logger.info("fitting PCA(n_components=3) on %d vectors...", n_items)
    pca = PCA(n_components=3, random_state=0)
    coords = pca.fit_transform(normalized).astype(np.float32)
    explained = [float(v) for v in pca.explained_variance_ratio_]
    logger.info(
        "explained variance: %.4f + %.4f + %.4f = %.4f",
        *explained, sum(explained),
    )

    # Subsample
    sample_size = min(args.sample_size, n_items)
    rng = np.random.default_rng(seed)
    weights = np.sqrt(popularity.astype(np.float64) + 1.0)
    weights /= weights.sum()
    idx = np.sort(rng.choice(n_items, size=sample_size, replace=False, p=weights))
    logger.info("HTML subsample: %d items", idx.shape[0])

    make_3d_html(
        coords=coords[idx],
        item_ids=item_ids[idx],
        popularity=popularity[idx],
        normalized_sample=normalized[idx],
        top_k=args.top_k,
        explained=explained,
        html_path=args.output,
    )
    logger.info("done. Open in browser: %s", args.output)


if __name__ == "__main__":
    main()
