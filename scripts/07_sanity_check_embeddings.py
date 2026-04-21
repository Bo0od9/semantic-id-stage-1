"""Stage 7 of the data preparation pipeline: item embeddings sanity check.

Reads ``data/processed/train.parquet`` (the authoritative item set after all
stage 1–6 filters) and the source embeddings parquet from the Hugging Face
Hub (``yandex/yambda``, file ``embeddings.parquet``, cached under
``data/raw_cache/``). Runs four checks corresponding to
``docs/instructions/dataset_prep.md`` section 7:

* 7.1 coverage and dimensionality — every train item must have an embedding
  (stage 2 guarantees this; we fail fast otherwise) and all vectors share
  the same dimension;
* 7.2 L2 norm distributions of **both** ``embed`` and ``normalized_embed``
  columns (histograms + percentiles);
* 7.3 nearest-neighbor sanity: for a seeded sample of anchor items, the
  top-K cosine neighbours from the full train-items set are recorded;
* 7.4 2D PCA visualisation of all train-item ``normalized_embed`` vectors,
  coloured by log popularity. Written both as a static PNG+PDF (all items)
  and as a standalone Plotly HTML with rich hover tooltips (a seeded
  ``sqrt(popularity)``-weighted subsample so the file stays openable in a
  browser).

Outputs:

* ``reports/09_embeddings_analysis.md``
* ``reports/figures/fig_09_embeddings_pca.{png,pdf,html}``
* ``reports/figures/fig_09_embedding_norms_{embed,normalized}.{png,pdf}``
* ``data/filter_stats.json``   — merged ``stage7`` section (siblings kept).
* ``data/splits_metadata.json`` — merged ``stage7_frozen_at_utc`` key.

No decision about ``embed`` vs ``normalized_embed`` is made here — both are
inspected and the pick is deferred to the training stage.

Fails fast on any invariant violation. No silent fallbacks.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
from huggingface_hub import hf_hub_download
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    EMBEDDING_DIM_EXPECTED,
    EMBEDDING_NN_SAMPLE_SIZE,
    EMBEDDING_NN_TOP_K,
    EMBEDDING_PCA_HTML_SAMPLE_SIZE,
    EMBEDDING_PCA_HTML_TOP_K,
    EMBEDDINGS_NORMS_EMBED_STEM,
    EMBEDDINGS_NORMS_NORMALIZED_STEM,
    EMBEDDINGS_PARQUET_REL,
    EMBEDDINGS_PCA_FIG_STEM,
    EMBEDDINGS_PCA_HTML_PATH,
    EMBEDDINGS_REPORT,
    FIGURES_DIR,
    FILTER_STATS_PATH,
    RAW_CACHE_DIR,
    SPLITS_METADATA_PATH,
    TRAIN_PARQUET_PATH,
    YAMBDA_REPO,
)
from utils.io import load_json, setup_logging, update_json_section  # noqa: E402
from utils.parquet import validate_listens_schema  # noqa: E402

logger = logging.getLogger(__name__)

VECTOR_COLUMNS: tuple[str, ...] = ("embed", "normalized_embed")
NORM_PERCENTILES: tuple[int, ...] = (1, 5, 50, 95, 99)


# ---------------------------------------------------------------------------
# Loader (same pattern as stages 01/02 — local copy, no util module)
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
    if not meta_path.exists():
        raise RuntimeError(f"splits metadata not found at {meta_path}")
    payload = load_json(meta_path)
    if "random_seed" not in payload:
        raise RuntimeError(
            f"{meta_path} missing 'random_seed'; earlier stages must run first"
        )
    seed = int(payload["random_seed"])
    logger.info("loaded seed=%d from %s", seed, meta_path)
    return seed


def read_train(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise RuntimeError(f"stage-5 train parquet not found at {path}")
    df = pl.read_parquet(path)
    validate_listens_schema(df, origin=f"stage-5 train ({path})")
    logger.info("read stage-5 train: %d rows", df.height)
    return df


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def load_train_embeddings(
    emb_path: Path, train_item_ids: pl.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the embeddings parquet, join on train items, coerce to numpy.

    Returns ``(item_ids, embed_arr, normalized_arr)`` with rows aligned
    across the three arrays. ``item_ids`` is sorted ascending (set by the
    inner join below).
    """
    wanted = pl.DataFrame({"item_id": train_item_ids}).unique().sort("item_id")
    logger.info("reading %s (columns: item_id, embed, normalized_embed)", emb_path)
    emb_df = pl.read_parquet(
        emb_path, columns=["item_id", "embed", "normalized_embed"]
    )
    logger.info("embeddings parquet: %d rows pre-join", emb_df.height)
    joined = emb_df.join(wanted, on="item_id", how="inner").sort("item_id")
    logger.info("after inner join on train items: %d rows", joined.height)

    if joined.height != wanted.height:
        missing = wanted.height - joined.height
        raise RuntimeError(
            f"coverage regression: {missing} train items have no embedding "
            f"(stage 2 should have enforced 100% coverage — re-run stage 2)"
        )

    item_ids = joined.get_column("item_id").to_numpy().astype(np.uint32, copy=False)
    embed_arr = np.asarray(
        joined.get_column("embed").to_list(), dtype=np.float32
    )
    normalized_arr = np.asarray(
        joined.get_column("normalized_embed").to_list(), dtype=np.float32
    )
    return item_ids, embed_arr, normalized_arr


def assert_shape_and_finite(
    arr: np.ndarray, label: str, expected_dim: int, n_items: int
) -> None:
    if arr.ndim != 2:
        raise RuntimeError(f"{label}: expected 2D array, got shape {arr.shape}")
    if arr.shape[0] != n_items:
        raise RuntimeError(
            f"{label}: row count {arr.shape[0]} != expected {n_items}"
        )
    if arr.shape[1] != expected_dim:
        raise RuntimeError(
            f"{label}: embedding dim {arr.shape[1]} != expected {expected_dim}"
        )
    if not np.all(np.isfinite(arr)):
        bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(f"{label}: {bad} non-finite values (NaN/Inf)")


def compute_popularity(train: pl.DataFrame) -> dict[int, int]:
    agg = (
        train.group_by("item_id")
        .agg(pl.len().alias("count"))
        .sort("item_id")
    )
    ids = agg.get_column("item_id").to_list()
    counts = agg.get_column("count").to_list()
    return {int(i): int(c) for i, c in zip(ids, counts)}


def popularity_array(
    item_ids: np.ndarray, pop_map: dict[int, int]
) -> np.ndarray:
    out = np.empty(item_ids.shape[0], dtype=np.int64)
    for idx, iid in enumerate(item_ids.tolist()):
        out[idx] = pop_map[iid]
    return out


# ---------------------------------------------------------------------------
# 7.2 norms
# ---------------------------------------------------------------------------


def compute_norm_stats(arr: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(arr, axis=1)
    stats: dict[str, float] = {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
    }
    for p in NORM_PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(norms, p))
    return stats


# ---------------------------------------------------------------------------
# 7.3 NN sanity
# ---------------------------------------------------------------------------


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return (arr / safe).astype(np.float32, copy=False)


def sample_anchor_ids(
    item_ids: np.ndarray, sample_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Pick ``sample_size`` anchor item_ids deterministically."""
    if sample_size > item_ids.shape[0]:
        raise RuntimeError(
            f"requested anchor sample {sample_size} > n_items {item_ids.shape[0]}"
        )
    sorted_ids = np.sort(item_ids)
    idx = rng.choice(sorted_ids.shape[0], size=sample_size, replace=False)
    return np.sort(sorted_ids[idx])


def nearest_neighbors_full(
    normalized: np.ndarray,
    item_ids: np.ndarray,
    anchor_ids: np.ndarray,
    top_k: int,
    pop_map: dict[int, int],
) -> list[dict[str, Any]]:
    """Compute top-K cosine neighbours of each anchor vs. the full set.

    ``normalized`` must be L2-normalised row-wise. Similarity is
    ``normalized @ normalized[anchor]^T``. Self is excluded.
    """
    id_to_idx = {int(iid): i for i, iid in enumerate(item_ids.tolist())}
    anchor_indices = np.array(
        [id_to_idx[int(a)] for a in anchor_ids.tolist()], dtype=np.int64
    )
    anchor_vecs = normalized[anchor_indices]  # (K, D)
    sims = normalized @ anchor_vecs.T  # (N, K)

    results: list[dict[str, Any]] = []
    for col, anchor_id in enumerate(anchor_ids.tolist()):
        col_sims = sims[:, col].copy()
        anchor_row = anchor_indices[col]
        col_sims[anchor_row] = -np.inf  # exclude self
        top_idx = np.argpartition(-col_sims, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-col_sims[top_idx])]
        neighbours = [
            {
                "item_id": int(item_ids[j]),
                "cos_sim": float(col_sims[j]),
                "popularity": int(pop_map[int(item_ids[j])]),
            }
            for j in top_idx.tolist()
        ]
        results.append(
            {
                "anchor_item_id": int(anchor_id),
                "anchor_popularity": int(pop_map[int(anchor_id)]),
                "neighbors": neighbours,
            }
        )
    return results


# ---------------------------------------------------------------------------
# 7.4 PCA visualisation
# ---------------------------------------------------------------------------


def fit_pca_2d(arr: np.ndarray) -> tuple[np.ndarray, list[float]]:
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(arr)
    return coords.astype(np.float32), [float(v) for v in pca.explained_variance_ratio_]


def plot_pca_static(
    coords: np.ndarray, popularity: np.ndarray, stem: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    color = np.log1p(popularity.astype(np.float64))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=2,
        c=color,
        cmap="viridis",
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("Item embeddings — 2D PCA (normalized_embed, all train items)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log(1 + interactions per item)")
    fig.tight_layout()
    savefig_both(fig, stem)


def sample_for_html(
    rng: np.random.Generator,
    n_items: int,
    sample_size: int,
    popularity: np.ndarray,
) -> np.ndarray:
    if sample_size >= n_items:
        return np.arange(n_items, dtype=np.int64)
    weights = np.sqrt(popularity.astype(np.float64) + 1.0)
    weights = weights / weights.sum()
    idx = rng.choice(n_items, size=sample_size, replace=False, p=weights)
    return np.sort(idx)


def _compose_hover_text(
    item_id: int, popularity: int, neighbours: list[tuple[int, float, int]]
) -> str:
    lines = [
        f"item_id: {item_id}",
        f"popularity: {popularity}",
        f"top-{len(neighbours)} neighbours (cosine, within HTML sample):",
    ]
    for rank, (nb_id, sim, nb_pop) in enumerate(neighbours, start=1):
        lines.append(f"  {rank}. id={nb_id}  cos={sim:.3f}  pop={nb_pop}")
    return "<br>".join(lines)


def plot_pca_html(
    coords: np.ndarray,
    item_ids: np.ndarray,
    popularity: np.ndarray,
    normalized_sample: np.ndarray,
    top_k: int,
    html_path: Path,
) -> None:
    """Build the interactive PCA scatter with rich hover tooltips.

    ``coords`` / ``item_ids`` / ``popularity`` / ``normalized_sample`` are
    already subsampled and row-aligned. Neighbours are computed inside the
    sample (not against the full train set) — the hover is for qualitative
    sanity only.
    """
    n = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=top_k + 1, metric="cosine", algorithm="brute")
    nn.fit(normalized_sample)
    distances, indices = nn.kneighbors(normalized_sample)
    # drop self (column 0); cosine sim = 1 - distance
    nb_idx = indices[:, 1 : top_k + 1]
    nb_sim = 1.0 - distances[:, 1 : top_k + 1]

    hover_texts: list[str] = []
    for i in range(n):
        nbs: list[tuple[int, float, int]] = []
        for rank in range(top_k):
            j = int(nb_idx[i, rank])
            nbs.append(
                (int(item_ids[j]), float(nb_sim[i, rank]), int(popularity[j]))
            )
        hover_texts.append(
            _compose_hover_text(int(item_ids[i]), int(popularity[i]), nbs)
        )

    color = np.log1p(popularity.astype(np.float64))
    fig = go.Figure(
        data=go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=4,
                color=color,
                colorscale="Viridis",
                opacity=0.65,
                colorbar=dict(title="log(1 + pop)"),
                showscale=True,
            ),
            text=hover_texts,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=(
            f"Item embeddings — 2D PCA (normalized_embed, "
            f"{n:,}-item popularity-weighted subsample)"
        ),
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
        width=1000,
        height=750,
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(html_path),
        include_plotlyjs="cdn",
        full_html=True,
    )
    logger.info("wrote interactive PCA HTML to %s", html_path)


# ---------------------------------------------------------------------------
# Plotting helpers (copied from stage 04 — same style the whole pipeline uses)
# ---------------------------------------------------------------------------


def setup_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def savefig_both(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{stem}.png"
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote figure %s(.png,.pdf)", stem)


def plot_norm_hist(arr: np.ndarray, column_label: str, stem: str) -> None:
    norms = np.linalg.norm(arr, axis=1).astype(np.float64)
    spread = float(norms.max() - norms.min())
    # When the column is already L2-normalised, the spread is ~1e-7 and
    # matplotlib refuses to build 80 finite-sized bins. Pin a tight symmetric
    # window around the mean so the figure still says "everything is on the
    # unit sphere" instead of crashing.
    if spread < 1e-4:
        center = float(norms.mean())
        bins = np.linspace(center - 5e-5, center + 5e-5, 41)
    else:
        bins = 80
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(norms, bins=bins, color="#4c72b0", edgecolor="none")
    axes[0].set_title(f"L2 norm — `{column_label}` (linear)")
    axes[0].set_xlabel("||v||_2")
    axes[0].set_ylabel("Number of items")

    axes[1].hist(norms, bins=bins, color="#4c72b0", edgecolor="none")
    axes[1].set_yscale("log")
    axes[1].set_title(f"L2 norm — `{column_label}` (log Y)")
    axes[1].set_xlabel("||v||_2")
    axes[1].set_ylabel("Number of items (log)")

    fig.tight_layout()
    savefig_both(fig, stem)


# ---------------------------------------------------------------------------
# Stats + report
# ---------------------------------------------------------------------------


def build_stage7_stats(
    *,
    train_item_count: int,
    embedding_dim: int,
    norms_by_col: dict[str, dict[str, float]],
    nn_seed: int,
    anchor_ids: np.ndarray,
    nn_results_by_col: dict[str, list[dict[str, Any]]],
    pca_n_items: int,
    pca_explained_variance: list[float],
    html_sample_size: int,
    html_sample_seed: int,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "description": (
            "Item embeddings sanity check: coverage, L2 norms, "
            "nearest-neighbour spot checks, 2D PCA visualisation."
        ),
        "source": {
            "train_parquet": str(TRAIN_PARQUET_PATH),
            "embeddings_parquet_rel": EMBEDDINGS_PARQUET_REL,
            "yambda_repo": YAMBDA_REPO,
            "train_item_count": train_item_count,
        },
        "coverage": {
            "train_items": train_item_count,
            "items_with_embedding": train_item_count,
            "missing": 0,
            "fraction": 1.0,
            "embedding_dim": embedding_dim,
            "vector_columns_inspected": list(VECTOR_COLUMNS),
        },
        "norms": norms_by_col,
        "nn_sanity": {
            "sample_size": int(anchor_ids.shape[0]),
            "top_k": EMBEDDING_NN_TOP_K,
            "seed": nn_seed,
            "sample_item_ids": [int(x) for x in anchor_ids.tolist()],
            "results_embed": nn_results_by_col["embed"],
            "results_normalized_embed": nn_results_by_col["normalized_embed"],
        },
        "pca": {
            "column_used": "normalized_embed",
            "n_items": pca_n_items,
            "explained_variance_ratio": pca_explained_variance,
            "html_sample_size": html_sample_size,
            "html_sample_seed": html_sample_seed,
            "html_top_k_neighbors": EMBEDDING_PCA_HTML_TOP_K,
        },
        "invariants_checked": [
            "train_item_coverage_100pct",
            "embedding_dim_matches_expected",
            "no_nan_or_inf",
            "anchor_sample_deterministic",
            "pca_row_count_matches_train_items",
        ],
        "artifacts": {
            "report": str(EMBEDDINGS_REPORT),
            "figures": [
                str(FIGURES_DIR / f"{EMBEDDINGS_PCA_FIG_STEM}.png"),
                str(FIGURES_DIR / f"{EMBEDDINGS_PCA_FIG_STEM}.pdf"),
                str(EMBEDDINGS_PCA_HTML_PATH),
                str(FIGURES_DIR / f"{EMBEDDINGS_NORMS_EMBED_STEM}.png"),
                str(FIGURES_DIR / f"{EMBEDDINGS_NORMS_EMBED_STEM}.pdf"),
                str(FIGURES_DIR / f"{EMBEDDINGS_NORMS_NORMALIZED_STEM}.png"),
                str(FIGURES_DIR / f"{EMBEDDINGS_NORMS_NORMALIZED_STEM}.pdf"),
            ],
        },
    }


def render_report(stats: dict[str, Any]) -> str:
    ts = stats["generated_at_utc"].replace("+00:00", " UTC")
    cov = stats["coverage"]
    norms = stats["norms"]
    nn = stats["nn_sanity"]
    pca = stats["pca"]
    src = stats["source"]

    def _norm_row(col: str) -> str:
        d = norms[col]
        return (
            f"| `{col}` | {d['mean']:.4f} | {d['std']:.4f} | {d['min']:.4f} | "
            f"{d['p1']:.4f} | {d['p5']:.4f} | {d['p50']:.4f} | "
            f"{d['p95']:.4f} | {d['p99']:.4f} | {d['max']:.4f} |"
        )

    def _nn_block(col: str) -> list[str]:
        results = nn[f"results_{col}"]
        lines = [
            f"### `{col}` — {nn['sample_size']} anchors × top-{nn['top_k']}",
            "",
            "| Anchor (id / pop) | "
            + " | ".join(f"n{i + 1} id (cos, pop)" for i in range(nn["top_k"]))
            + " |",
            "| :--- | " + " | ".join([":---"] * nn["top_k"]) + " |",
        ]
        for row in results:
            cells = [f"`{row['anchor_item_id']}` / {row['anchor_popularity']}"]
            for nb in row["neighbors"]:
                cells.append(
                    f"`{nb['item_id']}` ({nb['cos_sim']:.3f}, {nb['popularity']})"
                )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        return lines

    lines: list[str] = [
        "# Embeddings Analysis (Stage 7)",
        "",
        f"_Generated: {ts}_",
        "",
        "## What this stage does",
        "",
        "Sanity-checks the content-based item embeddings shipped with Yambda "
        "against the train-item set frozen by stage 5. All four paragraphs "
        "of spec section 7 (coverage, norms, nearest neighbours, PCA "
        "visualisation) are covered. No decision about which vector column "
        "(`embed` vs `normalized_embed`) to feed into content-based SASRec "
        "is made here — both are inspected and the pick is deferred to the "
        "training stage.",
        "",
        "## Inputs",
        "",
        f"- Train parquet: `{src['train_parquet']}`",
        f"- Embeddings source: `{src['yambda_repo']}` / `{src['embeddings_parquet_rel']}`"
        f" (cached under `{RAW_CACHE_DIR.relative_to(RAW_CACHE_DIR.parents[1])}/`)",
        f"- Train items: **{src['train_item_count']:,}**",
        f"- Vector columns inspected: {', '.join(f'`{c}`' for c in cov['vector_columns_inspected'])}",
        "",
        "## 7.1 Coverage and shape",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Train items | {cov['train_items']:,} |",
        f"| Items with embedding | {cov['items_with_embedding']:,} |",
        f"| Missing | {cov['missing']:,} |",
        f"| Coverage fraction | {cov['fraction']:.4f} |",
        f"| Embedding dimensionality | {cov['embedding_dim']} |",
        "",
        "Coverage must be 100% — stage 2 already filters train interactions "
        "down to items that have embeddings, so any missing item here signals "
        "a regression in stage 2 and the script fails fast. Dimensionality is "
        "asserted equal across all items and both columns.",
        "",
        "## 7.2 L2 norm distributions",
        "",
        "| Column | mean | std | min | p1 | p5 | p50 | p95 | p99 | max |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _norm_row("embed"),
        _norm_row("normalized_embed"),
        "",
        "`normalized_embed` is expected to lie on the unit sphere (all "
        "percentiles ≈ 1.0). `embed` is the raw audio-CNN output and its "
        "norms span a wider range. Picking the right column for SASRec is a "
        "training-stage decision, but the numbers are now available here "
        "for that choice.",
        "",
        "Figures:",
        "",
        f"- ![embed norms](figures/{EMBEDDINGS_NORMS_EMBED_STEM}.png)",
        f"- ![normalized_embed norms](figures/{EMBEDDINGS_NORMS_NORMALIZED_STEM}.png)",
        "",
        "## 7.3 Nearest-neighbour sanity",
        "",
        f"A deterministic sample of **{nn['sample_size']}** anchor items "
        f"(seed = {nn['seed']}) is drawn from the sorted train-item list. "
        f"For each anchor we compute cosine similarity against **all** "
        f"{cov['train_items']:,} train items (self excluded) and record the "
        f"top-{nn['top_k']} neighbours. Yambda does not ship genre or title "
        "metadata, so this is a numeric sanity check — distances should not "
        "be degenerate (e.g. all 1.0 or all 0.0) and neighbours should not "
        "all collapse onto the popularity head.",
        "",
    ]
    lines += _nn_block("embed")
    lines += _nn_block("normalized_embed")
    lines += [
        "## 7.4 PCA visualisation",
        "",
        f"Full 2D PCA is fit on **{pca['n_items']:,}** `normalized_embed` "
        f"vectors. Explained variance ratio per component: "
        f"{pca['explained_variance_ratio'][0]:.4f}, "
        f"{pca['explained_variance_ratio'][1]:.4f}"
        f" (cumulative "
        f"{sum(pca['explained_variance_ratio']):.4f}).",
        "",
        "**Static figure** — all items, coloured by `log(1 + popularity)`:",
        "",
        f"- ![PCA static](figures/{EMBEDDINGS_PCA_FIG_STEM}.png)",
        f"- PDF: `reports/figures/{EMBEDDINGS_PCA_FIG_STEM}.pdf`",
        "",
        "**Interactive figure** — `sqrt(popularity)`-weighted seeded "
        f"subsample of **{pca['html_sample_size']:,}** items "
        f"(seed = {pca['html_sample_seed']}). Hover tooltips show `item_id`, "
        f"`popularity` and the top-{pca['html_top_k_neighbors']} cosine "
        "neighbours **within the HTML subsample** (not the full set — that "
        "would be prohibitive at 260K × 260K). Rendering all train items "
        "into a single HTML would produce a ~60 MB, sluggish file; the "
        "subsample is a viewing compromise, not a downstream artifact.",
        "",
        f"- `reports/figures/fig_09_embeddings_pca.html` (open in a browser)",
        "",
        "## Invariant checks",
        "",
        "All of the following are asserted during the run; the script fails "
        "fast otherwise:",
        "",
    ]
    for inv in stats["invariants_checked"]:
        lines.append(f"- ✓ `{inv}`")
    lines += [
        "",
        "## Caveats",
        "",
        "- **No genre/title labels in Yambda.** NN sanity is numeric only. "
        "The real semantic check happens implicitly when SASRec is trained "
        "on top of these embeddings in a later stage.",
        "- **`embed` vs `normalized_embed`** choice is deferred. Both have "
        "been validated to be finite and of expected dimensionality; their "
        "norm distributions are in this report.",
        "- **Popularity colouring** in the PCA plot can visually mask "
        "semantic structure if popular items dominate one region — mitigated "
        "in the interactive version by the `sqrt(popularity)` weighting, but "
        "still worth keeping in mind when interpreting the plot.",
        "",
        "## Artifacts",
        "",
    ]
    for art in stats["artifacts"]["figures"]:
        rel = Path(art)
        try:
            rel = rel.relative_to(rel.parents[2])
        except ValueError:
            pass
        lines.append(f"- `{rel}`")
    lines += [
        "- `data/filter_stats.json` (section `stage7`)",
        "- `data/splits_metadata.json` (key `stage7_frozen_at_utc`)",
        "",
        "## Next stage",
        "",
        "Stage 8 (optional) pushes the processed dataset to the Hub. After "
        "that, experiment 1.1 (SASRec baseline) consumes `train.parquet` "
        "plus these embeddings.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", type=Path, default=TRAIN_PARQUET_PATH)
    p.add_argument("--splits-metadata", type=Path, default=SPLITS_METADATA_PATH)
    p.add_argument("--filter-stats", type=Path, default=FILTER_STATS_PATH)
    p.add_argument("--report", type=Path, default=EMBEDDINGS_REPORT)
    p.add_argument("--raw-cache", type=Path, default=RAW_CACHE_DIR)
    return p.parse_args()


def main() -> None:
    setup_logging()
    setup_plot_style()
    args = parse_args()

    seed = load_seed(args.splits_metadata)
    train = read_train(args.train_parquet)
    train_item_ids = train.get_column("item_id").unique()
    train_item_count = int(train_item_ids.len())
    logger.info("train has %d unique items", train_item_count)

    emb_path = download_parquet(EMBEDDINGS_PARQUET_REL, args.raw_cache)
    item_ids, embed_arr, normalized_arr = load_train_embeddings(
        emb_path, train_item_ids
    )

    # 7.1 — shape / finiteness / dimension match
    assert_shape_and_finite(
        embed_arr, "embed", EMBEDDING_DIM_EXPECTED, train_item_count
    )
    assert_shape_and_finite(
        normalized_arr,
        "normalized_embed",
        EMBEDDING_DIM_EXPECTED,
        train_item_count,
    )
    logger.info(
        "coverage 100%% (%d items) dim=%d, both columns finite",
        train_item_count,
        EMBEDDING_DIM_EXPECTED,
    )

    # 7.2 — norms
    norms_by_col = {
        "embed": compute_norm_stats(embed_arr),
        "normalized_embed": compute_norm_stats(normalized_arr),
    }
    logger.info("norm stats: %s", norms_by_col)
    plot_norm_hist(embed_arr, "embed", EMBEDDINGS_NORMS_EMBED_STEM)
    plot_norm_hist(
        normalized_arr, "normalized_embed", EMBEDDINGS_NORMS_NORMALIZED_STEM
    )

    # popularity lookup (used by NN + PCA colouring)
    pop_map = compute_popularity(train)
    popularity = popularity_array(item_ids, pop_map)

    # 7.3 — nearest neighbours against the full set, both columns
    rng = np.random.default_rng(seed)
    anchor_ids = sample_anchor_ids(item_ids, EMBEDDING_NN_SAMPLE_SIZE, rng)
    logger.info("anchor sample: %s", anchor_ids.tolist())

    normalized_for_embed = l2_normalize(embed_arr)
    normalized_for_norm = l2_normalize(normalized_arr)  # idempotent if truly unit
    nn_results = {
        "embed": nearest_neighbors_full(
            normalized_for_embed,
            item_ids,
            anchor_ids,
            EMBEDDING_NN_TOP_K,
            pop_map,
        ),
        "normalized_embed": nearest_neighbors_full(
            normalized_for_norm,
            item_ids,
            anchor_ids,
            EMBEDDING_NN_TOP_K,
            pop_map,
        ),
    }
    logger.info("nearest-neighbour spot checks computed for both columns")

    # 7.4 — 2D PCA on normalized_embed
    logger.info("fitting 2D PCA on %d normalized_embed vectors", train_item_count)
    coords, explained = fit_pca_2d(normalized_arr)
    logger.info(
        "PCA explained variance ratio: %.4f + %.4f = %.4f",
        explained[0],
        explained[1],
        sum(explained),
    )
    plot_pca_static(coords, popularity, EMBEDDINGS_PCA_FIG_STEM)

    html_idx = sample_for_html(
        rng, train_item_count, EMBEDDING_PCA_HTML_SAMPLE_SIZE, popularity
    )
    logger.info("HTML subsample: %d items", html_idx.shape[0])
    plot_pca_html(
        coords=coords[html_idx],
        item_ids=item_ids[html_idx],
        popularity=popularity[html_idx],
        normalized_sample=normalized_arr[html_idx],
        top_k=EMBEDDING_PCA_HTML_TOP_K,
        html_path=EMBEDDINGS_PCA_HTML_PATH,
    )

    stats = build_stage7_stats(
        train_item_count=train_item_count,
        embedding_dim=EMBEDDING_DIM_EXPECTED,
        norms_by_col=norms_by_col,
        nn_seed=seed,
        anchor_ids=anchor_ids,
        nn_results_by_col=nn_results,
        pca_n_items=train_item_count,
        pca_explained_variance=explained,
        html_sample_size=int(html_idx.shape[0]),
        html_sample_seed=seed,
    )
    update_json_section(args.filter_stats, "stage7", stats)
    update_json_section(
        args.splits_metadata,
        "stage7_frozen_at_utc",
        datetime.now(tz=timezone.utc).isoformat(),
    )

    report_md = render_report(stats)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_md, encoding="utf-8")
    logger.info("wrote %s", args.report)


if __name__ == "__main__":
    main()
