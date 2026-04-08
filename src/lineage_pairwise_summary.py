"""
lineage_pairwise_summary.py
Pairwise summary for lineage-level ecological differentiation.

Uses PCA scores already produced by niche_overlap_lineage.py and creates:
- pairwise centroid-distance heatmap
- pairwise sample size table
- summary text table

Usage:
    python src/lineage_pairwise_summary.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COL_LABEL = "LABEL"

TARGET_LABELS = [
    "A. torrentium - CSE",
    "A. torrentium - SB",
    "A. torrentium - NCD",
    "A. bihariensis",
]


def build_distance_matrix(dist_df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    mat = pd.DataFrame(0.0, index=labels, columns=labels)

    for _, row in dist_df.iterrows():
        a = row["group_1"]
        b = row["group_2"]
        d = row["centroid_distance_pc12"]
        mat.loc[a, b] = d
        mat.loc[b, a] = d

    return mat


def plot_heatmap(mat: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat.values, cmap="YlOrRd")

    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticklabels(mat.index)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            txt = f"{val:.2f}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="black" if val < mat.values.max() * 0.6 else "white",
                fontsize=9
            )

    ax.set_title("Pairwise centroid distances in PCA space")
    plt.colorbar(im, ax=ax, shrink=0.85, label="Distance")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    out_tables = Path("results_lineage/tables")
    out_figures = Path("results_lineage/figures")
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    pca_scores = pd.read_csv(out_tables / "lineage_pca_scores.csv")
    centroids = pd.read_csv(out_tables / "lineage_pca_centroids.csv")
    distances = pd.read_csv(out_tables / "lineage_pca_centroid_distances.csv")

    counts = (
        pca_scores[COL_LABEL]
        .value_counts()
        .rename_axis("group")
        .reset_index(name="n")
    )

    print("Group sizes:")
    print(counts.to_string(index=False))

    print("\nPairwise centroid distances:")
    print(distances.to_string(index=False))

    mat = build_distance_matrix(distances, TARGET_LABELS)
    mat.to_csv(out_tables / "lineage_pairwise_distance_matrix.csv")

    plot_heatmap(
        mat,
        out_figures / "lineage_pairwise_distance_heatmap.png"
    )

    summary = {
        "largest_distance_pair": distances.iloc[0].to_dict(),
        "smallest_distance_pair": distances.iloc[-1].to_dict(),
        "group_sizes": counts.to_dict("records"),
    }

    with open(out_tables / "lineage_pairwise_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(" - results_lineage/tables/lineage_pairwise_distance_matrix.csv")
    print(" - results_lineage/tables/lineage_pairwise_summary.json")
    print(" - results_lineage/figures/lineage_pairwise_distance_heatmap.png")


if __name__ == "__main__":
    main()