"""
niche_overlap_lineage.py
PCA-based ecological niche visualization for lineage-level groups.

Main outputs:
- PCA scatter plot colored by LABEL
- NCD subgroup markers by Haplogroup
- group centroids
- pairwise centroid distances in PCA space
- variance explained by PC1 and PC2

Usage:
    python src/niche_overlap_lineage.py --input data/raw/master_lineage.xlsx
"""

import argparse
import json
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TARGET_LABELS = [
    "A. torrentium - CSE",
    "A. torrentium - SB",
    "A. torrentium - NCD",
    "A. bihariensis",
]

COL_LABEL = "LABEL"
COL_HAPLO = "Haplogroup"
COL_ACCURACY = "Accuracy"
COL_DISTANCE = "distance_m"
COL_SEGMENT = "subc_id"

ENV_PREFIXES = ("l_CLI", "u_CLI", "l_TOP", "u_TOP", "l_LAC", "u_LAC", "l_SOL", "u_SOL")

LABEL_COLORS = {
    "A. torrentium - CSE": "#1f77b4",
    "A. torrentium - SB": "#ff7f0e",
    "A. torrentium - NCD": "#2ca02c",
    "A. bihariensis": "#d62728",
}

NCD_MARKERS = {
    "BAN": "o",
    "GK": "s",
    "ZPB": "^",
    "LD": "D",
    "ZV": "P",
    "KOR": "X",
    "VOJ": "v",
}


def load_data(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Starting with {len(df):,} records")

    df = df[df[COL_LABEL].isin(TARGET_LABELS)].copy()
    print(f"After LABEL filter: {len(df):,}")

    df = df[df[COL_ACCURACY] == "High"].copy()
    print(f"After Accuracy=High: {len(df):,}")

    df = df[df[COL_DISTANCE] <= 1000].copy()
    print(f"After distance <=1000 m: {len(df):,}")

    before = len(df)
    df = df.drop_duplicates(subset=[COL_LABEL, COL_SEGMENT], keep="first").copy()
    print(f"After deduplication by LABEL+subc_id: {len(df):,} (removed {before - len(df):,})")

    return df


def get_env_variables(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(ENV_PREFIXES)]


def handle_missing_values(df: pd.DataFrame, env_vars: list[str], max_missing_pct: float = 0.3):
    keep_vars = []
    for v in env_vars:
        if df[v].isna().mean() <= max_missing_pct:
            keep_vars.append(v)

    df = df.copy()
    for v in keep_vars:
        if df[v].isna().any():
            df[v] = df[v].fillna(df[v].median())

    return df, keep_vars


def remove_constant_variables(df: pd.DataFrame, env_vars: list[str]) -> list[str]:
    return [v for v in env_vars if df[v].nunique(dropna=False) > 1]


def remove_highly_correlated(df: pd.DataFrame, env_vars: list[str], threshold: float = 0.98) -> list[str]:
    if len(env_vars) <= 1:
        return env_vars

    corr = df[env_vars].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return [v for v in env_vars if v not in to_drop]


def compute_centroids(pca_df: pd.DataFrame) -> pd.DataFrame:
    return (
        pca_df.groupby(COL_LABEL)[["PC1", "PC2"]]
        .mean()
        .reset_index()
    )


def pairwise_centroid_distances(centroids: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(centroids[COL_LABEL], 2):
        row_a = centroids[centroids[COL_LABEL] == a].iloc[0]
        row_b = centroids[centroids[COL_LABEL] == b].iloc[0]
        d = np.sqrt((row_a["PC1"] - row_b["PC1"])**2 + (row_a["PC2"] - row_b["PC2"])**2)
        rows.append({
            "group_1": a,
            "group_2": b,
            "centroid_distance_pc12": float(d),
        })
    return pd.DataFrame(rows).sort_values("centroid_distance_pc12", ascending=False)


def plot_pca_scatter(pca_df: pd.DataFrame, centroids: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot non-NCD groups normally
    for label in TARGET_LABELS:
        sub = pca_df[pca_df[COL_LABEL] == label]
        color = LABEL_COLORS[label]

        if label == "A. torrentium - CSE":
            # Push CSE into the background a bit
            ax.scatter(
                sub["PC1"], sub["PC2"],
                s=22, alpha=0.35, c=color, label=label,
                edgecolors="none", zorder=1
            )

        elif label == "A. torrentium - SB":
            ax.scatter(
                sub["PC1"], sub["PC2"],
                s=38, alpha=0.70, c=color, label=label,
                edgecolors="none", zorder=2
            )

        elif label == "A. bihariensis":
            ax.scatter(
                sub["PC1"], sub["PC2"],
                s=42, alpha=0.75, c=color, label=label,
                edgecolors="white", linewidths=0.3, zorder=3
            )

        else:
            # NCD: split by haplogroup with distinct markers
            for hap, marker in NCD_MARKERS.items():
                sub_h = sub[sub[COL_HAPLO] == hap]
                if len(sub_h) == 0:
                    continue
                ax.scatter(
                    sub_h["PC1"], sub_h["PC2"],
                    s=62, alpha=0.90, c=color, marker=marker,
                    label=f"NCD - {hap}",
                    edgecolors="black", linewidths=0.5, zorder=4
                )

    # Plot centroids
    for _, row in centroids.iterrows():
        ax.scatter(
            row["PC1"], row["PC2"],
            s=260, c=LABEL_COLORS[row[COL_LABEL]],
            marker="*",
            edgecolors="black",
            linewidths=1.1,
            zorder=10
        )
        ax.text(
            row["PC1"], row["PC2"],
            "  " + row[COL_LABEL].replace("A. torrentium - ", "").replace("A. ", ""),
            fontsize=10, weight="bold", va="center", zorder=11
        )

    ax.set_title("PCA niche space of lineage-level groups")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.axvline(0, color="lightgray", linewidth=0.8)
    ax.legend(fontsize=8, loc="best", frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    args = parser.parse_args()

    out_tables = Path("results_lineage/tables")
    out_figures = Path("results_lineage/figures")
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    df = apply_filters(df)

    env_vars = get_env_variables(df)
    print(f"Environmental variables detected: {len(env_vars)}")

    df, env_vars = handle_missing_values(df, env_vars, max_missing_pct=0.30)
    env_vars = remove_constant_variables(df, env_vars)
    env_vars = remove_highly_correlated(df, env_vars, threshold=0.98)

    print(f"Variables retained for PCA: {len(env_vars)}")

    X = df[env_vars].values
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)

    pca_df = df[[COL_LABEL, COL_HAPLO]].copy()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]

    centroids = compute_centroids(pca_df)
    distances = pairwise_centroid_distances(centroids)

    pc1_var = float(pca.explained_variance_ratio_[0])
    pc2_var = float(pca.explained_variance_ratio_[1])

    print("\nPCA variance explained:")
    print(f"PC1: {pc1_var:.4f} ({pc1_var*100:.2f}%)")
    print(f"PC2: {pc2_var:.4f} ({pc2_var*100:.2f}%)")
    print(f"PC1+PC2: {(pc1_var + pc2_var):.4f} ({(pc1_var + pc2_var)*100:.2f}%)")

    print("\nCentroids:")
    print(centroids.to_string(index=False))

    print("\nPairwise centroid distances:")
    print(distances.to_string(index=False))

    pca_df.to_csv(out_tables / "lineage_pca_scores.csv", index=False)
    centroids.to_csv(out_tables / "lineage_pca_centroids.csv", index=False)
    distances.to_csv(out_tables / "lineage_pca_centroid_distances.csv", index=False)

    summary = {
        "n_records": int(len(df)),
        "n_features": int(len(env_vars)),
        "pc1_variance_explained": pc1_var,
        "pc2_variance_explained": pc2_var,
        "pc1_pc2_total": pc1_var + pc2_var,
        "class_counts": df[COL_LABEL].value_counts().to_dict(),
    }
    with open(out_tables / "lineage_pca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_pca_scatter(
        pca_df,
        centroids,
        out_figures / "lineage_pca_scatter_haplogroups.png"
    )

    print("\nSaved:")
    print(" - results_lineage/tables/lineage_pca_scores.csv")
    print(" - results_lineage/tables/lineage_pca_centroids.csv")
    print(" - results_lineage/tables/lineage_pca_centroid_distances.csv")
    print(" - results_lineage/tables/lineage_pca_summary.json")
    print(" - results_lineage/figures/lineage_pca_scatter_haplogroups.png")


if __name__ == "__main__":
    main()