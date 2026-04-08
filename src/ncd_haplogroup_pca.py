"""
ncd_haplogroup_pca.py
Exploratory PCA within the NCD subset only, colored by haplogroup.

Usage:
    python src/ncd_haplogroup_pca.py --input data/raw/master_lineage.xlsx
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

COL_LABEL = "LABEL"
COL_HAPLO = "Haplogroup"
COL_ACCURACY = "Accuracy"
COL_DISTANCE = "distance_m"
COL_SEGMENT = "subc_id"

ENV_PREFIXES = ("l_CLI", "u_CLI", "l_TOP", "u_TOP", "l_LAC", "u_LAC", "l_SOL", "u_SOL")

NCD_MARKERS = {
    "BAN": "o",
    "GK": "s",
    "ZPB": "^",
    "LD": "D",
    "ZV": "P",
    "KOR": "X",
    "VOJ": "v",
}

NCD_COLORS = {
    "BAN": "#1b9e77",
    "GK": "#d95f02",
    "ZPB": "#7570b3",
    "LD": "#e7298a",
    "ZV": "#66a61e",
    "KOR": "#e6ab02",
    "VOJ": "#a6761d",
}


def load_data(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[COL_LABEL] == "A. torrentium - NCD"].copy()
    df = df[df[COL_ACCURACY] == "High"].copy()
    df = df[df[COL_DISTANCE] <= 1000].copy()
    df = df.drop_duplicates(subset=[COL_LABEL, COL_SEGMENT], keep="first").copy()
    return df


def get_env_variables(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(ENV_PREFIXES)]


def handle_missing_values(df: pd.DataFrame, env_vars: list[str], max_missing_pct: float = 0.3):
    keep = []
    for v in env_vars:
        if df[v].isna().mean() <= max_missing_pct:
            keep.append(v)

    df = df.copy()
    for v in keep:
        if df[v].isna().any():
            df[v] = df[v].fillna(df[v].median())
    return df, keep


def remove_constant_variables(df: pd.DataFrame, env_vars: list[str]) -> list[str]:
    return [v for v in env_vars if df[v].nunique(dropna=False) > 1]


def remove_highly_correlated(df: pd.DataFrame, env_vars: list[str], threshold: float = 0.98) -> list[str]:
    if len(env_vars) <= 1:
        return env_vars
    corr = df[env_vars].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return [v for v in env_vars if v not in to_drop]


def compute_haplo_centroids(pca_df: pd.DataFrame) -> pd.DataFrame:
    return (
        pca_df.groupby(COL_HAPLO)[["PC1", "PC2"]]
        .mean()
        .reset_index()
    )


def plot_ncd_pca(pca_df: pd.DataFrame, centroids: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(9, 7))

    for hap, marker in NCD_MARKERS.items():
        sub = pca_df[pca_df[COL_HAPLO] == hap]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["PC1"], sub["PC2"],
            s=65, alpha=0.88,
            c=NCD_COLORS.get(hap, "#666666"),
            marker=marker,
            label=f"{hap} (n={len(sub)})",
            edgecolors="black", linewidths=0.45
        )

    for _, row in centroids.iterrows():
        ax.scatter(
            row["PC1"], row["PC2"],
            s=220, marker="*",
            c=NCD_COLORS.get(row[COL_HAPLO], "#666666"),
            edgecolors="black", linewidths=1.0, zorder=10
        )
        ax.text(
            row["PC1"], row["PC2"],
            "  " + row[COL_HAPLO],
            fontsize=9, weight="bold", va="center"
        )

    ax.set_title("Exploratory PCA within A. torrentium - NCD")
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

    print(f"NCD records after filtering: {len(df)}")
    print("\nHaplogroup sizes:")
    print(df[COL_HAPLO].value_counts(dropna=False).to_string())

    env_vars = get_env_variables(df)
    df, env_vars = handle_missing_values(df, env_vars, max_missing_pct=0.30)
    env_vars = remove_constant_variables(df, env_vars)
    env_vars = remove_highly_correlated(df, env_vars, threshold=0.98)

    X = df[env_vars].values
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)

    pca_df = df[[COL_HAPLO]].copy()
    pca_df["PC1"] = pcs[:, 0]
    pca_df["PC2"] = pcs[:, 1]

    centroids = compute_haplo_centroids(pca_df)

    pc1 = float(pca.explained_variance_ratio_[0])
    pc2 = float(pca.explained_variance_ratio_[1])

    print("\nPCA variance explained:")
    print(f"PC1: {pc1:.4f} ({pc1*100:.2f}%)")
    print(f"PC2: {pc2:.4f} ({pc2*100:.2f}%)")
    print(f"PC1+PC2: {(pc1+pc2):.4f} ({(pc1+pc2)*100:.2f}%)")

    print("\nHaplogroup centroids:")
    print(centroids.to_string(index=False))

    pca_df.to_csv(out_tables / "ncd_haplogroup_pca_scores.csv", index=False)
    centroids.to_csv(out_tables / "ncd_haplogroup_pca_centroids.csv", index=False)

    summary = {
        "n_records": int(len(df)),
        "haplogroup_counts": df[COL_HAPLO].value_counts(dropna=False).to_dict(),
        "pc1_variance_explained": pc1,
        "pc2_variance_explained": pc2,
        "pc1_pc2_total": pc1 + pc2,
    }
    with open(out_tables / "ncd_haplogroup_pca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_ncd_pca(
        pca_df,
        centroids,
        out_figures / "ncd_haplogroup_pca.png"
    )

    print("\nSaved:")
    print(" - results_lineage/tables/ncd_haplogroup_pca_scores.csv")
    print(" - results_lineage/tables/ncd_haplogroup_pca_centroids.csv")
    print(" - results_lineage/tables/ncd_haplogroup_pca_summary.json")
    print(" - results_lineage/figures/ncd_haplogroup_pca.png")


if __name__ == "__main__":
    main()