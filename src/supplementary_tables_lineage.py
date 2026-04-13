"""
supplementary_tables_lineage.py

Creates the two supplementary tables requested by Lucian:

1) Table S2: retained environmental variables with domain, description, unit,
   and mean ± SD by lineage group

2) Table S4: PCA loadings for PC1-PC5, plus variance explained

Outputs:
- results_lineage/tables/table_S2_variables.csv
- results_lineage/tables/table_S4_pca_loadings.csv

Usage:
    python src/supplementary_tables_lineage.py --input data/raw/master_lineage.xlsx --glossary data/raw/S2.xlsx
"""

import argparse
from pathlib import Path

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

SHORT_GROUPS = {
    "A. torrentium - CSE": "CSE",
    "A. torrentium - SB": "SB",
    "A. torrentium - NCD": "NCD",
    "A. bihariensis": "AUB",
}

COL_LABEL = "LABEL"
COL_ACCURACY = "Accuracy"
COL_DISTANCE = "distance_m"
COL_SEGMENT = "subc_id"

ENV_PREFIXES = ("l_CLI", "u_CLI", "l_TOP", "u_TOP", "l_LAC", "u_LAC", "l_SOL", "u_SOL")


# -----------------------------
# Shared data prep
# -----------------------------
def load_data(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[COL_LABEL].isin(TARGET_LABELS)].copy()
    df = df[df[COL_ACCURACY] == "High"].copy()
    df = df[df[COL_DISTANCE] <= 1000].copy()
    df = df.drop_duplicates(subset=[COL_LABEL, COL_SEGMENT], keep="first").copy()
    return df


def get_env_variables(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(ENV_PREFIXES)]


def handle_missing_values(df: pd.DataFrame, env_vars: list[str], max_missing_pct: float = 0.30):
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


def prepare_matrix(df: pd.DataFrame):
    env_vars = get_env_variables(df)
    df, env_vars = handle_missing_values(df, env_vars, max_missing_pct=0.30)
    env_vars = remove_constant_variables(df, env_vars)
    env_vars = remove_highly_correlated(df, env_vars, threshold=0.98)
    return df, env_vars


# -----------------------------
# Variable metadata from S2.xlsx
# -----------------------------
def clean_code(raw) -> str:
    s = str(raw).strip()
    if s == "nan" or s == "":
        return ""
    return s.replace("-", "_")


def classify_domain(var_name: str) -> str:
    if "CLI" in var_name:
        return "Climate"
    if "TOP" in var_name:
        return "Topography"
    if "SOL" in var_name:
        return "Soil"
    if "LAC" in var_name:
        return "Land Cover"
    return "Other"


def load_glossary(glossary_path: str) -> dict:
    """
    Returns:
        glossary[var_code] = {"description": ..., "unit": ..., "domain": ...}
    """
    glossary = {}

    # Climate
    cli = pd.read_excel(glossary_path, sheet_name="CLIMATE")
    for _, row in cli.iterrows():
        for raw_code in [row.get("Local Climate", ""), row.get("Upstream Climate", "")]:
            code = clean_code(raw_code)
            if code:
                glossary[code] = {
                    "description": str(row.get("Definition", "")).strip(),
                    "unit": str(row.get("Unit", "")).strip(),
                    "domain": "Climate",
                }

    # Soil
    sol = pd.read_excel(glossary_path, sheet_name="SOIL")
    for _, row in sol.iterrows():
        for raw_code in [row.get("Local Soil", ""), row.get("Upstream Soil", "")]:
            code = clean_code(raw_code)
            if code:
                glossary[code] = {
                    "description": str(row.get("Definition", "")).strip(),
                    "unit": str(row.get("Unit", "")).strip(),
                    "domain": "Soil",
                }

    # Land cover
    lac = pd.read_excel(glossary_path, sheet_name="LAND COVER")
    for _, row in lac.iterrows():
        for raw_code in [row.get("Local Climate", ""), row.get("Upstream Climate", "")]:
            code = clean_code(raw_code)
            if code:
                glossary[code] = {
                    "description": str(row.get("Definition", "")).strip(),
                    "unit": str(row.get("Unit", "")).strip(),
                    "domain": "Land Cover",
                }

    # Topography
    top = pd.read_excel(glossary_path, sheet_name="TOPOGRAPHY")
    for _, row in top.iterrows():
        for raw_code in [row.get("Local Topography", ""), row.get("Upstream Topography", "")]:
            code = clean_code(raw_code)
            if code:
                glossary[code] = {
                    "description": str(row.get("Definition (Hydrography90m)", "")).strip(),
                    "unit": str(row.get("Unit", "")).strip(),
                    "domain": "Topography",
                }

    return glossary


# -----------------------------
# Table S2
# -----------------------------
def make_table_s2(df: pd.DataFrame, env_vars: list[str], glossary: dict) -> pd.DataFrame:
    rows = []

    for var in env_vars:
        meta = glossary.get(var, {})
        row = {
            "variable": var,
            "domain": meta.get("domain", classify_domain(var)),
            "description": meta.get("description", ""),
            "unit": meta.get("unit", ""),
        }

        for full_group, short_group in SHORT_GROUPS.items():
            vals = df.loc[df[COL_LABEL] == full_group, var].astype(float)
            row[f"mean_{short_group}"] = float(vals.mean())
            row[f"sd_{short_group}"] = float(vals.std(ddof=1))

        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["domain", "variable"]).reset_index(drop=True)


# -----------------------------
# Table S4
# -----------------------------
def make_table_s4(df: pd.DataFrame, env_vars: list[str]) -> pd.DataFrame:
    X = df[env_vars].values
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=5, random_state=42)
    pca.fit(Xs)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=env_vars,
        columns=["PC1", "PC2", "PC3", "PC4", "PC5"]
    ).reset_index().rename(columns={"index": "variable"})

    variance_row = pd.DataFrame([{
        "variable": "variance_explained",
        "PC1": float(pca.explained_variance_ratio_[0]),
        "PC2": float(pca.explained_variance_ratio_[1]),
        "PC3": float(pca.explained_variance_ratio_[2]),
        "PC4": float(pca.explained_variance_ratio_[3]),
        "PC5": float(pca.explained_variance_ratio_[4]),
    }])

    out = pd.concat([loadings, variance_row], ignore_index=True)
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to master_lineage.xlsx")
    parser.add_argument("--glossary", "-g", default="data/raw/S2.xlsx", help="Path to S2.xlsx glossary")
    args = parser.parse_args()

    out_tables = Path("results_lineage/tables")
    out_tables.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    df = apply_filters(df)
    df, env_vars = prepare_matrix(df)

    print(f"Filtered dataset: {len(df)} records")
    print(f"Retained variables: {len(env_vars)}")

    glossary = load_glossary(args.glossary)
    print(f"Glossary entries loaded: {len(glossary)}")

    # Table S2
    table_s2 = make_table_s2(df, env_vars, glossary)
    s2_path = out_tables / "table_S2_variables.csv"
    table_s2.to_csv(s2_path, index=False)

    # Table S4
    table_s4 = make_table_s4(df, env_vars)
    s4_path = out_tables / "table_S4_pca_loadings.csv"
    table_s4.to_csv(s4_path, index=False)

    print(f"Saved: {s2_path}")
    print(f"Saved: {s4_path}")


if __name__ == "__main__":
    main()