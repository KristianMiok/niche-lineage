"""
robustness_checks_lineage.py

Runs the four robustness checks requested for the lineage paper:
1) Random-forest permutation test
2) PERMANOVA
3) Pairwise Schoener's D on PC1-PC5
4) Per-class metrics for DT and RF

Outputs:
- results_lineage/tables/permutation_test_summary.json
- results_lineage/tables/permanova_result.json
- results_lineage/tables/schoeners_d_pairwise.csv
- results_lineage/tables/classification_per_class.csv

Usage:
    python src/robustness_checks_lineage.py --input data/raw/master_lineage.xlsx
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import pairwise_distances


TARGET_LABELS = [
    "A. torrentium - CSE",
    "A. torrentium - SB",
    "A. torrentium - NCD",
    "A. bihariensis",
]

COL_LABEL = "LABEL"
COL_ACCURACY = "Accuracy"
COL_DISTANCE = "distance_m"
COL_SEGMENT = "subc_id"
ENV_PREFIXES = ("l_CLI", "u_CLI", "l_TOP", "u_TOP", "l_LAC", "u_LAC", "l_SOL", "u_SOL")


# -----------------------------
# Data preparation
# -----------------------------
def load_data(filepath: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


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
    X = df[env_vars].values
    y = df[COL_LABEL].values
    return df, X, y, env_vars


# -----------------------------
# 1) RF permutation test
# -----------------------------
def rf_cv_accuracy(X, y, cv, n_estimators=500):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    res = cross_validate(
        clf, X, y, cv=cv,
        scoring={"accuracy": "accuracy"},
        n_jobs=-1,
        return_train_score=False,
    )
    return float(res["test_accuracy"].mean())


def permutation_test_rf(X, y, n_permutations=1000, n_estimators=500, random_state=42):
    rng = np.random.default_rng(random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    observed_accuracy = rf_cv_accuracy(X, y, cv=cv, n_estimators=n_estimators)
    print(f"\nObserved RF accuracy: {observed_accuracy:.4f}")

    null_accuracies = []
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        acc = rf_cv_accuracy(X, y_perm, cv=cv, n_estimators=n_estimators)
        null_accuracies.append(acc)
        if (i + 1) % 50 == 0:
            print(f"  Permutation {i+1}/{n_permutations}")

    null_accuracies = np.array(null_accuracies)
    p_value = float((1 + np.sum(null_accuracies >= observed_accuracy)) / (1 + n_permutations))

    return {
        "null_accuracy_mean": float(null_accuracies.mean()),
        "null_accuracy_sd": float(null_accuracies.std()),
        "observed_accuracy": float(observed_accuracy),
        "p_value": p_value,
        "n_permutations": int(n_permutations),
    }


# -----------------------------
# 2) PERMANOVA
# -----------------------------
def permanova_from_distmat(D, groups, n_permutations=999, random_state=42):
    """
    Simple one-factor PERMANOVA using a Euclidean distance matrix.
    Returns pseudo-F, R2, p-value.
    """
    rng = np.random.default_rng(random_state)
    groups = np.asarray(groups)
    n = len(groups)

    # Gower-centered matrix
    A = -0.5 * (D ** 2)
    I = np.eye(n)
    J = np.ones((n, n)) / n
    G = (I - J) @ A @ (I - J)

    unique_groups = np.unique(groups)

    def compute_stats(group_labels):
        H = np.zeros((n, n))
        for g in np.unique(group_labels):
            idx = np.where(group_labels == g)[0]
            ng = len(idx)
            H[np.ix_(idx, idx)] = 1.0 / ng

        SS_total = np.trace(G)
        SS_between = np.trace(H @ G)
        SS_within = SS_total - SS_between

        df_between = len(np.unique(group_labels)) - 1
        df_within = n - len(np.unique(group_labels))

        MS_between = SS_between / df_between
        MS_within = SS_within / df_within
        F = MS_between / MS_within if MS_within > 0 else np.inf
        R2 = SS_between / SS_total if SS_total > 0 else np.nan
        return F, R2

    observed_F, observed_R2 = compute_stats(groups)

    perm_F = []
    for i in range(n_permutations):
        perm_groups = rng.permutation(groups)
        F_perm, _ = compute_stats(perm_groups)
        perm_F.append(F_perm)

    perm_F = np.array(perm_F)
    p_value = float((1 + np.sum(perm_F >= observed_F)) / (1 + n_permutations))

    return {
        "R2": float(observed_R2),
        "F": float(observed_F),
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
    }


# -----------------------------
# 3) Schoener's D on PC1-PC5
# -----------------------------
def schoeners_d_from_pcs(X, y, n_components=5, bins=8):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(Xs)

    results = []
    for g1, g2 in combinations(TARGET_LABELS, 2):
        A = pcs[y == g1]
        B = pcs[y == g2]

        AB = np.vstack([A, B])
        mins = AB.min(axis=0)
        maxs = AB.max(axis=0)

        edges = []
        for j in range(n_components):
            if mins[j] == maxs[j]:
                edges.append(np.array([mins[j] - 1e-6, maxs[j] + 1e-6]))
            else:
                edges.append(np.linspace(mins[j], maxs[j], bins + 1))

        hist_A, _ = np.histogramdd(A, bins=edges)
        hist_B, _ = np.histogramdd(B, bins=edges)

        pA = hist_A / hist_A.sum()
        pB = hist_B / hist_B.sum()

        D = 1.0 - 0.5 * np.abs(pA - pB).sum()
        results.append({
            "group_1": g1,
            "group_2": g2,
            "schoeners_D": float(D),
        })

    return pd.DataFrame(results)


# -----------------------------
# 4) Per-class metrics
# -----------------------------
def per_class_metrics(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    dt = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=1000,
        class_weight="balanced",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    y_pred_dt = cross_val_predict(dt, X, y, cv=cv)
    y_pred_rf = cross_val_predict(rf, X, y, cv=cv, n_jobs=-1)

    dt_report = classification_report(y, y_pred_dt, labels=TARGET_LABELS, output_dict=True, zero_division=0)
    rf_report = classification_report(y, y_pred_rf, labels=TARGET_LABELS, output_dict=True, zero_division=0)

    rows = []
    for g in TARGET_LABELS:
        rows.append({
            "group": g,
            "DT_precision": dt_report[g]["precision"],
            "DT_recall": dt_report[g]["recall"],
            "DT_F1": dt_report[g]["f1-score"],
            "RF_precision": rf_report[g]["precision"],
            "RF_recall": rf_report[g]["recall"],
            "RF_F1": rf_report[g]["f1-score"],
        })

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to master_lineage.xlsx")
    parser.add_argument("--rf_permutations", type=int, default=1000)
    parser.add_argument("--permanova_permutations", type=int, default=999)
    parser.add_argument("--rf_trees_permutation", type=int, default=500)
    args = parser.parse_args()

    out_tables = Path("results_lineage/tables")
    out_tables.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    df = apply_filters(df)
    df, X, y, env_vars = prepare_matrix(df)

    print(f"\nRetained matrix: {X.shape[0]} records × {X.shape[1]} variables")

    # 1) RF permutation test
    print("\n=== 1) RF PERMUTATION TEST ===")
    perm_summary = permutation_test_rf(
        X, y,
        n_permutations=args.rf_permutations,
        n_estimators=args.rf_trees_permutation,
        random_state=42,
    )
    with open(out_tables / "permutation_test_summary.json", "w") as f:
        json.dump(perm_summary, f, indent=2)
    print(json.dumps(perm_summary, indent=2))

    # 2) PERMANOVA
    print("\n=== 2) PERMANOVA ===")
    D = pairwise_distances(X, metric="euclidean")
    permanova = permanova_from_distmat(
        D, y,
        n_permutations=args.permanova_permutations,
        random_state=42,
    )
    with open(out_tables / "permanova_result.json", "w") as f:
        json.dump(permanova, f, indent=2)
    print(json.dumps(permanova, indent=2))

    # 3) Schoener's D
    print("\n=== 3) SCHOENER'S D (PC1-PC5) ===")
    schoener_df = schoeners_d_from_pcs(X, y, n_components=5, bins=8)
    schoener_df.to_csv(out_tables / "schoeners_d_pairwise.csv", index=False)
    print(schoener_df.to_string(index=False))

    # 4) Per-class metrics
    print("\n=== 4) PER-CLASS METRICS ===")
    class_df = per_class_metrics(X, y)
    class_df.to_csv(out_tables / "classification_per_class.csv", index=False)
    print(class_df.to_string(index=False))

    print("\nSaved files:")
    print(" - results_lineage/tables/permutation_test_summary.json")
    print(" - results_lineage/tables/permanova_result.json")
    print(" - results_lineage/tables/schoeners_d_pairwise.csv")
    print(" - results_lineage/tables/classification_per_class.csv")


if __name__ == "__main__":
    main()