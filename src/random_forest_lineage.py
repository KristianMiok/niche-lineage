"""
random_forest_lineage.py
Multiclass random forest robustness analysis for lineage-level groups.

Outputs:
- CV accuracy / macro-F1
- confusion matrix
- feature importances
- importance by variable type

Usage:
    python src/random_forest_lineage.py --input data/raw/master_lineage.xlsx
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict

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


def classify_variable(var_name: str) -> str:
    if "CLI" in var_name:
        return "Climate"
    if "TOP" in var_name:
        return "Topography"
    if "SOL" in var_name:
        return "Soil"
    if "LAC" in var_name:
        return "Land Cover"
    return "Other"


def importance_by_type(importances: pd.DataFrame) -> pd.DataFrame:
    tmp = importances.copy()
    tmp["type"] = tmp["variable"].apply(classify_variable)
    out = tmp.groupby("type", as_index=False)["importance"].sum()
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    out["pct"] = out["importance"] / out["importance"].sum() * 100
    return out


def plot_confusion(y_true, y_pred, labels, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_importance(importances: pd.DataFrame, output_path: str, n_top: int = 20):
    top = importances.head(n_top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, n_top * 0.32)))
    ax.barh(top["variable"], top["importance"])
    ax.set_xlabel("Random forest importance")
    ax.set_title(f"Top {n_top} variables")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--n-estimators", type=int, default=1000)
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
    print(f"Variables retained: {len(env_vars)}")

    X = df[env_vars].values
    y = df[COL_LABEL].values

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    cv_results = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    y_pred_cv = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    print("\nCross-validation performance:")
    print(f"Accuracy:    {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}")
    print(f"F1 macro:    {cv_results['test_f1_macro'].mean():.3f} ± {cv_results['test_f1_macro'].std():.3f}")
    print(f"F1 weighted: {cv_results['test_f1_weighted'].mean():.3f} ± {cv_results['test_f1_weighted'].std():.3f}")

    print("\nClassification report (cross-validated predictions):")
    print(classification_report(y, y_pred_cv, digits=3))

    clf.fit(X, y)
    importances = pd.DataFrame({
        "variable": env_vars,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importances["rank"] = np.arange(1, len(importances) + 1)
    importances["cumulative"] = importances["importance"].cumsum()

    by_type = importance_by_type(importances)

    print("\nTop 15 variables:")
    print(importances.head(15).to_string(index=False))

    print("\nImportance by variable type:")
    print(by_type.to_string(index=False))

    plot_confusion(
        y, y_pred_cv, TARGET_LABELS,
        out_figures / "rf_lineage_confusion_matrix.png"
    )
    plot_importance(
        importances,
        out_figures / "rf_lineage_feature_importance.png",
        n_top=20
    )

    importances.to_csv(out_tables / "rf_lineage_feature_importances.csv", index=False)
    by_type.to_csv(out_tables / "rf_lineage_importance_by_type.csv", index=False)

    summary = {
        "n_records": int(len(df)),
        "n_features": int(len(env_vars)),
        "class_counts": df[COL_LABEL].value_counts().to_dict(),
        "cv_accuracy_mean": float(cv_results["test_accuracy"].mean()),
        "cv_accuracy_sd": float(cv_results["test_accuracy"].std()),
        "cv_f1_macro_mean": float(cv_results["test_f1_macro"].mean()),
        "cv_f1_macro_sd": float(cv_results["test_f1_macro"].std()),
        "cv_f1_weighted_mean": float(cv_results["test_f1_weighted"].mean()),
        "cv_f1_weighted_sd": float(cv_results["test_f1_weighted"].std()),
        "importance_by_type": by_type.to_dict("records"),
        "top_variables": importances.head(20).to_dict("records"),
    }

    with open(out_tables / "rf_lineage_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(" - results_lineage/tables/rf_lineage_summary.json")
    print(" - results_lineage/tables/rf_lineage_feature_importances.csv")
    print(" - results_lineage/tables/rf_lineage_importance_by_type.csv")
    print(" - results_lineage/figures/rf_lineage_confusion_matrix.png")
    print(" - results_lineage/figures/rf_lineage_feature_importance.png")


if __name__ == "__main__":
    main()