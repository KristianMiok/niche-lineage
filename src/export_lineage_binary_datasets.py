# src/export_lineage_binary_datasets.py
"""
Create binary ML datasets from the A. torrentium lineage master table.

Input:
    data/raw/master_lineage.xlsx

Output folder (created next to the input file):
    data/raw/master_lineage_binary_datasets/

Exports:
    1) Pairwise LABEL comparisons:
       - CSE vs SB
       - CSE vs NCD
       - SB vs NCD
       - NCD vs A. bihariensis

    2) Pairwise Haplogroup comparisons within NCD:
       - all valid pairs with at least MIN_HAPLOGROUP_N rows per class

Each exported CSV contains:
    - numeric predictor columns only
    - y ........ binary target (0/1)
    - label .... readable class label for the row

Encoding:
    y = 0 for the first class in the filename
    y = 1 for the second class in the filename

Example:
    binary_label_cse_vs_sb.csv
    - y = 0 -> A. torrentium - CSE
    - y = 1 -> A. torrentium - SB
"""

from pathlib import Path
from itertools import combinations
import json
import re

import pandas as pd


INPUT_XLSX = Path("data/raw/master_lineage.xlsx")
OUTPUT_DIR_NAME = "master_lineage_binary_datasets"

# Minimum rows per class for haplogroup pair exports
MIN_HAPLOGROUP_N = 10

# LABEL comparisons to export
LABEL_PAIRS = [
    ("A. torrentium - CSE", "A. torrentium - SB"),
    ("A. torrentium - CSE", "A. torrentium - NCD"),
    ("A. torrentium - SB", "A. torrentium - NCD"),
    ("A. torrentium - NCD", "A. bihariensis"),
]

# Columns that should never be predictors
EXCLUDE_COLUMNS = {
    "OBJECTID",
    "WoCID",
    "lat_or",
    "long_or",
    "lat_snap",
    "long_snap",
    "Accuracy",
    "Crayfish_scientific_name",
    "Status",
    "Year_of_record",
    "basin_id",
    "subc_id",
    "reg_id",
    "NCBI_accession_code",
    "Claim_extinction",
    "Comments",
    "AccNoCOI",
    "Haplogroup",
    "LABEL",
}

# Prefixes of environmental predictors we want to keep
PREDICTOR_PREFIXES = (
    "l_CLI", "u_CLI",
    "l_TOP", "u_TOP",
    "l_SOL", "u_SOL",
    "l_LAC", "u_LAC",
)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("a. ", "a_")
    text = text.replace(" - ", "_")
    text = text.replace(" ", "_")
    text = text.replace("/", "_")
    text = text.replace("ž", "z").replace("š", "s").replace("č", "c")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convert mixed-format numeric columns to numeric.
    Handles comma decimals like '43,481852'.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s

    s = s.astype(str).str.strip()

    # Preserve missing values
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Replace decimal comma with decimal point
    s = s.str.replace(",", ".", regex=False)

    return pd.to_numeric(s, errors="coerce")


def load_master_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    print(f"Loading: {path}")
    df = pd.read_excel(path)
    print(f"Loaded shape: {df.shape}")
    return df


def get_predictor_columns(df: pd.DataFrame) -> list[str]:
    """
    Keep only environmental predictor columns.
    """
    predictor_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        if col.startswith(PREDICTOR_PREFIXES):
            predictor_cols.append(col)

    return predictor_cols


def coerce_predictors_to_numeric(df: pd.DataFrame, predictor_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in predictor_cols:
        df[col] = clean_numeric_series(df[col])
    return df


def export_binary_dataset(
    df: pd.DataFrame,
    predictor_cols: list[str],
    class_col: str,
    class_a: str,
    class_b: str,
    out_path: Path,
) -> dict:
    """
    Create one binary dataset:
    - filter rows to class_a and class_b
    - keep numeric predictors
    - add y and label
    """
    sub = df[df[class_col].isin([class_a, class_b])].copy()

    n_a = (sub[class_col] == class_a).sum()
    n_b = (sub[class_col] == class_b).sum()

    if n_a == 0 or n_b == 0:
        print(f"Skipping {out_path.name}: one class missing ({n_a}, {n_b})")
        return {
            "file": out_path.name,
            "class_col": class_col,
            "class_0": class_a,
            "class_1": class_b,
            "n_class_0": int(n_a),
            "n_class_1": int(n_b),
            "n_rows": int(len(sub)),
            "n_predictors": 0,
            "written": False,
            "reason": "missing class",
        }

    work = sub[predictor_cols].copy()

    # Drop columns that are entirely missing within this subset
    non_all_missing = [c for c in work.columns if not work[c].isna().all()]
    work = work[non_all_missing]

    # Keep only columns that are actually numeric after coercion
    numeric_cols = []
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            numeric_cols.append(col)

    work = work[numeric_cols].copy()

    work["y"] = (sub[class_col] == class_b).astype(int).values
    work["label"] = sub[class_col].values

    out_path.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(out_path, index=False)

    print(
        f"Wrote {out_path.name}: "
        f"{len(work):,} rows | {len(numeric_cols):,} predictors | "
        f"{class_a}={n_a:,}, {class_b}={n_b:,}"
    )

    return {
        "file": out_path.name,
        "class_col": class_col,
        "class_0": class_a,
        "class_1": class_b,
        "n_class_0": int(n_a),
        "n_class_1": int(n_b),
        "n_rows": int(len(work)),
        "n_predictors": int(len(numeric_cols)),
        "written": True,
    }


def export_label_pairs(df: pd.DataFrame, predictor_cols: list[str], output_dir: Path) -> list[dict]:
    results = []

    for class_a, class_b in LABEL_PAIRS:
        fname = f"binary_label_{slugify(class_a)}_vs_{slugify(class_b)}.csv"
        res = export_binary_dataset(
            df=df,
            predictor_cols=predictor_cols,
            class_col="LABEL",
            class_a=class_a,
            class_b=class_b,
            out_path=output_dir / fname,
        )
        results.append(res)

    return results


def export_haplogroup_pairs(df: pd.DataFrame, predictor_cols: list[str], output_dir: Path) -> list[dict]:
    """
    Use only rows from LABEL == 'A. torrentium - NCD'.
    Export all haplogroup pairs with at least MIN_HAPLOGROUP_N rows per class.
    """
    results = []

    ncd = df[df["LABEL"] == "A. torrentium - NCD"].copy()
    if ncd.empty:
        print("No rows for LABEL == 'A. torrentium - NCD'. Skipping haplogroup exports.")
        return results

    hap_counts = (
        ncd["Haplogroup"]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
    )

    valid_haps = hap_counts[hap_counts >= MIN_HAPLOGROUP_N].index.tolist()

    print("\nNCD haplogroup counts:")
    print(hap_counts.to_string())
    print(f"\nValid haplogroups for pairwise export (n >= {MIN_HAPLOGROUP_N}): {valid_haps}")

    for hap_a, hap_b in combinations(valid_haps, 2):
        fname = f"binary_haplogroup_ncd_{slugify(hap_a)}_vs_{slugify(hap_b)}.csv"
        res = export_binary_dataset(
            df=ncd,
            predictor_cols=predictor_cols,
            class_col="Haplogroup",
            class_a=hap_a,
            class_b=hap_b,
            out_path=output_dir / fname,
        )
        results.append(res)

    return results


def write_readme(output_dir: Path, summary: dict) -> None:
    lines = []
    lines.append("Binary lineage datasets exported from master_lineage.xlsx")
    lines.append("")
    lines.append("Target columns:")
    lines.append("- y: binary numeric target")
    lines.append("- label: original class label")
    lines.append("")
    lines.append("Encoding rule:")
    lines.append("- y = 0 for class_0")
    lines.append("- y = 1 for class_1")
    lines.append("")
    lines.append("Predictors:")
    lines.append("- numeric environmental variables only")
    lines.append("- mainly l_*/u_* climate, topography, soil, and land-cover predictors")
    lines.append("")
    lines.append("Files written:")
    lines.append("")

    for section_name in ["label_pairs", "haplogroup_pairs"]:
        lines.append(f"[{section_name}]")
        for item in summary[section_name]:
            if item["written"]:
                lines.append(
                    f"- {item['file']}: "
                    f"class_0='{item['class_0']}', class_1='{item['class_1']}', "
                    f"rows={item['n_rows']}, predictors={item['n_predictors']}"
                )
            else:
                lines.append(
                    f"- {item['file']}: NOT WRITTEN ({item.get('reason', 'unknown')})"
                )
        lines.append("")

    (output_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    df = load_master_table(INPUT_XLSX)

    required_cols = {"LABEL", "Haplogroup"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    predictor_cols = get_predictor_columns(df)
    print(f"Found {len(predictor_cols)} predictor columns before numeric coercion.")

    df = coerce_predictors_to_numeric(df, predictor_cols)

    output_dir = INPUT_XLSX.parent / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput folder: {output_dir}\n")

    label_results = export_label_pairs(df, predictor_cols, output_dir)
    hap_results = export_haplogroup_pairs(df, predictor_cols, output_dir)

    summary = {
        "input_file": str(INPUT_XLSX),
        "output_dir": str(output_dir),
        "n_predictor_candidates": len(predictor_cols),
        "label_pairs": label_results,
        "haplogroup_pairs": hap_results,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_readme(output_dir, summary)

    print("\nDone.")
    print(f"Summary written to: {output_dir / 'summary.json'}")
    print(f"README written to:  {output_dir / 'README.txt'}")


if __name__ == "__main__":
    main()