"""Student data preparation + EDA pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for consistent downstream handling."""
    renamed = df.copy()
    renamed.columns = (
        renamed.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    return renamed


def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    """Load a dataset from Excel without mutation."""
    raw = pd.read_excel(spec.path)
    return normalize_columns(raw)


def intake_report(df: pd.DataFrame, name: str) -> None:
    """Print a compact intake report for a dataset."""
    print(f"\n===== {name} INTAKE =====")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Column names:", list(df.columns))
    print("Duplicate IDs:", df["id"].duplicated().sum())


def missingness_report(df: pd.DataFrame, name: str) -> None:
    """Report missingness percentages for a dataset."""
    print(f"\n--- Missingness BEFORE cleaning ({name}) ---")
    print((df.isna().mean() * 100).round(2))


def clean_gender(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    normalized = str(value).lower().strip()
    if normalized in {"m", "male"}:
        return "m"
    if normalized in {"f", "female"}:
        return "f"
    return "unknown"


def clean_background(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    normalized = str(value).lower().strip()
    if "stem" in normalized:
        return "stem"
    if "art" in normalized:
        return "arts"
    if "econ" in normalized:
        return "econ"
    if "it" in normalized:
        return "it"
    return "unknown"


def clean_attendance(value: object) -> float:
    if pd.isna(value):
        return np.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    if 0 < numeric <= 1:
        return numeric * 100
    if 0 <= numeric <= 100:
        return numeric
    return np.nan


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize values and enforce basic bounds."""
    cleaned = df.copy()

    cleaned["gender"] = cleaned["gender"].apply(clean_gender)
    cleaned["background"] = cleaned["background"].apply(clean_background)

    cleaned["age"] = cleaned["age"].where(cleaned["age"].between(17, 25))
    cleaned["personal_work"] = cleaned["personal_work"].where(
        cleaned["personal_work"].between(0, 5)
    )
    cleaned["attendance"] = cleaned["attendance"].apply(clean_attendance)

    cleaned["serious"] = cleaned["serious"].where(
        cleaned["serious"].isin(["yes", "no", "sometimes", "rarely"])
    )
    cleaned["success"] = cleaned["success"].where(cleaned["success"].isin(["Pass", "Fail"]))

    return cleaned


def impute_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply pragmatic imputation rules for modeling readiness."""
    imputed = df.copy()

    imputed = imputed.dropna(subset=["success"])

    for col in ["age", "attendance", "personal_work"]:
        imputed[col] = imputed[col].fillna(imputed[col].median())

    imputed["gender"] = imputed["gender"].fillna("unknown")
    imputed["background"] = imputed["background"].fillna("unknown")
    imputed["serious"] = imputed["serious"].fillna(imputed["serious"].mode()[0])
    imputed["course"] = imputed["course"].fillna(imputed["course"].mode()[0])

    return imputed


def validate_dataset(df: pd.DataFrame, name: str) -> None:
    """Print post-imputation validation summary."""
    print(f"\n--- Validation AFTER cleaning ({name}) ---")
    print("Age range:", df["age"].min(), "-", df["age"].max())
    print("Attendance range:", df["attendance"].min(), "-", df["attendance"].max())
    print("Personal work range:", df["personal_work"].min(), "-", df["personal_work"].max())
    print("Remaining missing values (%):")
    print((df.isna().mean() * 100).round(2))


def plot_success(df: pd.DataFrame, title: str) -> None:
    sns.countplot(x="success", data=df)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_hist(df: pd.DataFrame, column: str, title: str) -> None:
    sns.histplot(df[column], bins=20, kde=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_violin(df: pd.DataFrame, column: str, title: str) -> None:
    sns.violinplot(x="success", y=column, data=df)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_quartiles(df: pd.DataFrame, title: str) -> None:
    working = df.copy()
    working["att_bin"] = pd.qcut(working["attendance"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rate = working.groupby("att_bin")["success"].apply(
        lambda series: (series == "Pass").mean() * 100
    )
    rate.plot(kind="bar")
    plt.ylabel("% Pass")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_dataset_overview(df: pd.DataFrame, title: str) -> None:
    """Show EDA for a single dataset in a dedicated view."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(title, fontsize=16)

    sns.countplot(data=df, x="success", ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Success Balance")

    sns.histplot(
        data=df, x="attendance", bins=20, kde=True, ax=axes[0, 1], color="#4c72b0"
    )
    axes[0, 1].set_title("Attendance Distribution")

    sns.histplot(data=df, x="age", bins=20, kde=True, ax=axes[0, 2], color="#55a868")
    axes[0, 2].set_title("Age Distribution")

    sns.histplot(
        data=df, x="personal_work", bins=20, kde=True, ax=axes[1, 0], color="#c44e52"
    )
    axes[1, 0].set_title("Personal Work Distribution")

    sns.violinplot(data=df, x="success", y="attendance", ax=axes[1, 1], palette="Set2")
    axes[1, 1].set_title("Attendance vs Success")

    sns.violinplot(
        data=df, x="success", y="personal_work", ax=axes[1, 2], palette="Set2"
    )
    axes[1, 2].set_title("Personal Work vs Success")

    working = df.copy()
    working["att_bin"] = pd.qcut(working["attendance"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rate = (
        working.groupby("att_bin")["success"]
        .apply(lambda series: (series == "Pass").mean() * 100)
        .reset_index()
    )
    sns.barplot(data=rate, x="att_bin", y="success", ax=axes[2, 0], palette="Blues")
    axes[2, 0].set_ylabel("% Pass")
    axes[2, 0].set_title("Pass Rate by Attendance Quartile")

    sns.countplot(data=df, y="gender", ax=axes[2, 1], palette="pastel")
    axes[2, 1].set_title("Gender Breakdown")
    axes[2, 1].set_xlabel("Count")

    sns.countplot(data=df, y="background", ax=axes[2, 2], palette="pastel")
    axes[2, 2].set_title("Background Breakdown")
    axes[2, 2].set_xlabel("Count")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_comparison(df: pd.DataFrame) -> None:
    """Compare Dataset 1 vs Dataset 2 without mixing them together."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Dataset Comparison", fontsize=16)

    sns.countplot(data=df, x="success", hue="dataset", ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Success Balance by Dataset")

    sns.histplot(
        data=df,
        x="attendance",
        hue="dataset",
        bins=20,
        kde=True,
        element="step",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Attendance Distribution by Dataset")

    sns.histplot(
        data=df,
        x="personal_work",
        hue="dataset",
        bins=20,
        kde=True,
        element="step",
        ax=axes[0, 2],
    )
    axes[0, 2].set_title("Personal Work Distribution by Dataset")

    sns.violinplot(data=df, x="dataset", y="attendance", ax=axes[1, 0], palette="Set2")
    axes[1, 0].set_title("Attendance Spread by Dataset")

    working = df.copy()
    working["att_bin"] = working.groupby("dataset")["attendance"].transform(
        lambda series: pd.qcut(series, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    )
    rate = (
        working.groupby(["dataset", "att_bin"])["success"]
        .apply(lambda series: (series == "Pass").mean() * 100)
        .reset_index()
    )
    sns.barplot(
        data=rate, x="att_bin", y="success", hue="dataset", ax=axes[1, 1], palette="Set2"
    )
    axes[1, 1].set_ylabel("% Pass")
    axes[1, 1].set_title("Pass Rate by Attendance Quartile")

    sns.countplot(data=df, y="course", hue="dataset", ax=axes[1, 2], palette="Set2")
    axes[1, 2].set_title("Course Breakdown by Dataset")
    axes[1, 2].set_xlabel("Count")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def encoding_readiness(df: pd.DataFrame) -> None:
    encoded = pd.get_dummies(df, columns=["gender", "course", "background", "serious"])
    unexpected = [col for col in encoded.columns if "??" in col or "nan" in col.lower()]
    print("\nEncoding quality check – unexpected columns:", unexpected)


def run_pipeline(dataset_specs: list[DatasetSpec]) -> None:
    cleaned_datasets = []

    for spec in dataset_specs:
        raw = load_dataset(spec)
        intake_report(raw, spec.name)
        missingness_report(raw, spec.name)

        cleaned = impute_dataset(clean_dataset(raw))
        validate_dataset(cleaned, spec.name)
        cleaned["dataset"] = spec.name
        cleaned_datasets.append(cleaned)

    combined = pd.concat(cleaned_datasets, ignore_index=True)
    plot_dataset_overview(cleaned_datasets[0], "Dataset 1 Overview")
    plot_dataset_overview(cleaned_datasets[1], "Dataset 2 Overview")
    plot_comparison(combined)
    encoding_readiness(combined)

    print("\nPIPELINE FINISHED SUCCESSFULLY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Student data preparation + EDA pipeline.")
    parser.add_argument("--dataset1", default="studentdata1.xlsx", type=Path)
    parser.add_argument("--dataset2", default="studentdata2.xlsx", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [
        DatasetSpec(name="Dataset 1", path=args.dataset1),
        DatasetSpec(name="Dataset 2", path=args.dataset2),
    ]
    run_pipeline(specs)


if __name__ == "__main__":
    main()
