"""Academic experiment pipeline for model evaluation, interpretable learning, and rule induction.

This script is intentionally designed for clarity and reasoning, not for performance tuning.
It produces a narrative-style log with sectioned interpretations that can be used in coursework.
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text




DatasetBundle = Dict[str, Any]


def should_plot(enable_plots: bool) -> bool:
    """Return whether plotting is enabled for this run."""
    return enable_plots


def print_header(title: str) -> None:
    """Print a visible section header to keep logs readable."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names so downstream code can use consistent identifiers."""
    clean = df.copy()
    clean.columns = clean.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    return clean


def clean_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight cleaning to expected coursework columns when present."""
    cleaned = df.copy()

    if "age" in cleaned.columns:
        cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce")
        cleaned["age"] = cleaned["age"].where(cleaned["age"].between(15, 100))

    if "attendance" in cleaned.columns:
        cleaned["attendance"] = pd.to_numeric(cleaned["attendance"], errors="coerce")
        # Accept percentages in [0, 100] and normalized values in (0, 1].
        normalized_mask = (cleaned["attendance"] > 0) & (cleaned["attendance"] <= 1)
        cleaned.loc[normalized_mask, "attendance"] = cleaned.loc[normalized_mask, "attendance"] * 100
        cleaned["attendance"] = cleaned["attendance"].where(cleaned["attendance"].between(0, 100))

    if "personal_work" in cleaned.columns:
        cleaned["personal_work"] = pd.to_numeric(cleaned["personal_work"], errors="coerce")
        cleaned["personal_work"] = cleaned["personal_work"].where(
            cleaned["personal_work"].between(0, 5)
        )

    return cleaned

def load_and_prepare_dataset(path: Path, dataset_name: str) -> DatasetBundle:
    """Load an Excel dataset, validate target, and split into features/target.

    This function does not fit any transformer and therefore avoids leakage by construction.
    """
    df = pd.read_excel(path)
    df = normalize_columns(df)
    df = clean_known_columns(df)

    if "success" not in df.columns:
        raise ValueError(f"Dataset '{dataset_name}' does not include required target column 'success'.")

    # Keep only rows with known Pass/Fail labels.
    valid_target = df["success"].astype(str).str.strip()
    mask = valid_target.isin(["Pass", "Fail"])
    filtered = df.loc[mask].copy()
    filtered["success"] = valid_target.loc[mask]

    # Explicitly separate predictors from the target.
    x = filtered.drop(columns=["success"])
    y = filtered["success"]

    return {"name": dataset_name, "raw": filtered, "features": x, "target": y}


def build_preprocessor(x: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build preprocessing that handles mixed data types.

    Numeric columns are passed through.
    Categorical columns are one-hot encoded with unknown categories ignored.
    """
    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols




def make_one_hot_encoder_dense() -> OneHotEncoder:
    """Create a dense OneHotEncoder compatible with multiple scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, object]:
    """Compute evaluation metrics in a single, reusable place."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_pass": precision_score(y_true, y_pred, pos_label="Pass", zero_division=0),
        "recall_pass": recall_score(y_true, y_pred, pos_label="Pass", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["Pass", "Fail"]),
    }


def print_metric_block(name: str, metrics: Dict[str, object]) -> None:
    """Print metrics with surrounding interpretation-oriented labels."""
    print(f"\n{name} metrics:")
    print(f"- Accuracy:   {metrics['accuracy']:.4f}")
    print(f"- Precision (Pass): {metrics['precision_pass']:.4f}")
    print(f"- Recall (Pass):    {metrics['recall_pass']:.4f}")
    print("- Confusion matrix [rows=true Pass/Fail, cols=pred Pass/Fail]:")
    print(metrics["confusion_matrix"])


def explain_split_assumptions(y: pd.Series) -> None:
    """Print why stratified split is used and what deployment assumptions it implies."""
    class_balance = y.value_counts(normalize=True)
    print("Why stratification is used:")
    print(
        "- Stratification keeps Pass/Fail class proportions similar in both train and test sets, "
        "reducing accidental optimism/pessimism caused by random imbalance."
    )
    print("Observed global label proportions:")
    for label, proportion in class_balance.items():
        print(f"  * {label}: {proportion:.2%}")

    print("Assumptions implied by this split for deployment:")
    print("- Future student cohorts are sampled from a similar distribution to the current dataset.")
    print("- Labeling policy for 'Pass'/'Fail' remains stable over time.")
    print("- Features available at prediction time match those present in training.")


def explain_baseline_harm(metrics: Dict[str, object], majority_class: str) -> None:
    """Interpret baseline behavior in educational terms."""
    print("Interpretation of the majority-class baseline:")
    print(
        f"- This model always predicts '{majority_class}', which can produce non-trivial accuracy "
        "if classes are imbalanced."
    )
    print("- It discards all student-specific evidence (attendance, study habits, background, etc.).")
    print(
        "- In education, this is harmful for intervention: at-risk students can be ignored if the "
        "majority class is Pass, or many capable students may be mischaracterized if majority is Fail."
    )
    print(
        "- The confusion matrix exposes this collapse: one predicted class receives all predictions, "
        "revealing zero discriminatory reasoning."
    )


def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    """Return transformed feature names after fitting preprocessing."""
    return preprocessor.get_feature_names_out()


def extract_readable_rules(
    tree_model: DecisionTreeClassifier,
    feature_names: Sequence[str],
    max_rules: int = 4,
) -> List[str]:
    """Extract a compact subset of rules as text from a decision tree."""
    raw_rules = export_text(tree_model, feature_names=list(feature_names))
    # Keep only a short subset to remain interpretable in printed log.
    lines = raw_rules.splitlines()
    selected = lines[: min(len(lines), 35)]
    compact = "\n".join(selected)
    return [compact] if compact else []


def root_and_unused_features(
    tree_model: DecisionTreeClassifier,
    feature_names: Sequence[str],
    top_k: int = 6,
) -> Tuple[List[str], List[str]]:
    """Identify features close to root and those never used by the tree."""
    used_indices = tree_model.tree_.feature
    used_indices = used_indices[used_indices >= 0]
    used_features = [feature_names[i] for i in used_indices]

    root_to_near_root = []
    for idx in used_indices[:top_k]:
        fname = feature_names[idx]
        if fname not in root_to_near_root:
            root_to_near_root.append(fname)

    never_used = [f for f in feature_names if f not in set(used_features)]
    return root_to_near_root, never_used


def analyze_error_context(cm: np.ndarray) -> None:
    """Interpret dominant error type for educational usage contexts."""
    # Matrix layout: rows true [Pass, Fail], cols pred [Pass, Fail]
    false_fail = int(cm[0, 1])  # true Pass predicted Fail
    false_pass = int(cm[1, 0])  # true Fail predicted Pass

    print("Evaluation interpretation beyond accuracy:")
    if false_fail > false_pass:
        print(
            "- Dominant error type: false fail (students who would pass are predicted as fail)."
        )
    elif false_pass > false_fail:
        print(
            "- Dominant error type: false pass (at-risk students are predicted as pass)."
        )
    else:
        print("- Dominant error type: balanced; both error types occur at similar rates.")

    print(
        "- For academic support and early warning, false pass is usually more critical because "
        "students needing help might be missed."
    )
    print(
        "- For administrative decisions (e.g., restrictive actions), false fail can be ethically "
        "serious because students may receive unnecessary negative flags."
    )
    print(
        "- Therefore, metric priority shifts by task: recall of at-risk students for interventions, "
        "precision and fairness checks for high-stakes administrative use."
    )


def explain_naive_bayes_assumptions() -> None:
    """Print conceptual interpretation of Naive Bayes assumptions."""
    print("Naive Bayes assumption analysis:")
    print(
        "- It assumes conditional independence: once class is known, each feature contributes "
        "independently to likelihood."
    )
    print(
        "- In student data this is usually violated (e.g., attendance and study behavior are correlated), "
        "yet the classifier can still rank classes effectively."
    )
    print(
        "- It cannot represent rich feature interactions directly (e.g., 'low attendance AND low personal_work' "
        "as a joint pattern unless encoded explicitly)."
    )
    print(
        "- Probability calibration matters because support decisions may use confidence thresholds; "
        "uncalibrated probabilities can over/understate certainty."
    )


def compare_models_textually(tree_metrics: Dict[str, object], nb_metrics: Dict[str, object]) -> None:
    """Provide assumption-conditional comparison without declaring an absolute winner."""
    print("Comparative interpretation (without absolute winner):")
    print("- Predictive behavior:")
    print(
        f"  * Decision Tree accuracy={tree_metrics['accuracy']:.4f}, "
        f"precision={tree_metrics['precision_pass']:.4f}, recall={tree_metrics['recall_pass']:.4f}."
    )
    print(
        f"  * Naive Bayes accuracy={nb_metrics['accuracy']:.4f}, "
        f"precision={nb_metrics['precision_pass']:.4f}, recall={nb_metrics['recall_pass']:.4f}."
    )
    print("- Robustness:")
    print(
        "  * Naive Bayes is often more stable with small data due to strong assumptions."
    )
    print(
        "  * Decision trees are sensitive to small data perturbations and can change split structure."
    )
    print("- Explainability:")
    print(
        "  * Decision trees provide explicit rule paths that are easier to audit case-by-case."
    )
    print(
        "  * Naive Bayes explains via additive likelihood contributions, which is simpler globally but "
        "less rule-like for human review."
    )
    print("- Bias amplification:")
    print(
        "  * Trees may encode sharp threshold effects that mirror dataset biases in explicit rules."
    )
    print(
        "  * Naive Bayes can still propagate bias through correlated proxy variables even under its "
        "independence simplification."
    )
    print("Assumption-conditional conclusion:")
    print(
        "- Prefer a decision tree when transparent local decision rules are required and stakeholders "
        "accept threshold-style logic."
    )
    print(
        "- Prefer Naive Bayes when data is limited, fast baseline probabilistic reasoning is desired, "
        "and interaction modeling is not central."
    )


def compute_rule_support(df_features: pd.DataFrame, conditions: Iterable[Tuple[str, str, float]]) -> float:
    """Compute support of a simple conjunction rule on original (non-encoded) feature space."""
    if df_features.empty:
        return 0.0
    mask = pd.Series(True, index=df_features.index)
    for col, op, threshold in conditions:
        if col not in df_features.columns:
            return 0.0
        if op == "<=":
            mask &= pd.to_numeric(df_features[col], errors="coerce") <= threshold
        elif op == ">":
            mask &= pd.to_numeric(df_features[col], errors="coerce") > threshold
    return float(mask.mean())




def plot_dataset_label_distributions(ds1: DatasetBundle, ds2: DatasetBundle) -> None:
    """Plot side-by-side label distributions for both datasets."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ds1_counts = ds1["target"].value_counts().reindex(["Pass", "Fail"], fill_value=0)
    ds2_counts = ds2["target"].value_counts().reindex(["Pass", "Fail"], fill_value=0)

    axes[0].bar(ds1_counts.index, ds1_counts.values, color=["#4daf4a", "#e41a1c"])
    axes[0].set_title(f"{ds1['name']} label distribution")
    axes[0].set_ylabel("Count")

    axes[1].bar(ds2_counts.index, ds2_counts.values, color=["#4daf4a", "#e41a1c"])
    axes[1].set_title(f"{ds2['name']} label distribution")

    fig.suptitle("Graphical view: Pass/Fail balance across datasets")
    plt.tight_layout()
    plt.show()


def plot_model_metric_comparison(
    baseline_metrics: Dict[str, object],
    tree_metrics: Dict[str, object],
    nb_metrics: Dict[str, object],
) -> None:
    """Graphically compare core metrics across all three models."""
    import matplotlib.pyplot as plt

    labels = ["Accuracy", "Precision(Pass)", "Recall(Pass)"]
    baseline_values = [
        float(baseline_metrics["accuracy"]),
        float(baseline_metrics["precision_pass"]),
        float(baseline_metrics["recall_pass"]),
    ]
    tree_values = [
        float(tree_metrics["accuracy"]),
        float(tree_metrics["precision_pass"]),
        float(tree_metrics["recall_pass"]),
    ]
    nb_values = [
        float(nb_metrics["accuracy"]),
        float(nb_metrics["precision_pass"]),
        float(nb_metrics["recall_pass"]),
    ]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, baseline_values, width=width, label="Baseline", color="#999999")
    ax.bar(x, tree_values, width=width, label="Decision Tree", color="#377eb8")
    ax.bar(x + width, nb_values, width=width, label="Naive Bayes", color="#ff7f00")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Graphical model comparison (no absolute winner claim)")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_rule_supports(accepted_support: float, rejected_low_support: float, rejected_ethics_support: float) -> None:
    """Visualize support values for accepted/rejected epistemic rules."""
    import matplotlib.pyplot as plt

    labels = ["Accepted rule", "Rejected (low support)", "Rejected (ethical risk)"]
    values = [accepted_support, rejected_low_support, rejected_ethics_support]
    colors = ["#4daf4a", "#e41a1c", "#984ea3"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Support")
    ax.set_title("Rule-support visualization for epistemic selection")

    for i, value in enumerate(values):
        ax.text(i, value + 0.01, f"{value:.2%}", ha="center")

    plt.tight_layout()
    plt.show()

def run_pipeline(dataset1_path: Path, dataset2_path: Path, enable_plots: bool = True) -> None:
    """Run the full practical coursework experiment from start to finish."""

    # ------------------------------------------------------------------
    # STEP 1: DATA LOADING AND PREPARATION
    # ------------------------------------------------------------------
    print_header("STEP 1 - DATA LOADING AND PREPARATION")
    ds1 = load_and_prepare_dataset(dataset1_path, "studentdata1")
    ds2 = load_and_prepare_dataset(dataset2_path, "studentdata2")

    print(f"Loaded {ds1['name']}: shape={ds1['raw'].shape}")
    print(f"Loaded {ds2['name']}: shape={ds2['raw'].shape}")
    print("Features and target are explicitly separated with target='success'.")
    print("Categorical encoding is performed inside a train-fitted preprocessing pipeline.")
    print("Leakage prevention: encoder is fit on training data only, then applied to test data.")

    # Graphical requirement: show class-balance overview for both datasets.
    if should_plot(enable_plots):
        plot_dataset_label_distributions(ds1, ds2)

    # Main supervised workflow uses dataset2.
    x = ds2["features"]
    y = ds2["target"]

    # ------------------------------------------------------------------
    # STEP 2: TRAIN-TEST SPLIT
    # ------------------------------------------------------------------
    print_header("STEP 2 - STRATIFIED TRAIN-TEST SPLIT")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    print(f"Train size: {x_train.shape[0]} rows | Test size: {x_test.shape[0]} rows")
    explain_split_assumptions(y)

    # Build shared preprocessing object (fitted later per model workflow).
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(x_train)
    print(f"Detected numeric columns: {numeric_cols}")
    print(f"Detected categorical columns: {categorical_cols}")

    # ------------------------------------------------------------------
    # STEP 3: BASELINE CLASSIFIER (NULL HYPOTHESIS)
    # ------------------------------------------------------------------
    print_header("STEP 3 - BASELINE CLASSIFIER (MAJORITY CLASS)")
    baseline_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    baseline_pipeline.fit(x_train, y_train)
    baseline_pred = baseline_pipeline.predict(x_test)
    baseline_metrics = evaluate_predictions(y_test, baseline_pred)
    print_metric_block("Baseline (Null Hypothesis)", baseline_metrics)
    majority_class = y_train.mode().iloc[0]
    explain_baseline_harm(baseline_metrics, majority_class)

    # ------------------------------------------------------------------
    # STEP 4: DECISION TREE MODEL
    # ------------------------------------------------------------------
    print_header("STEP 4 - DECISION TREE MODEL")
    tree_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_split=8,
                    random_state=42,
                ),
            ),
        ]
    )
    tree_pipeline.fit(x_train, y_train)
    print("DecisionTreeClassifier trained with explicit max_depth=4 and min_samples_split=8.")
    print("No hyperparameter optimization is performed by design.")
    print("Depth interpretation:")
    print("- Increasing depth increases expressive power but risks memorizing idiosyncrasies.")
    print("- Shallower trees are easier to interpret, audit, and communicate to educators.")
    print("- Trade-off: memorization capacity vs transparency and stability.")

    # ------------------------------------------------------------------
    # STEP 5: EVALUATION BEYOND ACCURACY
    # ------------------------------------------------------------------
    print_header("STEP 5 - EVALUATION BEYOND ACCURACY")
    tree_pred = tree_pipeline.predict(x_test)
    tree_metrics = evaluate_predictions(y_test, tree_pred)
    print_metric_block("Decision Tree", tree_metrics)
    analyze_error_context(tree_metrics["confusion_matrix"])

    # Plot confusion matrix for visual interpretability using matplotlib.
    if should_plot(enable_plots):
        import matplotlib.pyplot as plt

        disp = ConfusionMatrixDisplay(
            confusion_matrix=tree_metrics["confusion_matrix"],
            display_labels=["Pass", "Fail"],
        )
        disp.plot(cmap="Blues")
        plt.title("Decision Tree Confusion Matrix")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # STEP 6: DECISION TREE AS RULE SYSTEM
    # ------------------------------------------------------------------
    print_header("STEP 6 - DECISION TREE AS RULE SYSTEM")
    trained_prep = tree_pipeline.named_steps["prep"]
    trained_tree = tree_pipeline.named_steps["model"]
    feature_names = get_feature_names(trained_prep)

    extracted_rules = extract_readable_rules(trained_tree, feature_names)
    if extracted_rules:
        print("Extracted human-readable rule subset (tree text excerpt):")
        print(extracted_rules[0])

    near_root, ignored = root_and_unused_features(trained_tree, feature_names)
    print("\nVariables appearing near the root (high structural influence):")
    print(near_root if near_root else "None")
    print("Variables ignored by the tree (not used in any split):")
    print(ignored[:20] if ignored else "None")

    print("Rule subset analysis:")
    print(
        "- Plausibility: rules involving attendance/study effort are educationally plausible because "
        "they align with common pedagogical expectations."
    )
    print(
        "- Fairness: if sensitive/proxy attributes appear high in the tree, rules should be audited for "
        "disparate impact before policy use."
    )
    print(
        "- Sensitivity: small data changes can alter split thresholds or branch structure; therefore "
        "individual rules are hypotheses, not immutable truths."
    )

    # ------------------------------------------------------------------
    # STEP 7: NAIVE BAYES MODEL
    # ------------------------------------------------------------------
    print_header("STEP 7 - NAIVE BAYES MODEL")
    # Naive Bayes requires dense numeric array; one-hot output is dense for coursework simplicity.
    nb_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                make_one_hot_encoder_dense(),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    nb_pipeline = Pipeline(
        steps=[
            ("prep", nb_preprocessor),
            ("model", GaussianNB()),
        ]
    )
    nb_pipeline.fit(x_train, y_train)
    nb_pred = nb_pipeline.predict(x_test)
    nb_metrics = evaluate_predictions(y_test, nb_pred)
    print_metric_block("Naive Bayes", nb_metrics)
    explain_naive_bayes_assumptions()

    # ------------------------------------------------------------------
    # STEP 8: COMPARATIVE EVALUATION
    # ------------------------------------------------------------------
    print_header("STEP 8 - COMPARATIVE EVALUATION (TREE VS NAIVE BAYES)")
    compare_models_textually(tree_metrics, nb_metrics)
    if should_plot(enable_plots):
        plot_model_metric_comparison(baseline_metrics, tree_metrics, nb_metrics)

    # ------------------------------------------------------------------
    # STEP 9: RULE INDUCTION ON INDEPENDENT DATA (DISTRIBUTIONAL SHIFT)
    # ------------------------------------------------------------------
    print_header("STEP 9 - RULE INDUCTION ON studentdata1 AS DISTRIBUTIONAL SHIFT")
    x_shift = ds1["features"]
    y_shift = ds1["target"]
    shift_prep, _, _ = build_preprocessor(x_shift)

    shift_pipeline = Pipeline(
        steps=[
            ("prep", shift_prep),
            (
                "model",
                DecisionTreeClassifier(max_depth=3, min_samples_split=10, random_state=7),
            ),
        ]
    )
    shift_pipeline.fit(x_shift, y_shift)

    shift_feature_names = get_feature_names(shift_pipeline.named_steps["prep"])
    shift_tree = shift_pipeline.named_steps["model"]
    shift_rules = extract_readable_rules(shift_tree, shift_feature_names)

    print("Induced rules on independent dataset (studentdata1):")
    if shift_rules:
        print(shift_rules[0])

    shift_root, _ = root_and_unused_features(shift_tree, shift_feature_names)
    overlap = sorted(set(near_root).intersection(set(shift_root)))
    print("Stability/generalization discussion:")
    print(
        f"- Root-feature overlap between dataset2 and dataset1 trees: {overlap if overlap else 'None'}"
    )
    print(
        "- Higher overlap suggests structural stability under distributional shift; low overlap suggests "
        "context-dependent rules."
    )
    print(
        "- Independent-data induction is used here as epistemic stress testing for interpretability and "
        "rule portability."
    )

    # ------------------------------------------------------------------
    # STEP 10: RULE SELECTION AS EPISTEMIC ACT
    # ------------------------------------------------------------------
    print_header("STEP 10 - RULE SELECTION AS AN EPISTEMIC ACT")

    # We explicitly formulate one acceptable and two rejectable candidate rules.
    candidate_accept = [("attendance", ">", 75.0)]
    candidate_reject_low_support = [("personal_work", ">", 4.8), ("attendance", "<=", 20.0)]
    candidate_reject_ethics = [("age", "<=", 18.0)]

    support_accept = compute_rule_support(x_shift, candidate_accept)
    support_low = compute_rule_support(x_shift, candidate_reject_low_support)
    support_ethics = compute_rule_support(x_shift, candidate_reject_ethics)
    if should_plot(enable_plots):
        plot_rule_supports(support_accept, support_low, support_ethics)

    print("ACCEPTED RULE:")
    print("- If attendance > 75, then predict Pass (tentatively accepted).")
    print(
        f"  Justification: support={support_accept:.2%}, educational plausibility is high, and the rule "
        "is understandable for intervention planning."
    )

    print("\nREJECTED RULE 1:")
    print("- If personal_work > 4.8 and attendance <= 20, then predict Fail.")
    print(
        f"  Rejection basis: low support={support_low:.2%}; this can be a dataset-specific artefact and "
        "is too brittle for policy reasoning."
    )

    print("\nREJECTED RULE 2:")
    print("- If age <= 18, then predict Fail.")
    print(
        f"  Rejection basis: support={support_ethics:.2%}; ethically risky and weakly causal. Age-only "
        "rules risk unfair profiling and should not guide high-stakes educational decisions."
    )

    print("\nFinal experiment note:")
    print(
        "- Rule adoption is treated as a knowledge-justification process, not merely a metric-maximization step."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the coursework pipeline for model evaluation and rule induction."
    )
    parser.add_argument("--dataset1", default="studentdata1.xlsx", help="Path to studentdata1.xlsx")
    parser.add_argument("--dataset2", default="studentdata2.xlsx", help="Path to studentdata2.xlsx")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib plots for environments without GUI or matplotlib installation.",
    )
    args = parser.parse_args()

    run_pipeline(Path(args.dataset1), Path(args.dataset2), enable_plots=not args.no_plots)
