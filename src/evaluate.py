"""
Evaluation: test-set metrics with bootstrap CIs, baselines, ablation, subgroups,
robustness, shuffled-label control and leakage checks.

Test-set discipline
-------------------
`python -m src.evaluate --split test` scores the test split exactly once.  If
`reports/metrics.json` already exists, the command refuses to run unless
`--force` is passed; a forced rerun increments `test_eval_count` and is
disclosed in the report.

`python -m src.evaluate --split val` is a debug/confirmation run.  It writes
only `reports/metrics_val.json` and never touches the test set.

The saved split indices (`reports/split_indices.npz`) and the trained model
(`reports/final_model.joblib`) are loaded from disk.  Nothing is refit on the
test split.
"""
import argparse
import importlib.metadata
import json
import platform as platform_mod
import warnings
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import (
    APP_DIR,
    CI_LEVEL,
    CV_N_JOBS,
    CV_N_SPLITS,
    DATA_FILE,
    EXAM_AVAILABILITY_COLUMNS,
    EXAM_DISPLAY_NAMES,
    EXAM_UNAVAILABLE_COLUMNS,
    FIGURES_DIR,
    ID_COL,
    MISSING_COLS,
    N_BOOTSTRAP,
    NPV_TARGET,
    PPV_TARGET,
    REPORTS_DIR,
    REQUIRED_EXAMS,
    SEED,
    TARGET_COL,
    TARGET_COLS,
)
from src.features import (
    get_ablation_specs,
    get_feature_columns,
    get_ml_feature_columns,
    prepare_dataframe,
)
from src.predict import (
    get_example_patients,
    load_model_and_config,
    predict_batch,
)
from src.rules import (
    COMPLETE,
    NOT_COMPLETE,
    UNCERTAIN,
    apply_hard_stops,
    report_hard_stop_stats,
    rule_based_protocol,
)
from src.train import _jsonable, build_pipeline, filter_hard_stops

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

DECISION_TO_CODE = {COMPLETE: 1, NOT_COMPLETE: 0, UNCERTAIN: -1}
CODE_TO_DECISION = {1: COMPLETE, 0: NOT_COMPLETE, -1: UNCERTAIN}

B2_FEATURES = [
    "mri_mrTRG",
    "mri_volume_reduction_pct",
    "endo_pct_wall_involved",
    "mri_wall_thickness_mm",
    "mri_DWI_restriction",
    "mri_tumor_volume_ml",
    "dre_finding",
    "endo_finding",
]

PRIMARY_HEADERS = [
    "Population",
    "n",
    "Prevalence",
    "AUROC (95% CI)",
    "AUPRC (95% CI)",
    "Brier (95% CI)",
    "PPV COMPLETE (95% CI)",
    "NPV NOT COMPLETE (95% CI)",
    "Sensitivity",
    "Specificity",
    "False-COMPLETE rate",
    "Coverage",
    "UNCERTAIN",
]

FAMILY_COLORS = {
    "sex": "#1f77b4",
    "age": "#ff7f0e",
    "regimen": "#2ca02c",
    "distance": "#d62728",
    "missingness": "#9467bd",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _fmt_ci(metric, digits=3):
    if metric is None:
        return "n/a"
    if not isinstance(metric, dict):
        return _fmt(metric, digits)
    point = metric.get("point")
    lower = metric.get("ci_lower")
    upper = metric.get("ci_upper")
    if point is None:
        return "n/a"
    if lower is None or upper is None:
        return f"{_fmt(point, digits)} (n/a)"
    return f"{_fmt(point, digits)} ({_fmt(lower, digits)}-{_fmt(upper, digits)})"


def _fmt_pct(value, digits=1):
    if value is None:
        return "n/a"
    return f"{_fmt(value, digits)}%"


def _table(headers, rows):
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _metric_row(label, m):
    return [
        label,
        m["n"],
        _fmt(m["prevalence"], 3),
        _fmt_ci(m["auroc"]),
        _fmt_ci(m["auprc"]),
        _fmt_ci(m["brier"]),
        _fmt_ci(m["ppv_complete"]),
        _fmt_ci(m["npv_not_complete"]),
        _fmt_ci(m["sensitivity_complete"]),
        _fmt_ci(m["specificity_not_complete"]),
        _fmt_ci(m["false_complete_rate"]),
        _fmt_pct(m["coverage_pct"]),
        m["n_uncertain"],
    ]


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def _bootstrap_metric(y, score, metric_fn, n_boot=N_BOOTSTRAP, ci=CI_LEVEL,
                      seed=SEED):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if len(y) == 0:
        return {"point": None, "ci_lower": None, "ci_upper": None}
    try:
        point = float(metric_fn(y, score))
    except Exception:
        point = float("nan")
    if np.isnan(point):
        return {"point": None, "ci_lower": None, "ci_upper": None}

    rng = np.random.RandomState(seed)
    n = len(y)
    values = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            value = float(metric_fn(y[idx], score[idx]))
        except Exception:
            continue
        if not np.isnan(value):
            values.append(value)
    if len(values) < 10:
        return {"point": point, "ci_lower": None, "ci_upper": None}
    alpha = (1 - ci) / 2
    return {
        "point": point,
        "ci_lower": float(np.percentile(values, alpha * 100)),
        "ci_upper": float(np.percentile(values, (1 - alpha) * 100)),
    }


def _bootstrap_decision_metric(y, decisions, metric_fn, n_boot=N_BOOTSTRAP,
                               ci=CI_LEVEL, seed=SEED):
    y = np.asarray(y).astype(int)
    decisions = np.asarray(decisions).astype(int)
    if len(y) == 0:
        return {"point": None, "ci_lower": None, "ci_upper": None}
    try:
        point = float(metric_fn(y, decisions))
    except Exception:
        point = float("nan")
    if np.isnan(point):
        return {"point": None, "ci_lower": None, "ci_upper": None}

    rng = np.random.RandomState(seed)
    n = len(y)
    values = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            value = float(metric_fn(y[idx], decisions[idx]))
        except Exception:
            continue
        if not np.isnan(value):
            values.append(value)
    if len(values) < 10:
        return {"point": point, "ci_lower": None, "ci_upper": None}
    alpha = (1 - ci) / 2
    return {
        "point": point,
        "ci_lower": float(np.percentile(values, alpha * 100)),
        "ci_upper": float(np.percentile(values, (1 - alpha) * 100)),
    }


def _ppv(y, d):
    mask = d == 1
    if not mask.any():
        return np.nan
    return float(y[mask].mean())


def _npv(y, d):
    mask = d == 0
    if not mask.any():
        return np.nan
    return float((1 - y[mask]).mean())


def _sensitivity(y, d):
    mask = y == 1
    if not mask.any():
        return np.nan
    return float((d[mask] == 1).mean())


def _specificity(y, d):
    mask = y == 0
    if not mask.any():
        return np.nan
    return float((d[mask] == 0).mean())


def _false_complete_rate(y, d):
    mask = y == 0
    if not mask.any():
        return np.nan
    return float(((d == 1) & mask).sum() / mask.sum())


def _false_not_complete_rate(y, d):
    mask = y == 1
    if not mask.any():
        return np.nan
    return float(((d == 0) & mask).sum() / mask.sum())


def _coverage(y, d):
    return float((d != -1).mean())


def _definite_accuracy(y, d):
    return float((d == y).mean())


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------
def decisions_to_codes(decisions):
    return np.array(
        [DECISION_TO_CODE.get(str(d), -1) for d in decisions], dtype=int
    )


def decide_codes(probabilities, t_low, t_high):
    probabilities = np.asarray(probabilities, dtype=float)
    codes = np.full(len(probabilities), -1, dtype=int)
    codes[probabilities >= t_high] = 1
    codes[probabilities <= t_low] = 0
    return codes


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_decision_metrics(y, probabilities, decisions, prefix=""):
    """All key metrics with 95% bootstrap CIs for one population."""
    y = np.asarray(y).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    decisions = np.asarray(decisions).astype(int)
    score = np.nan_to_num(probabilities, nan=0.0)

    metrics = {
        "n": int(len(y)),
        "n_safe": int((y == 1).sum()),
        "n_not_safe": int((y == 0).sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "thresholds": None,
    }

    if len(y) and len(np.unique(y)) == 2:
        metrics["auroc"] = _bootstrap_metric(
            y, score, roc_auc_score, seed=SEED + 1
        )
        metrics["auprc"] = _bootstrap_metric(
            y, score, average_precision_score, seed=SEED + 2
        )
        metrics["brier"] = _bootstrap_metric(
            y, score, brier_score_loss, seed=SEED + 3
        )
    else:
        metrics["auroc"] = {"point": None, "ci_lower": None, "ci_upper": None}
        metrics["auprc"] = {"point": None, "ci_lower": None, "ci_upper": None}
        metrics["brier"] = {"point": None, "ci_lower": None, "ci_upper": None}

    metrics["ppv_complete"] = _bootstrap_decision_metric(
        y, decisions, _ppv, seed=SEED + 4
    )
    metrics["npv_not_complete"] = _bootstrap_decision_metric(
        y, decisions, _npv, seed=SEED + 5
    )
    metrics["sensitivity_complete"] = _bootstrap_decision_metric(
        y, decisions, _sensitivity, seed=SEED + 6
    )
    metrics["specificity_not_complete"] = _bootstrap_decision_metric(
        y, decisions, _specificity, seed=SEED + 7
    )
    metrics["false_complete_rate"] = _bootstrap_decision_metric(
        y, decisions, _false_complete_rate, seed=SEED + 8
    )
    metrics["false_not_complete_rate"] = _bootstrap_decision_metric(
        y, decisions, _false_not_complete_rate, seed=SEED + 9
    )
    metrics["coverage"] = _bootstrap_decision_metric(
        y, decisions, _coverage, seed=SEED + 10
    )
    metrics["definite_accuracy"] = _bootstrap_decision_metric(
        y, decisions, _definite_accuracy, seed=SEED + 11
    )

    n_complete = int((decisions == 1).sum())
    n_not_complete = int((decisions == 0).sum())
    n_uncertain = int((decisions == -1).sum())
    metrics["n_complete"] = n_complete
    metrics["n_not_complete"] = n_not_complete
    metrics["n_uncertain"] = n_uncertain
    metrics["coverage_pct"] = (
        round((n_complete + n_not_complete) / len(y) * 100, 1) if len(y) else None
    )
    metrics["n_false_complete"] = int(((decisions == 1) & (y == 0)).sum())
    metrics["n_missed_complete"] = int(((decisions == 0) & (y == 1)).sum())

    cm = np.zeros((3, 2), dtype=int)
    for row_code, row_i in ((1, 0), (0, 1), (-1, 2)):
        for y_value, col_i in ((0, 0), (1, 1)):
            cm[row_i, col_i] = int(
                ((decisions == row_code) & (y == y_value)).sum()
            )
    metrics["confusion_matrix_3x2"] = cm.tolist()
    metrics["confusion_matrix_3x2_rows"] = [COMPLETE, NOT_COMPLETE, UNCERTAIN]
    metrics["confusion_matrix_3x2_cols"] = ["not_safe_y0", "safe_y1"]

    binary_pred = (score >= 0.5).astype(int)
    metrics["confusion_matrix_at_0_5"] = confusion_matrix(
        y, binary_pred, labels=[0, 1]
    ).tolist()

    metrics["decision_counts"] = {
        COMPLETE: n_complete,
        NOT_COMPLETE: n_not_complete,
        UNCERTAIN: n_uncertain,
    }
    return metrics


def compute_rule_metrics(y, predictions, prefix="b1_"):
    predictions = np.asarray(predictions).astype(int)
    return compute_decision_metrics(
        y, predictions.astype(float), predictions, prefix=prefix
    )


def compute_b0_metrics(y, majority_class):
    y = np.asarray(y).astype(int)
    predictions = np.full(len(y), int(majority_class), dtype=int)
    return compute_decision_metrics(
        y, predictions.astype(float), predictions, prefix="b0_"
    )


def _decision_counts(y, decisions):
    y = np.asarray(y).astype(int)
    decisions = np.asarray(decisions).astype(int)
    n = len(y)
    n_false_complete = int(((decisions == 1) & (y == 0)).sum())
    n_missed_complete = int(((decisions == 0) & (y == 1)).sum())
    return {
        "n": int(n),
        "n_complete": int((decisions == 1).sum()),
        "n_not_complete": int((decisions == 0).sum()),
        "n_uncertain": int((decisions == -1).sum()),
        "n_false_complete": n_false_complete,
        "n_missed_complete": n_missed_complete,
        "n_confident_wrong": n_false_complete + n_missed_complete,
        "n_false_complete_rate": (
            float(n_false_complete / (y == 0).sum()) if (y == 0).any() else None
        ),
    }


def _mean_std(values):
    clean = [
        float(v)
        for v in values
        if v is not None
        and not (isinstance(v, float) and np.isnan(v))
        and np.isfinite(float(v))
    ]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "n": len(clean),
    }


def _summarize_runs(runs):
    keys = [
        "auroc",
        "mean_abs_prob_shift",
        "n_uncertain",
        "n_decision_changed_model",
        "n_confident_wrong",
        "n_false_complete",
        "n_uncertain_deployed",
        "n_decision_changed_deployed",
        "n_confident_wrong_deployed",
        "n_false_complete_deployed",
        "n_policy_uncertain",
    ]
    return {key: _mean_std([run.get(key) for run in runs]) for key in keys}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_calibration(y, probabilities, t_low, t_high, brier_point, save_path):
    y = np.asarray(y).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    try:
        prob_true, prob_pred = calibration_curve(
            y, probabilities, n_bins=10, strategy="uniform"
        )
    except Exception:
        prob_true, prob_pred = calibration_curve(
            y, probabilities, n_bins=5, strategy="quantile"
        )
    ax1.plot(prob_pred, prob_true, "o-", color="#1f77b4", label="Model")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed fraction of safe patients")
    ax1.set_title(f"Calibration (Brier = {_fmt(brier_point, 3)})")
    ax1.legend(loc="best")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    if (y == 1).any():
        ax2.hist(
            probabilities[y == 1],
            bins=20,
            alpha=0.55,
            density=True,
            color="#2ca02c",
            label="Safe (1)",
        )
    if (y == 0).any():
        ax2.hist(
            probabilities[y == 0],
            bins=20,
            alpha=0.55,
            density=True,
            color="#d62728",
            label="Not safe (0)",
        )
    ax2.axvline(
        t_low, color="gray", linestyle="--", linewidth=1,
        label=f"t_low = {t_low:.3f}",
    )
    ax2.axvline(
        t_high, color="gray", linestyle=":", linewidth=1,
        label=f"t_high = {t_high:.3f}",
    )
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Density")
    ax2.set_title("Probability distribution by class")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "ML subset; thresholds selected on OOF training predictions",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y, probabilities, metrics, t_high, save_path):
    y = np.asarray(y).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y, probabilities)
    ax1.plot(
        fpr, tpr, color="#1f77b4", linewidth=2,
        label=f"AUROC = {_fmt_ci(metrics['auroc'])}",
    )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC curve (ML subset)")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    precision, recall, _ = precision_recall_curve(y, probabilities)
    prevalence = float(y.mean())
    ax2.plot(
        recall, precision, color="#d62728", linewidth=2,
        label=f"AUPRC = {_fmt_ci(metrics['auprc'])}",
    )
    ax2.axhline(
        prevalence, color="k", linestyle="--", linewidth=1,
        label=f"Prevalence = {prevalence:.3f}",
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-recall curve (ML subset)")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    complete = probabilities >= t_high
    if (y == 0).any() and (y == 1).any() and complete.any():
        fpr_at = ((complete) & (y == 0)).sum() / (y == 0).sum()
        tpr_at = ((complete) & (y == 1)).sum() / (y == 1).sum()
        ax1.scatter(
            [fpr_at], [tpr_at], marker="*", s=140, color="#ff7f0e",
            zorder=5, label=f"COMPLETE calls (t_high={t_high:.3f})",
        )
        ax1.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_risk_coverage(y, probabilities, t_low, t_high, save_path):
    y = np.asarray(y).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.linspace(max(0.50, t_low + 0.01), 0.99, 50)

    coverages, risks = [], []
    complete_coverages, complete_risks = [], []
    for threshold in thresholds:
        decisions = decide_codes(probabilities, t_low, threshold)
        decided = decisions != -1
        coverages.append(float(decided.mean()))
        risks.append(
            float((decisions[decided] != y[decided]).mean())
            if decided.any() else np.nan
        )
        complete = probabilities >= threshold
        complete_coverages.append(float(complete.mean()))
        complete_risks.append(
            float(((complete) & (y == 0)).sum() / complete.sum())
            if complete.any() else np.nan
        )

    deployed = decide_codes(probabilities, t_low, t_high)
    decided = deployed != -1
    deployed_coverage = float(decided.mean())
    deployed_risk = (
        float((deployed[decided] != y[decided]).mean())
        if decided.any() else np.nan
    )
    deployed_complete = probabilities >= t_high
    deployed_complete_coverage = float(deployed_complete.mean())
    deployed_complete_risk = (
        float(((deployed_complete) & (y == 0)).sum() / deployed_complete.sum())
        if deployed_complete.any() else np.nan
    )

    finite_risks = [r for r in risks if np.isfinite(r)]
    finite_complete = [r for r in complete_risks if np.isfinite(r)]
    ylim1 = max(0.2, max(finite_risks) * 1.2) if finite_risks else 0.2
    ylim2 = (
        max(0.2, max(finite_complete) * 1.2) if finite_complete else 0.2
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(coverages, risks, color="#1f77b4", linewidth=2)
    ax1.scatter(
        [deployed_coverage], [deployed_risk], marker="*", s=160,
        color="#ff7f0e", zorder=5,
        label=f"Deployed (t_high={t_high:.3f})",
    )
    ax1.set_xlabel("Coverage (fraction with a definite decision)")
    ax1.set_ylabel("Definite-decision error rate")
    ax1.set_title("Risk-coverage (ML subset)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, ylim1)
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)

    ax2.plot(complete_coverages, complete_risks, color="#d62728", linewidth=2)
    ax2.scatter(
        [deployed_complete_coverage], [deployed_complete_risk],
        marker="*", s=160, color="#ff7f0e", zorder=5,
        label=f"Deployed (t_high={t_high:.3f})",
    )
    ax2.axhline(
        1 - PPV_TARGET, color="k", linestyle="--", linewidth=1,
        label=f"Target risk = {1 - PPV_TARGET:.2f}",
    )
    ax2.set_xlabel("COMPLETE-call coverage")
    ax2.set_ylabel("False-COMPLETE rate among COMPLETE calls")
    ax2.set_title("COMPLETE-call risk-coverage (ML subset)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, ylim2)
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"t_low fixed at {t_low:.3f}; t_high varied. "
        f"Deployed point uses t_high={t_high:.3f}.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, save_path):
    cm = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not safe (y=0)", "Safe (y=1)"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([COMPLETE, NOT_COMPLETE, UNCERTAIN])
    ax.set_xlabel("True label")
    ax.set_ylabel("Deployed decision")
    ax.set_title("Deployed decisions vs target (evaluated population)")

    threshold = cm.max() / 2 if cm.max() else 0
    for row in range(3):
        row_total = cm[row].sum()
        for col in range(2):
            pct = (cm[row, col] / row_total * 100) if row_total else 0.0
            ax.text(
                col, row, f"{cm[row, col]}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if cm[row, col] > threshold else "black",
                fontsize=10,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Patients")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(ablation, save_path):
    items = []
    for name, result in ablation.items():
        if result.get("auroc_mean") is None:
            continue
        x = 5.5 if result.get("adds_baseline") else float(result["n_exams"])
        items.append((x, name, result))
    items.sort(key=lambda item: item[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    if items:
        xs = [item[0] for item in items]
        auroc = [item[2]["auroc_mean"] for item in items]
        auroc_std = [item[2]["auroc_std"] for item in items]
        auprc = [item[2]["auprc_mean"] for item in items]
        auprc_std = [item[2]["auprc_std"] for item in items]
        ax.errorbar(
            xs, auroc, yerr=auroc_std, marker="o", capsize=4, linewidth=2,
            color="#1f77b4", label="AUROC",
        )
        ax.errorbar(
            xs, auprc, yerr=auprc_std, marker="s", capsize=4, linewidth=2,
            color="#d62728", label="AUPRC",
        )
        for x, name, result in items:
            ax.annotate(
                name, (x, result["auroc_mean"]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=7,
            )
        ax.axvline(5.25, color="gray", linestyle="--", linewidth=1)
        ax.text(
            5.28, 0.55, "add baseline,\ntreatment,\npost-lab",
            fontsize=8, va="bottom",
        )
        all_values = auroc + auprc
        ax.set_ylim(max(0.0, min(all_values) - 0.08), 1.01)

    ax.set_xticks([1, 2, 3, 4, 5, 5.5])
    ax.set_xticklabels(
        ["MRI", "+Endo", "+DRE", "+Biopsy", "+CT", "+Baseline"]
    )
    ax.set_xlabel("Number of exam groups")
    ax.set_ylabel("Score")
    ax.set_title(
        "Ablation: performance vs number of exams "
        "(5x5 CV on training ML split)"
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_subgroups(subgroups, overall_ml_auroc, save_path):
    items = [
        (key, value)
        for key, value in subgroups["subgroups"].items()
        if value.get("auroc_ml") is not None
    ]
    if not items:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, "No subgroup had enough data for AUROC.",
            ha="center", va="center",
        )
        ax.axis("off")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    items.sort(key=lambda item: (item[1]["family"], item[0]))
    labels = [item[0] for item in items]
    values = [item[1]["auroc_ml"] for item in items]
    colors = [
        FAMILY_COLORS.get(item[1]["family"], "#7f7f7f") for item in items
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(items))
    ax.bar(x, values, color=colors, alpha=0.85)
    if overall_ml_auroc is not None:
        ax.axhline(
            overall_ml_auroc, color="k", linestyle="--", linewidth=1,
            label=f"Overall ML AUROC = {overall_ml_auroc:.3f}",
        )
    for i, (_, value) in enumerate(items):
        ax.text(
            i, value["auroc_ml"] + 0.005, f"n={value['n_ml']}",
            ha="center", va="bottom", fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ML AUROC")
    ax.set_ylim(max(0.0, min(values) - 0.08), 1.01)
    ax.set_title(
        "ML AUROC by subgroup (evaluated ML subset; "
        "small subgroups are exploratory)"
    )
    families = sorted({item[1]["family"] for item in items})
    handles = [
        Patch(color=FAMILY_COLORS.get(family, "#7f7f7f"), label=family)
        for family in families
    ]
    if overall_ml_auroc is not None:
        handles.append(
            plt.Line2D([0], [0], color="k", linestyle="--", label="Overall")
        )
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(robustness, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.38

    exams = list(robustness["drop_exam"].keys())
    x = np.arange(len(exams))
    model_false_complete = [
        robustness["drop_exam"][exam]["model_only"]["n_false_complete"]
        for exam in exams
    ]
    deployed_false_complete = [
        robustness["drop_exam"][exam]["deployed"]["n_false_complete"]
        for exam in exams
    ]
    ax1.bar(
        x - width / 2, model_false_complete, width,
        label="Model only (no policy)", color="#d62728",
    )
    ax1.bar(
        x + width / 2, deployed_false_complete, width,
        label="Deployed policy", color="#2ca02c",
    )
    for i, exam in enumerate(exams):
        result = robustness["drop_exam"][exam]
        ax1.text(
            i - width / 2, model_false_complete[i] + 0.05,
            str(model_false_complete[i]), ha="center", va="bottom", fontsize=8,
        )
        ax1.text(
            i + width / 2, deployed_false_complete[i] + 0.05,
            f"{deployed_false_complete[i]}\n"
            f"({result['deployed']['n_uncertain_pct']:.0f}% unc.)",
            ha="center", va="bottom", fontsize=7,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"Drop {exam}" for exam in exams], rotation=20, ha="right"
    )
    ax1.set_ylabel("False COMPLETE calls")
    ax1.set_title("Drop one exam: policy prevents false COMPLETE calls")
    ax1.set_ylim(
        0, max(1, max(model_false_complete + deployed_false_complete) * 1.3)
    )
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3, axis="y")

    scenarios = [("noise", "Feature noise"), ("extra_missingness", "20% missing")]
    x2 = np.arange(len(scenarios))
    model_fc2 = [
        robustness[key]["summary"]["n_false_complete"]["mean"] or 0.0
        for key, _ in scenarios
    ]
    deployed_fc2 = [
        robustness[key]["summary"]["n_false_complete_deployed"]["mean"] or 0.0
        for key, _ in scenarios
    ]
    ax2.bar(
        x2 - width / 2, model_fc2, width,
        label="Model only (no policy)", color="#d62728",
    )
    ax2.bar(
        x2 + width / 2, deployed_fc2, width,
        label="Deployed policy", color="#2ca02c",
    )
    for i, (key, _) in enumerate(scenarios):
        summary = robustness[key]["summary"]
        ax2.text(
            i - width / 2, model_fc2[i] + 0.05, f"{model_fc2[i]:.1f}",
            ha="center", va="bottom", fontsize=8,
        )
        ax2.text(
            i + width / 2, deployed_fc2[i] + 0.05, f"{deployed_fc2[i]:.1f}",
            ha="center", va="bottom", fontsize=8,
        )
        ax2.text(
            i, max(model_fc2[i], deployed_fc2[i]) + 0.6,
            f"unc.: {_fmt(summary['n_uncertain']['mean'], 1)} -> "
            f"{_fmt(summary['n_uncertain_deployed']['mean'], 1)}",
            ha="center", va="bottom", fontsize=7,
        )
    ax2.set_xticks(x2)
    ax2.set_xticklabels([label for _, label in scenarios])
    ax2.set_ylabel("Mean false COMPLETE calls across seeds")
    ax2.set_title("Noise and extra missingness (ML subset, 5 seeds)")
    ax2.set_ylim(0, max(1, max(model_fc2 + deployed_fc2) * 1.5))
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def run_ablation(df, train_idx_ml, model_name, calibration, ml_feature_cols):
    specs = get_ablation_specs()
    results = {}
    train_df = df.iloc[train_idx_ml]
    y = df[TARGET_COL].iloc[train_idx_ml]

    last_spec = specs.get("All 5 exams + baseline")
    if last_spec and set(last_spec["features"]) != set(ml_feature_cols):
        print(
            "  WARNING: ablation's final feature set differs from the "
            "trained model's ml_feature_cols."
        )

    for name, spec in specs.items():
        features = spec["features"]
        print(f"  Ablation: {name} ({len(features)} features)...")
        try:
            X = prepare_dataframe(train_df, features)
            pipe = build_pipeline(features, model_name, calibration)
            aurocs, auprcs, briers = [], [], []
            for repeat in range(5):
                cv = StratifiedKFold(
                    n_splits=CV_N_SPLITS,
                    shuffle=True,
                    random_state=SEED + repeat,
                )
                probabilities = cross_val_predict(
                    pipe, X, y, cv=cv, method="predict_proba",
                    n_jobs=CV_N_JOBS,
                )[:, 1]
                aurocs.append(float(roc_auc_score(y, probabilities)))
                auprcs.append(
                    float(average_precision_score(y, probabilities))
                )
                briers.append(
                    float(brier_score_loss(y, probabilities))
                )
            results[name] = {
                "n_exams": spec["n_exams"],
                "adds_baseline": spec["adds_baseline"],
                "n_features": len(features),
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs)),
                "auprc_mean": float(np.mean(auprcs)),
                "auprc_std": float(np.std(auprcs)),
                "brier_mean": float(np.mean(briers)),
                "brier_std": float(np.std(briers)),
            }
            print(
                f"    AUROC={results[name]['auroc_mean']:.4f}"
                f"+/-{results[name]['auroc_std']:.4f}"
            )
        except Exception as exc:
            print(f"    failed: {exc}")
            results[name] = {
                "n_exams": spec["n_exams"],
                "adds_baseline": spec["adds_baseline"],
                "n_features": len(features),
                "auroc_mean": None,
                "error": str(exc),
            }
    return results


def _subgroup_metrics(
    test_df, test_df_ml, y_full, decisions_full,
    y_ml, probabilities_ml, decisions_ml, mask_fn,
):
    mask_full = mask_fn(test_df).values
    mask_ml = mask_fn(test_df_ml).values
    y_f = y_full[mask_full]
    d_f = decisions_full[mask_full]
    y_m = y_ml[mask_ml]
    p_m = probabilities_ml[mask_ml]

    result = {
        "n_full": int(mask_full.sum()),
        "n_ml": int(mask_ml.sum()),
        "n_safe": int((y_f == 1).sum()),
        "n_not_safe": int((y_f == 0).sum()),
        "prevalence": float(y_f.mean()) if len(y_f) else None,
        "n_complete": int((d_f == 1).sum()),
        "n_not_complete": int((d_f == 0).sum()),
        "n_uncertain": int((d_f == -1).sum()),
        "coverage_pct": (
            round(float((d_f != -1).mean() * 100), 1) if len(y_f) else None
        ),
        "ppv_complete": (
            _bootstrap_decision_metric(
                y_f, d_f, _ppv, n_boot=500, seed=SEED + 21
            ) if len(y_f) else None
        ),
        "false_complete_rate": (
            _bootstrap_decision_metric(
                y_f, d_f, _false_complete_rate, n_boot=500, seed=SEED + 22
            ) if len(y_f) else None
        ),
        "auroc_ml": None,
        "auprc_ml": None,
        "ml_reliable": False,
        "reliable": bool(
            len(y_f) >= 10
            and (y_f == 1).sum() >= 5
            and (y_f == 0).sum() >= 5
        ),
    }
    if len(y_m) >= 5 and len(np.unique(y_m)) == 2:
        result["auroc_ml"] = float(roc_auc_score(y_m, p_m))
        result["auprc_ml"] = float(average_precision_score(y_m, p_m))
        result["ml_reliable"] = bool(
            len(y_m) >= 10
            and (y_m == 1).sum() >= 5
            and (y_m == 0).sum() >= 5
        )
    return result


def run_subgroups(
    df, eval_idx, eval_idx_ml, y_full, decisions_full,
    y_ml, probabilities_ml, decisions_ml, overall_ml_auroc,
):
    test_df = df.iloc[eval_idx]
    test_df_ml = df.iloc[eval_idx_ml]
    subgroups = {}

    def add(key, family, mask_fn):
        if mask_fn(test_df).sum() < 5:
            return
        subgroups[key] = {
            "family": family,
            **_subgroup_metrics(
                test_df, test_df_ml, y_full, decisions_full,
                y_ml, probabilities_ml, decisions_ml, mask_fn,
            ),
        }

    for sex in ["M", "F"]:
        add(
            f"sex={sex}", "sex",
            lambda d, s=sex: d["sex"] == s,
        )

    age_bands = ["<50", "50-65", ">65"]
    for band in age_bands:
        add(
            f"age {band}", "age",
            lambda d, b=band: pd.cut(
                d["age"], bins=[0, 50, 65, 100], labels=age_bands,
                right=False,
            ) == b,
        )

    for regimen in sorted(df["regimen"].dropna().unique()):
        add(
            f"regimen={regimen}", "regimen",
            lambda d, r=regimen: d["regimen"] == r,
        )

    distance_bands = ["<=2 cm", "2-4 cm", ">4 cm"]
    for band in distance_bands:
        add(
            f"distance {band}", "distance",
            lambda d, b=band: pd.cut(
                d["tumor_distance_anal_verge_cm"],
                bins=[0, 2, 4, 10], labels=distance_bands, right=False,
            ) == b,
        )

    add(
        "any missing", "missingness",
        lambda d: d[MISSING_COLS].isnull().any(axis=1),
    )
    add(
        "no missing", "missingness",
        lambda d: ~d[MISSING_COLS].isnull().any(axis=1),
    )

    gaps = {}
    families = sorted({value["family"] for value in subgroups.values()})
    for family in families:
        reliable = [
            value for value in subgroups.values()
            if value["family"] == family and value["reliable"]
        ]
        aurocs = [
            value["auroc_ml"] for value in reliable
            if value.get("auroc_ml") is not None
        ]
        ppvs = [
            value["ppv_complete"]["point"] for value in reliable
            if value.get("ppv_complete")
            and value["ppv_complete"].get("point") is not None
        ]
        coverages = [
            value["coverage_pct"] for value in reliable
            if value.get("coverage_pct") is not None
        ]
        gaps[family] = {
            "n_reliable_subgroups": len(reliable),
            "auroc_gap": (
                float(max(aurocs) - min(aurocs)) if len(aurocs) >= 2 else None
            ),
            "ppv_gap": (
                float(max(ppvs) - min(ppvs)) if len(ppvs) >= 2 else None
            ),
            "coverage_gap": (
                float(max(coverages) - min(coverages))
                if len(coverages) >= 2 else None
            ),
        }

    return {
        "subgroups": subgroups,
        "gaps": gaps,
        "overall_ml_auroc": overall_ml_auroc,
    }


def _predict_proba_df(pipe, frame, ml_feature_cols):
    prepared = prepare_dataframe(frame, ml_feature_cols)
    return pipe.predict_proba(prepared)[:, 1]


def run_robustness(df, train_idx, eval_idx, eval_idx_ml, pipe, config):
    t_low = float(config["t_low"])
    t_high = float(config["t_high"])
    ml_feature_cols = config["ml_feature_cols"]

    test_full = df.iloc[eval_idx].copy()
    test_ml = df.iloc[eval_idx_ml].copy()
    y_full = df[TARGET_COL].iloc[eval_idx].values.astype(int)
    y_ml = df[TARGET_COL].iloc[eval_idx_ml].values.astype(int)
    train_df = df.iloc[train_idx]

    probabilities_base_ml = _predict_proba_df(
        pipe, test_ml, ml_feature_cols
    )
    decisions_base_ml = decide_codes(
        probabilities_base_ml, t_low, t_high
    )
    batch_base_ml = predict_batch(test_ml, pipe, config, log=False)
    decisions_base_deployed_ml = decisions_to_codes(
        batch_base_ml["decision"].values
    )

    results = {"drop_exam": {}, "noise": {}, "extra_missingness": {}}

    # (a) Drop one whole exam at inference.
    for exam in REQUIRED_EXAMS:
        drop_columns = EXAM_UNAVAILABLE_COLUMNS[exam]
        full_drop = test_full.copy()
        for column in drop_columns:
            if column in full_drop.columns:
                full_drop[column] = np.nan

        stops_drop = apply_hard_stops(full_drop)
        eligible = ~stops_drop["hard_stop_fired"].values
        eligible_df = full_drop.loc[eligible]
        y_eligible = y_full[eligible]

        if len(eligible_df):
            probabilities_eligible = _predict_proba_df(
                pipe, eligible_df, ml_feature_cols
            )
            decisions_eligible = decide_codes(
                probabilities_eligible, t_low, t_high
            )
            probabilities_base_eligible = _predict_proba_df(
                pipe, test_full.loc[eligible], ml_feature_cols
            )
            model_only = _decision_counts(
                y_eligible, decisions_eligible
            )
            shift = np.abs(
                probabilities_eligible - probabilities_base_eligible
            )
            finite_shift = np.isfinite(shift)
            model_only["mean_abs_prob_shift"] = (
                float(np.mean(shift[finite_shift]))
                if finite_shift.any() else None
            )
            model_only["auroc"] = (
                float(roc_auc_score(y_eligible, probabilities_eligible))
                if len(np.unique(y_eligible)) == 2 else None
            )
        else:
            model_only = {
                "n": 0, "n_complete": 0, "n_not_complete": 0,
                "n_uncertain": 0, "n_false_complete": 0,
                "n_missed_complete": 0, "n_confident_wrong": 0,
                "n_false_complete_rate": None,
                "mean_abs_prob_shift": None, "auroc": None,
            }

        if exam == "ct":
            original_ct = test_full["ct_distant_progression"].values[
                eligible
            ]
            model_only["n_distant_progression_eligible"] = int(
                (original_ct == 1).sum()
            )
            model_only["n_distant_progression_called_complete"] = int(
                ((original_ct == 1) & (decisions_eligible == 1)).sum()
            ) if len(eligible_df) else 0

        batch_drop = predict_batch(full_drop, pipe, config, log=False)
        decisions_drop = decisions_to_codes(
            batch_drop["decision"].values
        )
        deployed = _decision_counts(y_full, decisions_drop)
        deployed["n_uncertain_pct"] = round(
            float((decisions_drop == -1).mean() * 100), 1
        )
        deployed["n_policy_uncertain"] = int(
            (batch_drop["source"] == "policy").sum()
        )
        deployed["n_hard_stop"] = int(
            (batch_drop["source"] == "rule").sum()
        )

        note = (
            f"Deployed policy forces UNCERTAIN for "
            f"{deployed['n_policy_uncertain']} non-hard-stopped patients; "
            f"deployed COMPLETE calls = {deployed['n_complete']}."
        )
        if exam == "ct":
            note += (
                f" Model-only path would call "
                f"{model_only.get('n_distant_progression_called_complete', 0)} "
                f"of {model_only.get('n_distant_progression_eligible', 0)} "
                f"distant-progression patients COMPLETE."
            )

        results["drop_exam"][exam] = {
            "display_name": EXAM_DISPLAY_NAMES.get(exam, exam),
            "model_only": model_only,
            "deployed": deployed,
            "note": note,
        }

    # (b) Feature noise: DWI (binary flip), ADC and wall thickness (Gaussian).
    seeds = [SEED + i for i in range(5)]
    noise_runs = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        noisy = test_ml.copy()

        for column in ["mri_ADC_mean", "mri_wall_thickness_mm"]:
            if column not in noisy.columns:
                continue
            observed = noisy[column].notna().values
            if not observed.any():
                continue
            training_values = train_df[column].dropna()
            if len(training_values) < 2:
                continue
            sd = float(training_values.std()) * 0.2
            if not np.isfinite(sd) or sd == 0:
                continue
            low = float(training_values.min())
            high = float(training_values.max())
            new_values = np.clip(
                noisy.loc[observed, column].values
                + rng.normal(0, sd, observed.sum()),
                low,
                high,
            )
            noisy.loc[observed, column] = new_values

        for column in ["mri_DWI_restriction"]:
            if column not in noisy.columns:
                continue
            observed = noisy[column].notna().values
            flip = observed & (rng.random(len(noisy)) < 0.1)
            if flip.any():
                noisy.loc[flip, column] = 1 - noisy.loc[flip, column]

        probabilities_noisy = _predict_proba_df(
            pipe, noisy, ml_feature_cols
        )
        decisions_noisy = decide_codes(
            probabilities_noisy, t_low, t_high
        )
        batch_noisy = predict_batch(noisy, pipe, config, log=False)
        decisions_noisy_deployed = decisions_to_codes(
            batch_noisy["decision"].values
        )

        run = _decision_counts(y_ml, decisions_noisy)
        run["auroc"] = (
            float(roc_auc_score(y_ml, probabilities_noisy))
            if len(np.unique(y_ml)) == 2 else None
        )
        run["mean_abs_prob_shift"] = float(
            np.mean(np.abs(probabilities_noisy - probabilities_base_ml))
        )
        run["n_decision_changed_model"] = int(
            (decisions_noisy != decisions_base_ml).sum()
        )
        deployed = _decision_counts(y_ml, decisions_noisy_deployed)
        run["n_uncertain_deployed"] = deployed["n_uncertain"]
        run["n_decision_changed_deployed"] = int(
            (decisions_noisy_deployed != decisions_base_deployed_ml).sum()
        )
        run["n_confident_wrong_deployed"] = deployed["n_confident_wrong"]
        run["n_false_complete_deployed"] = deployed["n_false_complete"]
        noise_runs.append(run)

    results["noise"] = {
        "n_seeds": len(noise_runs),
        "n": int(len(y_ml)),
        "model": (
            "Gaussian noise at 0.2x training SD on mri_ADC_mean and "
            "mri_wall_thickness_mm; 10% label flip on mri_DWI_restriction."
        ),
        "per_seed": noise_runs,
        "summary": _summarize_runs(noise_runs),
        "note": (
            "ML subset. Model-only ignores the missing-exam policy; deployed "
            "applies it. Noise alone does not create missing key exams, so the "
            "policy does not block on noise alone."
        ),
    }

    # (c) Extra missingness: each ML feature missing independently at 20%.
    missing_runs = []
    for seed in seeds:
        rng = np.random.RandomState(seed + 100)
        missing = test_ml.copy()
        for column in ml_feature_cols:
            if column not in missing.columns:
                continue
            mask = rng.random(len(missing)) < 0.2
            missing.loc[mask, column] = np.nan

        probabilities_missing = _predict_proba_df(
            pipe, missing, ml_feature_cols
        )
        decisions_missing = decide_codes(
            probabilities_missing, t_low, t_high
        )
        batch_missing = predict_batch(missing, pipe, config, log=False)
        decisions_missing_deployed = decisions_to_codes(
            batch_missing["decision"].values
        )

        run = _decision_counts(y_ml, decisions_missing)
        run["auroc"] = (
            float(roc_auc_score(y_ml, probabilities_missing))
            if len(np.unique(y_ml)) == 2 else None
        )
        run["mean_abs_prob_shift"] = float(
            np.mean(
                np.abs(probabilities_missing - probabilities_base_ml)
            )
        )
        run["n_decision_changed_model"] = int(
            (decisions_missing != decisions_base_ml).sum()
        )
        deployed = _decision_counts(y_ml, decisions_missing_deployed)
        run["n_uncertain_deployed"] = deployed["n_uncertain"]
        run["n_decision_changed_deployed"] = int(
            (decisions_missing_deployed != decisions_base_deployed_ml).sum()
        )
        run["n_confident_wrong_deployed"] = deployed["n_confident_wrong"]
        run["n_false_complete_deployed"] = deployed["n_false_complete"]
        run["n_policy_uncertain"] = int(
            (batch_missing["source"] == "policy").sum()
        )
        missing_runs.append(run)

    results["extra_missingness"] = {
        "n_seeds": len(missing_runs),
        "n": int(len(y_ml)),
        "model": (
            "Each ML feature is independently set missing with probability 0.2."
        ),
        "per_seed": missing_runs,
        "summary": _summarize_runs(missing_runs),
        "note": (
            "ML subset. The deployed policy forces UNCERTAIN when a key exam "
            "column is missing; the model-only path imputes and continues."
        ),
    }
    return results


def run_shuffled_label_control(
    df, train_idx_ml, val_idx_ml, model_name, calibration,
    ml_feature_cols, n_shuffles=5,
):
    X_train = prepare_dataframe(
        df.iloc[train_idx_ml], ml_feature_cols
    )
    y_train = df[TARGET_COL].iloc[train_idx_ml].values.astype(int)
    X_val = prepare_dataframe(df.iloc[val_idx_ml], ml_feature_cols)
    y_val = df[TARGET_COL].iloc[val_idx_ml].values.astype(int)

    values = []
    for i in range(n_shuffles):
        rng = np.random.RandomState(SEED + 1000 + i)
        shuffled = rng.permutation(y_train)
        pipe = build_pipeline(ml_feature_cols, model_name, calibration)
        try:
            pipe.fit(X_train, shuffled)
            probabilities = pipe.predict_proba(X_val)[:, 1]
            values.append(float(roc_auc_score(y_val, probabilities)))
        except Exception:
            continue

    if not values:
        return {
            "n_shuffles": 0,
            "auroc_mean": None,
            "auroc_std": None,
            "auroc_min": None,
            "auroc_max": None,
            "values": [],
        }
    return {
        "n_shuffles": len(values),
        "auroc_mean": float(np.mean(values)),
        "auroc_std": float(np.std(values)),
        "auroc_min": float(np.min(values)),
        "auroc_max": float(np.max(values)),
        "values": values,
        "expected": "~0.5 (no signal after label shuffling)",
    }


def run_leakage_checks(df, pipe, ml_feature_cols, train_idx, val_idx, test_idx):
    feature_target_overlap = sorted(
        set(ml_feature_cols) & set(TARGET_COLS)
    )
    sample_X = prepare_dataframe(df.head(20), ml_feature_cols)
    targets_in_X = sorted(set(TARGET_COLS) & set(sample_X.columns))

    preprocessor = None
    if hasattr(pipe, "named_steps"):
        preprocessor = pipe.named_steps.get("preprocessor")
    preprocessor_input = list(
        getattr(preprocessor, "feature_names_in_", [])
    ) if preprocessor is not None else []
    targets_in_preprocessor = sorted(
        set(TARGET_COLS) & set(preprocessor_input)
    )

    duplicate_ids = int(df[ID_COL].duplicated().sum())
    target_relationships = {
        "residual_tumor_0_equals_local_0": bool(
            (
                (df["residual_tumor"] == 0)
                == (df["local_residual_tumor"] == 0)
            ).all()
        ),
        "local_0_equals_response_complete": bool(
            (
                (df["local_residual_tumor"] == 0)
                == (df["response_category"] == "complete")
            ).all()
        ),
        "safe_equals_local0_and_ct0": bool(
            (
                df[TARGET_COL]
                == (
                    (df["local_residual_tumor"] == 0)
                    & (df["ct_distant_progression"] == 0)
                ).astype(int)
            ).all()
        ),
    }
    split_sets = [set(train_idx), set(val_idx), set(test_idx)]
    splits_disjoint = bool(
        not (split_sets[0] & split_sets[1])
        and not (split_sets[0] & split_sets[2])
        and not (split_sets[1] & split_sets[2])
    )
    passed = bool(
        not feature_target_overlap
        and not targets_in_X
        and not targets_in_preprocessor
        and duplicate_ids == 0
        and all(target_relationships.values())
        and splits_disjoint
        and "ct_distant_progression" not in ml_feature_cols
    )
    return {
        "passed": passed,
        "feature_target_overlap": feature_target_overlap,
        "target_columns_in_X": targets_in_X,
        "target_columns_in_preprocessor_input": targets_in_preprocessor,
        "duplicate_patient_ids": duplicate_ids,
        "target_relationships": target_relationships,
        "splits_disjoint": splits_disjoint,
        "ct_distant_progression_in_ml_features": (
            "ct_distant_progression" in ml_feature_cols
        ),
        "note": (
            "ct_distant_progression is a Layer 1 rule input and is "
            "deliberately excluded from the ML feature matrix."
        ),
    }


def build_data_audit(
    df, train_idx, val_idx, test_idx,
    train_idx_ml, val_idx_ml, test_idx_ml,
):
    rule_stats = report_hard_stop_stats(df)
    target_counts = df[TARGET_COL].value_counts().to_dict()
    missing_counts = {
        column: int(df[column].isnull().sum()) for column in MISSING_COLS
    }
    missing_exam_counts = {
        exam: int(
            df[EXAM_AVAILABILITY_COLUMNS[exam][0]].isnull().sum()
        )
        for exam in REQUIRED_EXAMS
    }
    n_complete_local_distant = int(
        (
            (df["local_residual_tumor"] == 0)
            & (df["ct_distant_progression"] == 1)
        ).sum()
    )
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target_counts": {
            str(key): int(value) for key, value in target_counts.items()
        },
        "target_prevalence": float(df[TARGET_COL].mean()),
        "missing_counts": missing_counts,
        "missing_exam_counts": missing_exam_counts,
        "n_hard_stops": int(rule_stats["_combined"]["n_stopped"]),
        "n_ml_remaining": int(rule_stats["_combined"]["n_remaining"]),
        "n_complete_local_with_distant_progression": (
            n_complete_local_distant
        ),
        "hard_stop_stats": rule_stats,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "train_ml": int(len(train_idx_ml)),
            "val_ml": int(len(val_idx_ml)),
            "test_ml": int(len(test_idx_ml)),
        },
        "n_features_ml": int(len(get_ml_feature_columns())),
        "n_features_all": int(len(get_feature_columns())),
        "note": (
            "safe_for_watch_and_wait = 1 requires local complete response "
            "(residual_tumor = 0) and no distant progression "
            "(ct_distant_progression = 0). The "
            f"{n_complete_local_distant} patients with local complete "
            "response but distant progression are correctly excluded by "
            "the CT hard stop."
        ),
    }


def build_environment():
    def _version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    import sklearn

    return {
        "python": platform_mod.python_version(),
        "platform": platform_mod.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
        "streamlit": _version("streamlit"),
        "shap": _version("shap"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_results_md(metrics, split, figure_prefix="figures"):
    eval_metrics = metrics["eval"]
    full = eval_metrics["full"]
    ml = eval_metrics["ml"]
    data = metrics["data"]
    baselines = metrics["baselines"]
    ablation = metrics["ablation"]
    subgroups = metrics["subgroups"]
    robustness = metrics["robustness"]
    model = metrics["model"]
    leakage = metrics["leakage_checks"]
    shuffled = metrics["shuffled_label_control"]
    split_label = "TEST" if split == "test" else "VALIDATION (debug)"

    if split == "test":
        discipline = (
            "The test split was scored exactly once "
            f"(`test_eval_count = {metrics['test_eval_count']}`). "
            "No test labels or metrics were used for model selection, "
            "calibration, threshold selection, feature selection, or "
            "example selection."
        )
        if metrics["test_eval_count"] > 1:
            discipline += (
                " **Protocol deviation:** the test set has been scored "
                f"{metrics['test_eval_count']} times; this rerun is disclosed."
            )
    else:
        discipline = (
            "This is a validation debug/confirmation report. The test split "
            "has not been scored (`test_eval_count = 0`)."
        )

    def fig_path(filename):
        return f"{figure_prefix}/{filename}" if figure_prefix else filename

    def _gap_line(family, gap):
        return (
            f"- **{family}**: AUROC gap {_fmt(gap['auroc_gap'])}, "
            f"PPV gap {_fmt(gap['ppv_gap'])}, coverage gap "
            f"{_fmt_pct(gap['coverage_gap'])} "
            f"({gap['n_reliable_subgroups']} reliable subgroups)."
        )

    lines = []
    lines.append(
        "# Rectal cancer post-CRT decision support - evaluation report"
    )
    lines.append("")
    lines.append(f"**Evaluated split:** {split_label}  ")
    lines.append(f"**Generated:** {metrics['generated_at']}  ")
    lines.append(
        f"**Model:** `{model['best_model']}` with calibration "
        f"`{model['calibration']}`; thresholds "
        f"`t_low={model['t_low']:.3f}`, `t_high={model['t_high']:.3f}`.  "
    )
    lines.append(f"**Model version:** `{model['model_version']}`")
    lines.append("")
    lines.append("## Scope and honest framing")
    lines.append("")
    lines.append(
        "The pipeline, decision logic, abstention behaviour and interface "
        "are demonstrated on the provided exam-output data; clinical "
        "validation requires external real cases with the same schema, "
        "split by patient."
    )
    lines.append("")
    lines.append(
        "This is decision support only. It is not an autonomous diagnostic "
        "device. Every definite COMPLETE or NOT COMPLETE call must be "
        "reviewed by the multidisciplinary team (MDT)."
    )
    lines.append("")
    lines.append(
        "The provided dataset is highly separable. Near-ceiling AUROC is a "
        "property of this dataset and should not be read as clinical-grade "
        "performance."
    )
    lines.append("")
    lines.append("## Test-set discipline")
    lines.append("")
    lines.append(discipline)
    lines.append("")
    lines.append(
        "- Model selection: repeated 5x5 cross-validation on the training "
        "split only."
    )
    lines.append(
        "- Calibration selection: repeated 5x5 cross-validation on the "
        "training split only."
    )
    lines.append(
        "- Threshold selection: out-of-fold (OOF) predictions from the "
        "training split, using the bootstrap lower-bound rule in "
        "`src/config.py`."
    )
    lines.append(
        "- Validation split: confirmation only; it did not choose the "
        "model, calibration, or thresholds."
    )
    lines.append("")
    lines.append("## Data audit")
    lines.append("")
    lines.append(
        _table(
            ["Item", "Value"],
            [
                ["Rows", data["n_rows"]],
                ["Columns", data["n_columns"]],
                ["Safe (target=1)", data["target_counts"].get("1", 0)],
                ["Not safe (target=0)", data["target_counts"].get("0", 0)],
                ["Prevalence", _fmt(data["target_prevalence"], 3)],
                ["Hard-stopped patients", data["n_hard_stops"]],
                ["ML-eligible patients", data["n_ml_remaining"]],
                [
                    "Local complete but distant progression",
                    data["n_complete_local_with_distant_progression"],
                ],
                ["ML features", data["n_features_ml"]],
                [
                    "All features (incl. CT progression rule input)",
                    data["n_features_all"],
                ],
            ],
        )
    )
    lines.append("")
    lines.append("Split sizes (full population / ML-eligible):")
    lines.append("")
    sizes = data["split_sizes"]
    lines.append(
        _table(
            ["Split", "Full", "ML-eligible"],
            [
                ["Train", sizes["train"], sizes["train_ml"]],
                ["Validation", sizes["val"], sizes["val_ml"]],
                ["Test", sizes["test"], sizes["test_ml"]],
            ],
        )
    )
    lines.append("")
    lines.append("Missingness and exam availability:")
    lines.append("")
    lines.append(
        _table(
            ["Column / exam", "Missing (n)"],
            [
                [f"`{key}`", value]
                for key, value in data["missing_counts"].items()
            ]
            + [
                [
                    f"Key exam column: {EXAM_DISPLAY_NAMES.get(exam, exam)}",
                    data["missing_exam_counts"][exam],
                ]
                for exam in REQUIRED_EXAMS
            ],
        )
    )
    lines.append("")
    lines.append("### Layer 1 hard stops")
    lines.append("")
    lines.append(
        _table(
            [
                "Rule",
                "Patients",
                "Correct NOT COMPLETE",
                "False NOT COMPLETE",
                "Agreement",
            ],
            [
                [
                    key,
                    result["n_patients"],
                    result["correct_not_safe"],
                    result["false_not_safe"],
                    _fmt_pct(result["agreement_pct"]),
                ]
                for key, result in data["hard_stop_stats"].items()
                if key != "_combined"
            ],
        )
    )
    lines.append("")
    combined = data["hard_stop_stats"]["_combined"]
    n_stopped = combined["n_stopped"]
    n_correct = combined["correct_not_safe"]
    n_false = combined["false_not_safe"]
    lines.append(
        f"Combined: **{n_stopped}** patients hard-stopped, "
        f"**{n_correct}** correct "
        f"({_fmt(n_correct / n_stopped * 100, 1)}%), "
        f"**{n_false}** false NOT COMPLETE. The {n_false} false NOT COMPLETE "
        "patients are true complete responders who were forced to NOT "
        "COMPLETE. This is a safety-conservative error, but it is an error "
        "and is reported."
    )
    lines.append("")
    lines.append(
        "The target is a composite: `safe_for_watch_and_wait = 1` requires "
        "local complete response (`residual_tumor = 0`) and no distant "
        "progression (`ct_distant_progression = 0`). The "
        f"{data['n_complete_local_with_distant_progression']} patients with "
        "local complete response but distant progression are correctly "
        "excluded by the CT hard stop."
    )
    lines.append("")
    lines.append("## Model and thresholds")
    lines.append("")
    lines.append(
        f"Selected model: **{model['best_model']}**; calibration: "
        f"**{model['calibration']}**; OOF AUROC "
        f"**{_fmt(model['oof_auroc'])}**, OOF Brier "
        f"**{_fmt(model['oof_brier'])}**; validation AUROC "
        f"**{_fmt(model['val_auroc'])}**, validation Brier "
        f"**{_fmt(model['val_brier'])}**."
    )
    lines.append("")
    threshold = model["threshold_selection"]
    lines.append(
        f"Thresholds: `t_low = {model['t_low']:.3f}`, "
        f"`t_high = {model['t_high']:.3f}`. Selection rule: "
        f"{threshold['method']}. OOF coverage "
        f"**{_fmt_pct(threshold['coverage_pct'])}**; OOF PPV "
        f"**{_fmt(threshold['ppv_complete'])}**; OOF NPV "
        f"**{_fmt(threshold['npv_not_complete'])}**. Fallback used: "
        f"t_high={'yes' if threshold['fallback_high'] else 'no'}, "
        f"t_low={'yes' if threshold['fallback_low'] else 'no'}."
    )
    lines.append("")
    lines.append("Candidate models (repeated 5x5 CV on training):")
    lines.append("")
    cv_rows = []
    for name, result in model["cv_results"].items():
        if "auroc" not in result:
            continue
        cv_rows.append(
            [
                name,
                f"{result['auroc']:.4f} +/- {result['auroc_std']:.4f}",
                f"{result['auprc']:.4f} +/- {result['auprc_std']:.4f}",
                f"{result['brier']:.4f} +/- {result['brier_std']:.4f}",
                result["log_loss"],
            ]
        )
    lines.append(
        _table(
            ["Model", "AUROC", "AUPRC", "Brier", "Log loss"], cv_rows
        )
    )
    lines.append("")
    lines.append("Calibration comparison (repeated 5x5 CV on training):")
    lines.append("")
    calibration_rows = []
    for name, result in model["calibration_results"].items():
        if "brier" not in result:
            continue
        calibration_rows.append(
            [
                name,
                result["brier"],
                result["brier_std"],
                result["log_loss"],
                result["auroc"],
            ]
        )
    lines.append(
        _table(
            ["Calibration", "Brier", "Brier SD", "Log loss", "AUROC"],
            calibration_rows,
        )
    )
    lines.append("")
    lines.append(
        "The uncalibrated logistic model was selected because its OOF "
        "Brier and log loss were best or tied; a small Brier difference is "
        "broken in favour of simplicity."
    )
    lines.append("")
    lines.append("## Primary metrics")
    lines.append("")
    lines.append(
        _table(
            PRIMARY_HEADERS,
            [
                _metric_row(
                    f"Full deployed pipeline ({split_label})", full
                ),
                _metric_row("ML-only model", ml),
            ],
        )
    )
    lines.append("")
    lines.append(
        "Full-pipeline ranking metrics assign score 0 to rule-based NOT "
        "COMPLETE decisions; they describe the deployed ordering, not model "
        "calibration. Calibration and ROC/PR figures use the ML subset. The "
        f"full pipeline made **{full['n_complete']}** COMPLETE calls, "
        f"**{full['n_not_complete']}** NOT COMPLETE calls and "
        f"**{full['n_uncertain']}** UNCERTAIN calls."
    )
    lines.append("")
    lines.append(
        "The ML-only metrics use the hard-stop-eligible subset and ignore "
        "the missing-exam policy; the full pipeline applies both. This is "
        "why the two populations can differ."
    )
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    b0_full = baselines["B0_majority_full"]
    b0_ml = baselines["B0_majority_ml"]
    b1_full = baselines["B1_rule_full"]
    b2_ml = baselines["B2_logreg8_ml"]
    b2_m = baselines["B2_logreg8_with_m_thresholds_ml"]
    lines.append(
        _table(
            PRIMARY_HEADERS,
            [
                _metric_row("B0 majority (full)", b0_full),
                _metric_row("B0 majority (ML)", b0_ml),
                _metric_row("B1 rule protocol (full)", b1_full),
                _metric_row(
                    "B2 8-feature logistic (ML, threshold 0.5)", b2_ml
                ),
                _metric_row(
                    "B2 with M thresholds (ML, sensitivity)", b2_m
                ),
                _metric_row("M full model (full)", full),
                _metric_row("M full model (ML)", ml),
            ],
        )
    )
    lines.append("")
    lines.append(
        "B0 is the majority-class baseline. B1 is the simple rule protocol "
        "without abstention. B2 is a logistic model on 8 hand-picked "
        "features without abstention. "
        f"B2 AUROC is **{_fmt_ci(b2_ml['auroc'])}** versus M ML AUROC "
        f"**{_fmt_ci(ml['auroc'])}**: this is essentially parity. The honest "
        "interpretation is that the tabular signal is strong and the "
        "8-feature baseline captures most of it. The contribution of M is "
        "the safety layer, abstention, exam-level explanation, and "
        "interface, not a large discrimination gain over B2."
    )
    lines.append("")
    lines.append(
        "B2 with M's thresholds is a sensitivity analysis; its coverage "
        f"({_fmt_pct(b2_m['coverage_pct'])}) and PPV/NPV are not directly "
        "comparable because the thresholds were selected for M."
    )
    lines.append("")
    lines.append("## Ablation by exam")
    lines.append("")
    ablation_rows = []
    for name, result in ablation.items():
        if result.get("auroc_mean") is None:
            ablation_rows.append(
                [
                    name,
                    result["n_exams"],
                    result["n_features"],
                    "n/a",
                    "n/a",
                    "n/a",
                ]
            )
            continue
        ablation_rows.append(
            [
                name,
                result["n_exams"],
                result["n_features"],
                f"{result['auroc_mean']:.4f} +/- "
                f"{result['auroc_std']:.4f}",
                f"{result['auprc_mean']:.4f} +/- "
                f"{result['auprc_std']:.4f}",
                f"{result['brier_mean']:.4f} +/- "
                f"{result['brier_std']:.4f}",
            ]
        )
    lines.append(
        _table(
            ["Feature set", "Exams", "Features", "AUROC", "AUPRC", "Brier"],
            ablation_rows,
        )
    )
    lines.append("")
    lines.append(
        "Repeated 5x5 CV on the training ML split, using the same model "
        "class and calibration as the final model. The CT group excludes "
        "`ct_distant_progression` because Layer 1 removes every patient "
        "with distant progression; within the ML population that column is "
        "constant. The last point adds baseline, treatment and "
        "post-treatment lab features. It should be close to the OOF AUROC, "
        "which is a consistency check."
    )
    lines.append("")
    lines.append(f"![Ablation by exam]({fig_path('ablation_by_exam.png')})")
    lines.append("")
    lines.append(
        "Interpretation: AUROC rises from MRI-only through the four-exam "
        "sets; adding CT alone is roughly flat in this dataset, and the "
        "best CV AUROC comes from the full feature set. The gains are "
        "small, and the 8-feature B2 baseline already captures most of "
        "the discrimination. The novelty is the safety/abstention layer, "
        "not a large raw AUROC gain."
    )
    lines.append("")
    lines.append("## Calibration and abstention")
    lines.append("")
    lines.append(
        f"![Calibration]({fig_path('calibration_ml.png')})"
    )
    lines.append("")
    lines.append(f"![ROC and PR]({fig_path('roc_pr_ml.png')})")
    lines.append("")
    lines.append(
        f"![Risk-coverage]({fig_path('risk_coverage.png')})"
    )
    lines.append("")
    lines.append(
        f"Coverage is **{_fmt_pct(ml['coverage_pct'])}** on the ML subset "
        f"and **{_fmt_pct(full['coverage_pct'])}** for the full deployed "
        f"pipeline. The tool abstains on **{ml['n_uncertain']}** ML patients "
        f"and **{full['n_uncertain']}** full-pipeline patients rather than "
        "forcing a call."
    )
    lines.append("")
    lines.append(
        "The risk-coverage curve uses the ML subset with `t_low` fixed at "
        "the selected value and `t_high` varied. Panel (a) shows the "
        "definite-decision error rate vs coverage; panel (b) shows the "
        "false-COMPLETE rate among COMPLETE calls vs COMPLETE-call "
        f"coverage, with the {int(PPV_TARGET * 100)}% PPV target line. The "
        "deployed point is marked. This is the honest trade-off: higher "
        "coverage is possible, but only by accepting a higher "
        "false-COMPLETE risk."
    )
    lines.append("")
    lines.append(
        f"![Confusion matrix]({fig_path('confusion_matrix.png')})"
    )
    lines.append("")
    lines.append(
        f"Full-pipeline confusion matrix: {full['n_false_complete']} false "
        f"COMPLETE calls and {full['n_missed_complete']} missed complete "
        "responders (NOT COMPLETE on true safe). The missed complete "
        "responders include the hard-stop false NOT COMPLETE patients."
    )
    lines.append("")
    lines.append("## Subgroups")
    lines.append("")
    subgroup_rows = []
    for key, result in subgroups["subgroups"].items():
        subgroup_rows.append(
            [
                key,
                result["family"],
                result["n_full"],
                result["n_ml"],
                _fmt(result["prevalence"], 3),
                _fmt(result["auroc_ml"]),
                _fmt_ci(result["ppv_complete"]),
                _fmt_ci(result["false_complete_rate"]),
                _fmt_pct(result["coverage_pct"]),
                "yes" if result["reliable"] else "no",
            ]
        )
    lines.append(
        _table(
            [
                "Subgroup",
                "Family",
                "n full",
                "n ML",
                "Prevalence",
                "ML AUROC",
                "PPV COMPLETE",
                "False-COMPLETE",
                "Coverage",
                "Reliable",
            ],
            subgroup_rows,
        )
    )
    lines.append("")
    lines.append(
        "Subgroups with fewer than 5 patients in the evaluated split are "
        "omitted from the table. Reliable = at least 10 patients in the "
        "full subgroup and at least 5 in each target class; smaller "
        "subgroups are exploratory and should not drive conclusions. The "
        "age bands are `<50`, `50-65`, `>65` (50 belongs to `50-65`, 65 "
        "to `>65`); distance bands are `<=2 cm`, `2-4 cm`, `>4 cm` (2 "
        "belongs to `2-4`, 4 to `>4`)."
    )
    lines.append("")
    gap_lines = [
        _gap_line(family, gap)
        for family, gap in subgroups["gaps"].items()
    ]
    if gap_lines:
        lines.append("Gaps across reliable subgroups:")
        lines.append("")
        lines.extend(gap_lines)
        lines.append("")
    lines.append(f"![Subgroups]({fig_path('subgroups.png')})")
    lines.append("")
    lines.append("## Robustness")
    lines.append("")
    robustness_rows = []
    for exam, result in robustness["drop_exam"].items():
        model_only = result["model_only"]
        deployed = result["deployed"]
        robustness_rows.append(
            [
                f"Drop {result['display_name']}",
                model_only["n"],
                _fmt(model_only.get("auroc")),
                model_only["n_false_complete"],
                deployed["n_false_complete"],
                _fmt_pct(deployed["n_uncertain_pct"]),
                result["note"],
            ]
        )
    for key, label in [
        ("noise", "Feature noise"),
        ("extra_missingness", "20% extra missingness"),
    ]:
        result = robustness[key]
        summary = result["summary"]
        uncertain_mean = summary["n_uncertain_deployed"]["mean"]
        robustness_rows.append(
            [
                f"{label} ({result['n_seeds']} seeds)",
                result["n"],
                _fmt(summary["auroc"]["mean"]),
                _fmt(summary["n_false_complete"]["mean"], 1),
                _fmt(summary["n_false_complete_deployed"]["mean"], 1),
                _fmt_pct(
                    uncertain_mean / result["n"] * 100
                    if uncertain_mean is not None else None
                ),
                result["note"],
            ]
        )
    lines.append(
        _table(
            [
                "Scenario",
                "n",
                "Model-only AUROC",
                "Model-only false COMPLETE",
                "Deployed false COMPLETE",
                "Deployed UNCERTAIN",
                "Notes",
            ],
            robustness_rows,
        )
    )
    lines.append("")
    lines.append(
        "Drop-exam analysis: the deployed missing-exam policy forces "
        "UNCERTAIN for every non-hard-stopped patient when a required exam "
        "is missing, so deployed COMPLETE calls are zero. The model-only "
        "column shows what would happen if the model were trusted without "
        "the policy. For CT, the model-only path calls distant-progression "
        "patients COMPLETE; the policy prevents exactly this failure. The "
        "deployed pipeline still has hard-stop false NOT COMPLETE errors, "
        "which are reported separately in the data audit."
    )
    lines.append("")
    lines.append(
        "Noise: Gaussian noise at 0.2x the training standard deviation on "
        "`mri_ADC_mean` and `mri_wall_thickness_mm`, and a 10% label flip "
        "on `mri_DWI_restriction`, over 5 seeds. Extra missingness: each ML "
        "feature is independently set missing with probability 0.2, over 5 "
        "seeds. Both use the ML evaluated subset. Model-only ignores the "
        "policy; "
        "deployed applies it. The policy trades coverage for safety."
    )
    lines.append("")
    lines.append(f"![Robustness]({fig_path('robustness.png')})")
    lines.append("")
    lines.append("## Validation confirmation")
    lines.append("")
    if split == "test":
        lines.append(
            "The validation split was used only to confirm the selected "
            "model, calibration and thresholds. It did not choose them."
        )
    else:
        lines.append(
            "This report evaluates the validation split. The test split has "
            "not been scored."
        )
    lines.append("")
    validation = metrics["validation"]
    validation_full = validation["full"]
    validation_ml = validation["ml"]
    lines.append(
        _table(
            [
                "Population",
                "n",
                "AUROC",
                "Brier",
                "PPV COMPLETE",
                "NPV NOT COMPLETE",
                "Coverage",
            ],
            [
                [
                    "Validation full",
                    validation_full["n"],
                    _fmt_ci(validation_full["auroc"]),
                    _fmt_ci(validation_full["brier"]),
                    _fmt_ci(validation_full["ppv_complete"]),
                    _fmt_ci(validation_full["npv_not_complete"]),
                    _fmt_pct(validation_full["coverage_pct"]),
                ],
                [
                    "Validation ML",
                    validation_ml["n"],
                    _fmt_ci(validation_ml["auroc"]),
                    _fmt_ci(validation_ml["brier"]),
                    _fmt_ci(validation_ml["ppv_complete"]),
                    _fmt_ci(validation_ml["npv_not_complete"]),
                    _fmt_pct(validation_ml["coverage_pct"]),
                ],
            ],
        )
    )
    lines.append("")
    confirmation = model["validation_confirmation"]
    lines.append(
        f"Training-time confirmation: PPV "
        f"{_fmt(confirmation['ppv_complete'])} "
        f"(lower {_fmt(confirmation['ppv_ci_lower'])}), NPV "
        f"{_fmt(confirmation['npv_not_complete'])} "
        f"(lower {_fmt(confirmation['npv_ci_lower'])}), coverage "
        f"{_fmt_pct(confirmation['coverage_pct'])}. Passes PPV rule: "
        f"{confirmation['passes_ppv_rule']}; passes NPV rule: "
        f"{confirmation['passes_npv_rule']}."
    )
    lines.append("")
    if not confirmation["passes_npv_rule"]:
        lines.append(
            "**Honest caveat:** the OOF threshold selection passed the NPV "
            "rule on training OOF, but the training-time validation "
            "confirmation (computed on the ML subset, model only) did not: "
            f"the NPV point estimate was "
            f"{_fmt(confirmation['npv_not_complete'])}, but the 95% lower "
            f"bound was {_fmt(confirmation['npv_ci_lower'])}, below the "
            f"{NPV_TARGET:.2f} target. In this evaluation the ML-only "
            f"validation NPV is "
            f"{_fmt_ci(validation_ml['npv_not_complete'])}, while the full "
            f"deployed pipeline (rules + model) has "
            f"{_fmt_ci(validation_full['npv_not_complete'])}. The full "
            "pipeline's higher NPV is partly due to hard-stop NOT COMPLETE "
            "calls and does not rescue the model-only NPV rule. The "
            "model-only NPV rule is therefore not confirmed; this is a "
            "limitation and is not hidden."
        )
        lines.append("")
    lines.append("## Shuffled-label control and leakage checks")
    lines.append("")
    lines.append(
        f"Shuffled training labels: AUROC "
        f"**{_fmt(shuffled['auroc_mean'])}** "
        f"(SD {_fmt(shuffled['auroc_std'])}, range "
        f"{_fmt(shuffled['auroc_min'])}-{_fmt(shuffled['auroc_max'])}, "
        f"{shuffled['n_shuffles']} shuffles). Expected ~0.5. This confirms "
        "that the pipeline is not leaking the target through preprocessing "
        "or split handling."
    )
    lines.append("")
    lines.append(
        _table(
            ["Leakage check", "Result"],
            [
                [
                    "Overall",
                    "PASSED" if leakage["passed"] else "FAILED",
                ],
                [
                    "Target columns in ML features",
                    ", ".join(leakage["feature_target_overlap"]) or "none",
                ],
                [
                    "Target columns in prepared X",
                    ", ".join(leakage["target_columns_in_X"]) or "none",
                ],
                [
                    "Target columns in preprocessor input",
                    ", ".join(
                        leakage["target_columns_in_preprocessor_input"]
                    ) or "none",
                ],
                [
                    "Duplicate patient IDs",
                    leakage["duplicate_patient_ids"],
                ],
                ["Splits disjoint", leakage["splits_disjoint"]],
                [
                    "ct_distant_progression in ML features",
                    leakage["ct_distant_progression_in_ml_features"],
                ],
                [
                    "residual_tumor=0 equals local_residual_tumor=0",
                    leakage["target_relationships"][
                        "residual_tumor_0_equals_local_0"
                    ],
                ],
                [
                    "local_residual_tumor=0 equals "
                    "response_category=complete",
                    leakage["target_relationships"][
                        "local_0_equals_response_complete"
                    ],
                ],
                [
                    "safe = local complete AND no distant progression",
                    leakage["target_relationships"][
                        "safe_equals_local0_and_ct0"
                    ],
                ],
            ],
        )
    )
    lines.append("")
    lines.append(
        "`ct_distant_progression` is a Layer 1 rule input and is "
        "deliberately excluded from the ML feature matrix. The target "
        "columns are never passed to `prepare_dataframe`."
    )
    lines.append("")
    lines.append("## Limitations and what this is not")
    lines.append("")
    lines.append(
        "- **Data:** provided exam-output data, not external prospective "
        "cases. Clinical validation requires external real cases with the "
        "same schema, split by patient."
    )
    lines.append(
        f"- **Sample size:** the evaluated population is small "
        f"(full n={full['n']}, ML n={ml['n']}). Confidence intervals are "
        "wide; subgroup estimates are exploratory."
    )
    lines.append(
        "- **Near-ceiling performance:** the dataset is highly separable; "
        "AUROC near 1.0 is a dataset property, not evidence of "
        "clinical-grade performance."
    )
    lines.append(
        "- **B2 parity:** an 8-feature logistic model is essentially at "
        "parity with the full model. The full model's value is the safety "
        "layer, abstention, exam-level explanation and interface, not a "
        "large discrimination gain."
    )
    lines.append(
        f"- **Hard stops are not perfect:** the rules make {n_false} false "
        "NOT COMPLETE calls. They are conservative, not error-free."
    )
    lines.append(
        "- **No imaging model:** the tool uses structured exam outputs, not "
        "raw MRI/CT images. It does not replace radiology or pathology "
        "review."
    )
    lines.append(
        "- **No survival endpoint:** the target is a composite "
        "watch-and-wait safety label, not disease-free survival. The tool "
        "does not predict recurrence or survival."
    )
    lines.append(
        "- **No external validation or prospective silent trial:** "
        "calibration drift and workflow impact are unknown."
    )
    lines.append(
        "- **Local demo only:** the app runs locally, makes no network "
        "calls, and stores only a salted SHA-256 audit hash of the patient "
        "ID. It is not a regulated device."
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```powershell")
    lines.append("pip install -r requirements.txt")
    lines.append("python -m src.train")
    lines.append("python -m src.evaluate --split val")
    lines.append("python -m src.evaluate --split test")
    lines.append("streamlit run app/app.py")
    lines.append("python -m pytest -q")
    lines.append("```")
    lines.append("")
    lines.append(
        "The test evaluation is guarded: if `reports/metrics.json` exists, "
        "the command refuses to run unless `--force` is passed. A forced "
        "rerun increments `test_eval_count` and is disclosed at the top of "
        "this report."
    )
    lines.append("")
    lines.append("Environment:")
    lines.append("")
    lines.append(
        _table(
            ["Package", "Version"],
            [
                [key, _fmt(value)]
                for key, value in metrics["environment"].items()
            ],
        )
    )
    lines.append("")
    figures = metrics.get("figures", {})
    if figures:
        lines.append("## Figures")
        lines.append("")
        lines.append(
            _table(
                ["Figure", "File"],
                [
                    [key, f"`{value}`"]
                    for key, value in figures.items()
                ],
            )
        )
        lines.append("")
    lines.append("## Bottom line")
    lines.append("")
    lines.append(
        f"The deployed pipeline made **{full['n_complete']}** COMPLETE calls "
        f"with **{_fmt_ci(full['ppv_complete'])}** PPV and "
        f"**{full['n_false_complete']}** false COMPLETE calls; it abstained "
        f"on **{full['n_uncertain']}** patients. The ML model alone achieved "
        f"AUROC **{_fmt_ci(ml['auroc'])}** on the ML subset. The rules "
        f"correctly handled **{n_correct}/{n_stopped}** hard-stop patients "
        f"({_fmt(n_correct / n_stopped * 100, 1)}% agreement) and made "
        f"**{n_false}** false NOT COMPLETE calls. This is a working, "
        "auditable decision-support prototype; it is not clinically "
        "validated."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def load_saved_data_and_splits():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_FILE}"
        )
    split_path = REPORTS_DIR / "split_indices.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            "reports/split_indices.npz not found. Run "
            "`python -m src.train` first."
        )

    df = pd.read_csv(DATA_FILE)
    splits = np.load(split_path)
    train_idx = splits["train"].astype(int)
    val_idx = splits["val"].astype(int)
    test_idx = splits["test"].astype(int)
    train_ml = splits["train_ml"].astype(int)
    val_ml = splits["val_ml"].astype(int)
    test_ml = filter_hard_stops(df, test_idx)

    expected_train_ml = filter_hard_stops(df, train_idx)
    expected_val_ml = filter_hard_stops(df, val_idx)
    if not np.array_equal(train_ml, expected_train_ml):
        raise ValueError(
            "split_indices.npz train_ml does not match the current hard "
            "stops. Retrain or restore the matching split file."
        )
    if not np.array_equal(val_ml, expected_val_ml):
        raise ValueError(
            "split_indices.npz val_ml does not match the current hard "
            "stops. Retrain or restore the matching split file."
        )

    return df, train_idx, val_idx, test_idx, train_ml, val_ml, test_ml


def run_evaluation(split="test", force=False):
    print("=" * 70)
    print(f"EVALUATION: {split.upper()} split")
    print("=" * 70)

    metrics_path = REPORTS_DIR / "metrics.json"
    if split == "test" and metrics_path.exists() and not force:
        raise SystemExit(
            "reports/metrics.json already exists. The test set has already "
            "been scored. Pass --force to score it again and disclose the "
            "rerun."
        )

    test_eval_count = 0
    if split == "test":
        if metrics_path.exists() and force:
            with open(metrics_path, encoding="utf-8") as handle:
                old_metrics = json.load(handle)
            test_eval_count = int(
                old_metrics.get("test_eval_count", 1)
            ) + 1
        else:
            test_eval_count = 1

    df, train_idx, val_idx, test_idx, train_idx_ml, val_idx_ml, test_idx_ml = (
        load_saved_data_and_splits()
    )
    pipe, config = load_model_and_config()
    t_low = float(config["t_low"])
    t_high = float(config["t_high"])
    ml_feature_cols = config["ml_feature_cols"]

    if set(ml_feature_cols) != set(get_ml_feature_columns()):
        print(
            "  WARNING: train_config.json ml_feature_cols differ from the "
            "current code. Using the config list saved with the model."
        )

    eval_idx = test_idx if split == "test" else val_idx
    eval_idx_ml = test_idx_ml if split == "test" else val_idx_ml
    eval_full_df = df.iloc[eval_idx]
    eval_ml_df = df.iloc[eval_idx_ml]
    y_full = df[TARGET_COL].iloc[eval_idx].values.astype(int)
    y_ml = df[TARGET_COL].iloc[eval_idx_ml].values.astype(int)

    print("\n[1/9] Deployed predictions and primary metrics...")
    batch_full = predict_batch(
        eval_full_df, pipe, config, log=False
    )
    p_full = batch_full["probability"].astype(float).values
    source = batch_full["source"].values
    p_full_score = np.where(
        source == "rule", 0.0, np.nan_to_num(p_full, nan=0.0)
    )
    d_full = decisions_to_codes(batch_full["decision"].values)
    full_metrics = compute_decision_metrics(
        y_full, p_full_score, d_full, prefix="full_"
    )
    full_metrics["thresholds"] = {"t_low": t_low, "t_high": t_high}
    full_metrics["source_counts"] = {
        "rule": int((source == "rule").sum()),
        "policy": int((source == "policy").sum()),
        "model": int((source == "model").sum()),
    }
    full_metrics["score_note"] = (
        "Rule-based NOT COMPLETE rows are assigned score 0 for ranking "
        "metrics; calibration figures use the ML subset."
    )

    p_ml = _predict_proba_df(pipe, eval_ml_df, ml_feature_cols)
    d_ml = decide_codes(p_ml, t_low, t_high)
    ml_metrics = compute_decision_metrics(
        y_ml, p_ml, d_ml, prefix="ml_"
    )
    ml_metrics["thresholds"] = {"t_low": t_low, "t_high": t_high}
    ml_metrics["source_counts"] = {
        "rule": 0,
        "policy": 0,
        "model": int(len(y_ml)),
    }
    ml_metrics["score_note"] = (
        "Pure model probabilities on the hard-stop-eligible subset."
    )

    print(
        f"  Full: n={full_metrics['n']}  "
        f"AUROC={_fmt_ci(full_metrics['auroc'])}  "
        f"PPV={_fmt_ci(full_metrics['ppv_complete'])}  "
        f"coverage={_fmt_pct(full_metrics['coverage_pct'])}"
    )
    print(
        f"  ML:   n={ml_metrics['n']}  "
        f"AUROC={_fmt_ci(ml_metrics['auroc'])}  "
        f"PPV={_fmt_ci(ml_metrics['ppv_complete'])}  "
        f"coverage={_fmt_pct(ml_metrics['coverage_pct'])}"
    )

    print("\n[2/9] Baselines...")
    b0_majority_full = int(
        df[TARGET_COL].iloc[train_idx].mode().iloc[0]
    )
    b0_majority_ml = int(
        df[TARGET_COL].iloc[train_idx_ml].mode().iloc[0]
    )
    b0_full = compute_b0_metrics(y_full, b0_majority_full)
    b0_ml = compute_b0_metrics(y_ml, b0_majority_ml)
    b1_full = compute_rule_metrics(
        y_full, rule_based_protocol(eval_full_df), prefix="b1_"
    )

    b2_pipe = build_pipeline(B2_FEATURES, "logistic_l2", "none")
    b2_X_train = prepare_dataframe(
        df.iloc[train_idx_ml], B2_FEATURES
    )
    b2_y_train = df[TARGET_COL].iloc[train_idx_ml]
    b2_X_eval = prepare_dataframe(eval_ml_df, B2_FEATURES)
    b2_pipe.fit(b2_X_train, b2_y_train)
    p_b2 = b2_pipe.predict_proba(b2_X_eval)[:, 1]
    b2_ml = compute_decision_metrics(
        y_ml, p_b2, (p_b2 >= 0.5).astype(int), prefix="b2_"
    )
    b2_m = compute_decision_metrics(
        y_ml, p_b2, decide_codes(p_b2, t_low, t_high), prefix="b2_m_"
    )
    baselines = {
        "B0_majority_full": b0_full,
        "B0_majority_ml": b0_ml,
        "B1_rule_full": b1_full,
        "B2_logreg8_ml": b2_ml,
        "B2_logreg8_with_m_thresholds_ml": b2_m,
        "B0_note": (
            "Majority class from the training split; no model, no "
            "abstention."
        ),
        "B1_note": (
            "Simple rule protocol: mrTRG <= 2, DRE scar/no abnormality, "
            "endoscopy scar, biopsy not positive, CT no distant "
            "progression. No abstention."
        ),
        "B2_note": (
            "Logistic regression on 8 hand-picked features; trained on the "
            "training ML split; no abstention at threshold 0.5."
        ),
    }
    print(
        f"  B1 rule AUROC={_fmt_ci(b1_full['auroc'])}  "
        f"PPV={_fmt_ci(b1_full['ppv_complete'])}"
    )
    print(
        f"  B2 8-feature AUROC={_fmt_ci(b2_ml['auroc'])}  "
        f"(M ML AUROC={_fmt_ci(ml_metrics['auroc'])})"
    )

    print("\n[3/9] Ablation by exam...")
    ablation = run_ablation(
        df, train_idx_ml, config["best_model"], config["calibration"],
        ml_feature_cols,
    )

    print("\n[4/9] Subgroups...")
    subgroups = run_subgroups(
        df, eval_idx, eval_idx_ml, y_full, d_full,
        y_ml, p_ml, d_ml, ml_metrics["auroc"]["point"],
    )

    print("\n[5/9] Robustness...")
    robustness = run_robustness(
        df, train_idx, eval_idx, eval_idx_ml, pipe, config
    )

    print("\n[6/9] Shuffled-label control...")
    shuffled = run_shuffled_label_control(
        df, train_idx_ml, val_idx_ml, config["best_model"],
        config["calibration"], ml_feature_cols,
    )
    print(
        f"  Shuffled-label AUROC mean={_fmt(shuffled['auroc_mean'])} "
        "(expected ~0.5)"
    )

    print("\n[7/9] Leakage checks...")
    leakage = run_leakage_checks(
        df, pipe, ml_feature_cols, train_idx, val_idx, test_idx
    )
    print(f"  Leakage check: {'PASSED' if leakage['passed'] else 'FAILED'}")
    if not leakage["passed"]:
        raise RuntimeError(
            "Leakage checks failed; refusing to write final metrics."
        )

    print("\n[8/9] Data audit...")
    data = build_data_audit(
        df, train_idx, val_idx, test_idx,
        train_idx_ml, val_idx_ml, test_idx_ml,
    )
    print(
        f"  Hard stops: {data['n_hard_stops']}  "
        f"ML remaining: {data['n_ml_remaining']}"
    )

    print("\n[9/9] Saving metrics...")
    if split == "test":
        validation_full = None
        validation_ml = None
        val_batch = predict_batch(
            df.iloc[val_idx], pipe, config, log=False
        )
        val_p_full = val_batch["probability"].astype(float).values
        val_source = val_batch["source"].values
        val_p_full_score = np.where(
            val_source == "rule", 0.0,
            np.nan_to_num(val_p_full, nan=0.0),
        )
        val_d_full = decisions_to_codes(
            val_batch["decision"].values
        )
        val_y_full = df[TARGET_COL].iloc[val_idx].values.astype(int)
        validation_full = compute_decision_metrics(
            val_y_full, val_p_full_score, val_d_full, prefix="val_full_"
        )
        validation_full["thresholds"] = {
            "t_low": t_low, "t_high": t_high
        }
        val_p_ml = _predict_proba_df(
            pipe, df.iloc[val_idx_ml], ml_feature_cols
        )
        val_d_ml = decide_codes(val_p_ml, t_low, t_high)
        val_y_ml = df[TARGET_COL].iloc[val_idx_ml].values.astype(int)
        validation_ml = compute_decision_metrics(
            val_y_ml, val_p_ml, val_d_ml, prefix="val_ml_"
        )
        validation_ml["thresholds"] = {
            "t_low": t_low, "t_high": t_high
        }
        validation = {
            "full": validation_full, "ml": validation_ml
        }
    else:
        validation = {"full": full_metrics, "ml": ml_metrics}

    model_metrics = {
        "model_version": config.get("model_version", "unknown"),
        "best_model": config["best_model"],
        "calibration": config["calibration"],
        "t_low": t_low,
        "t_high": t_high,
        "threshold_selection": config["threshold_selection"],
        "validation_confirmation": config["validation_confirmation"],
        "cv_results": config["cv_results"],
        "calibration_results": config["calibration_results"],
        "oof_auroc": config["oof_auroc"],
        "oof_brier": config["oof_brier"],
        "val_auroc": config["val_auroc"],
        "val_brier": config["val_brier"],
    }

    metrics = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_split": split,
        "test_eval_count": test_eval_count,
        "model": model_metrics,
        "data": data,
        "eval": {"full": full_metrics, "ml": ml_metrics},
        "validation": validation,
        "baselines": baselines,
        "ablation": ablation,
        "subgroups": subgroups,
        "robustness": robustness,
        "shuffled_label_control": shuffled,
        "leakage_checks": leakage,
        "environment": build_environment(),
    }
    if split == "test":
        metrics["test"] = metrics["eval"]

    metrics = _jsonable(metrics)

    if split == "test":
        metrics_path = REPORTS_DIR / "metrics.json"
    else:
        metrics_path = REPORTS_DIR / "metrics_val.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if split == "test":
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        figures = {
            "Calibration (ML subset)": "figures/calibration_ml.png",
            "ROC and PR (ML subset)": "figures/roc_pr_ml.png",
            "Risk-coverage (ML subset)": "figures/risk_coverage.png",
            "Confusion matrix": "figures/confusion_matrix.png",
            "Ablation by exam": "figures/ablation_by_exam.png",
            "Subgroups": "figures/subgroups.png",
            "Robustness": "figures/robustness.png",
        }
        metrics["figures"] = figures

        plot_calibration(
            y_ml, p_ml, t_low, t_high,
            ml_metrics["brier"]["point"],
            FIGURES_DIR / "calibration_ml.png",
        )
        plot_roc_pr(
            y_ml, p_ml, ml_metrics, t_high,
            FIGURES_DIR / "roc_pr_ml.png",
        )
        plot_risk_coverage(
            y_ml, p_ml, t_low, t_high,
            FIGURES_DIR / "risk_coverage.png",
        )
        plot_confusion_matrix(
            full_metrics["confusion_matrix_3x2"],
            FIGURES_DIR / "confusion_matrix.png",
        )
        plot_ablation(
            ablation, FIGURES_DIR / "ablation_by_exam.png"
        )
        plot_subgroups(
            subgroups, ml_metrics["auroc"]["point"],
            FIGURES_DIR / "subgroups.png",
        )
        plot_robustness(
            robustness, FIGURES_DIR / "robustness.png"
        )

        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        report = write_results_md(metrics, split, figure_prefix="figures")
        report_path = REPORTS_DIR / "results.md"
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report)

        test_out = pd.DataFrame(
            {
                "patient_id": eval_full_df["patient_id"].values,
                "y_true": y_full,
                "model_probability": p_full,
                "deployed_score": p_full_score,
                "decision": batch_full["decision"].values,
                "source": batch_full["source"].values,
                "rule": batch_full["rule"].values,
                "unavailable_exams": [
                    ",".join(value) if isinstance(value, list)
                    else (value or "")
                    for value in batch_full["unavailable_exams"].values
                ],
            }
        )
        test_out.to_csv(
            REPORTS_DIR / "test_predictions.csv", index=False
        )

        examples = get_example_patients(
            df, eval_idx, pipe, config
        )
        examples_payload = {
            "generated_from": "test split",
            "model_version": model_metrics["model_version"],
            "thresholds": {"t_low": t_low, "t_high": t_high},
            "selection": (
                "deterministic model-based; see selection_rule per example"
            ),
            "examples": examples,
        }
        APP_DIR.mkdir(parents=True, exist_ok=True)
        with open(
            APP_DIR / "examples.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(_jsonable(examples_payload), handle, indent=2)

    print("\n" + "=" * 70)
    print(f"{split.upper()} evaluation complete")
    print(
        f"  Full: n={full_metrics['n']}  "
        f"AUROC={_fmt_ci(full_metrics['auroc'])}  "
        f"PPV={_fmt_ci(full_metrics['ppv_complete'])}  "
        f"NPV={_fmt_ci(full_metrics['npv_not_complete'])}  "
        f"coverage={_fmt_pct(full_metrics['coverage_pct'])}  "
        f"false-COMPLETE={full_metrics['n_false_complete']}"
    )
    print(
        f"  ML:   n={ml_metrics['n']}  "
        f"AUROC={_fmt_ci(ml_metrics['auroc'])}  "
        f"PPV={_fmt_ci(ml_metrics['ppv_complete'])}  "
        f"NPV={_fmt_ci(ml_metrics['npv_not_complete'])}  "
        f"coverage={_fmt_pct(ml_metrics['coverage_pct'])}"
    )
    print(f"  Metrics: {metrics_path}")
    if split == "test":
        print(f"  Report:  {REPORTS_DIR / 'results.md'}")
        print(f"  Figures: {FIGURES_DIR}")
    print("=" * 70)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained rectal-cr pipeline. Use --split val for a "
            "debug/confirmation run (writes reports/metrics_val.json only) "
            "or --split test for the one-time final test evaluation."
        )
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help=(
            "val = debug/confirmation (writes metrics_val.json only); "
            "test = one-time final evaluation"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "allow re-scoring the test split; the rerun is disclosed in "
            "the report"
        ),
    )
    args = parser.parse_args()
    run_evaluation(split=args.split, force=args.force)


if __name__ == "__main__":
    main()
