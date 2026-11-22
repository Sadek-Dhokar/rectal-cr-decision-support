"""
Training: model comparison, calibration selection, OOF threshold selection.

Discipline
----------
* Model choice and calibration are selected by repeated stratified CV on the
  training split only.
* Abstention thresholds are selected on out-of-fold (OOF) predictions from the
  training split, using a bootstrap lower-bound rule defined in config.py.
* The validation split is used only to confirm the selected thresholds.
* The test split is never read here.
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import (
    StratifiedShuffleSplit, StratifiedKFold, cross_val_predict,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, precision_score, recall_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline
import joblib

from src.config import (
    DATA_FILE, SEED, TARGET_COL, ID_COL,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    PPV_TARGET, NPV_TARGET, CI_LEVEL, N_BOOTSTRAP,
    MIN_COMPLETE_CALLS, MIN_NOT_COMPLETE_CALLS,
    THRESHOLD_GRID_STEP, DEFAULT_T_LOW, DEFAULT_T_HIGH,
    CV_N_REPEATS, CV_N_SPLITS, CV_N_JOBS, REPORTS_DIR, APP_DIR,
    MODEL_VERSION, MODEL_NAME, MISSING_COLS,
)
from src.features import (
    get_feature_columns, get_ml_feature_columns, build_preprocessor,
    prepare_dataframe,
)
from src.rules import (
    apply_hard_stops, report_hard_stop_stats, rule_based_protocol,
    COMPLETE, NOT_COMPLETE, UNCERTAIN,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ── Splitting ────────────────────────────────────────────────────────────
def load_and_split():
    """Load data and do stratified train/val/test split (fixed seed)."""
    df = pd.read_csv(DATA_FILE)
    y_full = df[TARGET_COL]

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_RATIO, random_state=SEED
    )
    trainval_idx, test_idx = next(sss1.split(df, y_full))

    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=SEED
    )
    df_trainval = df.iloc[trainval_idx]
    y_trainval = y_full.iloc[trainval_idx]
    train_sub_idx, val_sub_idx = next(sss2.split(df_trainval, y_trainval))

    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    return df, train_idx, val_idx, test_idx


def filter_hard_stops(df, indices):
    """Return indices of patients NOT caught by Layer 1 hard stops."""
    sub = df.iloc[indices]
    sub_stops = apply_hard_stops(sub)
    remaining_mask = ~sub_stops["hard_stop_fired"]
    return indices[remaining_mask.values]


def prepare_splits(df, train_idx, val_idx, test_idx, feature_cols=None):
    """Prepare X, y for each split. Targets and patient_id are dropped from X."""
    if feature_cols is None:
        feature_cols = get_ml_feature_columns()

    df_prep = prepare_dataframe(df, feature_cols, drop_targets=True)
    miss_cols = [f"{c}_missing" for c in MISSING_COLS if c in feature_cols]
    all_cols = feature_cols + miss_cols

    X_train = df_prep.iloc[train_idx]
    y_train = df[TARGET_COL].iloc[train_idx]
    X_val = df_prep.iloc[val_idx]
    y_val = df[TARGET_COL].iloc[val_idx]
    X_test = df_prep.iloc[test_idx]
    y_test = df[TARGET_COL].iloc[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test, all_cols


# ── Model builders ───────────────────────────────────────────────────────
def build_base_model(model_name: str):
    """Return an unfitted base estimator."""
    if model_name == "logistic_l2":
        return LogisticRegression(
            C=1.0, max_iter=2000, random_state=SEED, solver="lbfgs"
        )
    if model_name == "logistic_l1":
        return LogisticRegression(
            C=1.0, max_iter=3000, random_state=SEED,
            solver="saga", l1_ratio=1,
        )
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=20, random_state=SEED,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20,
        )
    if model_name == "hgb_conservative":
        return HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.05,
            min_samples_leaf=30, random_state=SEED,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20,
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_model(model_name: str, calibration: str = "none"):
    """Return a base or calibrated estimator."""
    base = build_base_model(model_name)
    if calibration == "none":
        return base
    if calibration in {"sigmoid", "isotonic"}:
        return CalibratedClassifierCV(base, method=calibration, cv=5)
    raise ValueError(f"Unknown calibration: {calibration}")


def build_pipeline(feature_cols, model_name: str, calibration: str = "none"):
    return Pipeline([
        ("preprocessor", build_preprocessor(feature_cols)),
        ("model", build_model(model_name, calibration)),
    ])


def build_candidates():
    return {
        "logistic_l2": build_base_model("logistic_l2"),
        "logistic_l1": build_base_model("logistic_l1"),
        "hgb": build_base_model("hgb"),
        "hgb_conservative": build_base_model("hgb_conservative"),
    }


# ── CV helpers ───────────────────────────────────────────────────────────
def _cv_predictions(X, y, pipe, n_repeats=CV_N_REPEATS,
                    n_splits=CV_N_SPLITS):
    """Return list of OOF probability arrays, one per repeat."""
    probas = []
    for rep in range(n_repeats):
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=SEED + rep
        )
        p = cross_val_predict(
            pipe, X, y, cv=cv, method="predict_proba", n_jobs=CV_N_JOBS
        )[:, 1]
        probas.append(p)
    return probas


def run_cv_comparison(X, y, preprocessor, candidates):
    """Repeated stratified CV comparison of candidate models (uncalibrated)."""
    results = {}
    for name, model in candidates.items():
        print(f"  CV: {name}...")
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        aurocs, auprcs, briers, loglosses = [], [], [], []
        try:
            probas = _cv_predictions(X, y, pipe)
        except Exception as exc:
            print(f"    failed: {exc}")
            results[name] = {"error": str(exc)}
            continue

        for p in probas:
            aurocs.append(roc_auc_score(y, p))
            auprcs.append(average_precision_score(y, p))
            briers.append(brier_score_loss(y, p))
            loglosses.append(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))

        results[name] = {
            "auroc": round(float(np.mean(aurocs)), 4),
            "auroc_std": round(float(np.std(aurocs)), 4),
            "auprc": round(float(np.mean(auprcs)), 4),
            "auprc_std": round(float(np.std(auprcs)), 4),
            "brier": round(float(np.mean(briers)), 4),
            "brier_std": round(float(np.std(briers)), 4),
            "log_loss": round(float(np.mean(loglosses)), 4),
        }
        print(f"    AUROC={results[name]['auroc']:.4f}"
              f"±{results[name]['auroc_std']:.4f}  "
              f"AUPRC={results[name]['auprc']:.4f}  "
              f"Brier={results[name]['brier']:.4f}")

    return results


def run_calibration_comparison(X, y, feature_cols, model_name):
    """Compare no calibration, sigmoid, isotonic by repeated CV."""
    results = {}
    for calibration in ["none", "sigmoid", "isotonic"]:
        print(f"  Calibration: {calibration}...")
        pipe = build_pipeline(feature_cols, model_name, calibration)
        try:
            probas = _cv_predictions(X, y, pipe)
        except Exception as exc:
            print(f"    failed: {exc}")
            results[calibration] = {"error": str(exc)}
            continue

        aurocs, briers, loglosses = [], [], []
        for p in probas:
            aurocs.append(roc_auc_score(y, p))
            briers.append(brier_score_loss(y, p))
            loglosses.append(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))

        results[calibration] = {
            "auroc": round(float(np.mean(aurocs)), 4),
            "brier": round(float(np.mean(briers)), 4),
            "brier_std": round(float(np.std(briers)), 4),
            "log_loss": round(float(np.mean(loglosses)), 4),
        }
        print(f"    Brier={results[calibration]['brier']:.4f}  "
              f"LogLoss={results[calibration]['log_loss']:.4f}")

    # Choose by Brier; a small difference is broken in favour of simplicity.
    valid = {k: v for k, v in results.items() if "brier" in v}
    if not valid:
        return results, "none"

    best_brier = min(v["brier"] for v in valid.values())
    close = {k: v for k, v in valid.items() if v["brier"] <= best_brier + 0.001}
    order = {"none": 0, "sigmoid": 1, "isotonic": 2}
    best = min(close, key=lambda k: (order[k], close[k]["brier"]))
    return results, best


# ── Threshold selection ──────────────────────────────────────────────────
def _bootstrap_lower_bound(values: np.ndarray, n_boot: int = N_BOOTSTRAP,
                           ci: float = CI_LEVEL, seed: int = SEED):
    """Lower bound of the mean of `values` by percentile bootstrap."""
    if len(values) < 2:
        return None
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(values)
    for i in range(n_boot):
        sample = values[rng.randint(0, n, n)]
        means[i] = sample.mean()
    alpha = (1 - ci) / 2
    return float(np.percentile(means, alpha * 100))


def find_thresholds_oof(y, proba, ppv_target=PPV_TARGET,
                        npv_target=NPV_TARGET):
    """
    Select t_high and t_low on OOF probabilities.

    t_high: smallest threshold whose PPV bootstrap lower bound meets the
            PPV target and which yields at least MIN_COMPLETE_CALLS calls.
    t_low : largest threshold whose NPV bootstrap lower bound meets the
            NPV target and which yields at least MIN_NOT_COMPLETE_CALLS calls.
    """
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)

    scan_high = []
    chosen_high = None
    best_high_lower = -1.0
    for t in np.arange(0.50, 0.991, THRESHOLD_GRID_STEP):
        selected = proba >= t
        n = int(selected.sum())
        if n < MIN_COMPLETE_CALLS:
            continue
        vals = y[selected]
        ppv = float(vals.mean())
        lower = _bootstrap_lower_bound(vals, seed=SEED + int(round(t * 1000)))
        scan_high.append({
            "t": round(float(t), 3), "n": n, "ppv": round(ppv, 4),
            "ppv_ci_lower": None if lower is None else round(lower, 4),
        })
        if lower is not None and lower > best_high_lower:
            best_high_lower = lower
        if chosen_high is None and lower is not None and lower >= ppv_target:
            chosen_high = float(t)

    fallback_high = chosen_high is None
    if fallback_high:
        if not scan_high:
            chosen_high = DEFAULT_T_HIGH
        else:
            chosen_high = max(
                scan_high,
                key=lambda r: (r["ppv_ci_lower"] or -1, r["n"]),
            )["t"]

    scan_low = []
    chosen_low = None
    best_low_lower = -1.0
    for t in np.arange(0.49, 0.009, -THRESHOLD_GRID_STEP):
        selected = proba <= t
        n = int(selected.sum())
        if n < MIN_NOT_COMPLETE_CALLS:
            continue
        vals = 1 - y[selected]
        npv = float(vals.mean())
        lower = _bootstrap_lower_bound(vals, seed=SEED + int(round(t * 1000)))
        scan_low.append({
            "t": round(float(t), 3), "n": n, "npv": round(npv, 4),
            "npv_ci_lower": None if lower is None else round(lower, 4),
        })
        if lower is not None and lower > best_low_lower:
            best_low_lower = lower
        if chosen_low is None and lower is not None and lower >= npv_target:
            chosen_low = float(t)

    fallback_low = chosen_low is None
    if fallback_low:
        if not scan_low:
            chosen_low = DEFAULT_T_LOW
        else:
            chosen_low = max(
                scan_low,
                key=lambda r: (r["npv_ci_lower"] or -1, r["n"]),
            )["t"]

    adjusted = False
    if chosen_low >= chosen_high:
        chosen_low = max(0.05, chosen_high - 0.10)
        adjusted = True

    chosen_high = round(float(chosen_high), 3)
    chosen_low = round(float(chosen_low), 3)

    n_complete = int((proba >= chosen_high).sum())
    n_not_complete = int((proba <= chosen_low).sum())
    n_uncertain = int(len(proba) - n_complete - n_not_complete)
    stats = {
        "method": "OOF bootstrap lower-bound rule",
        "t_low": chosen_low,
        "t_high": chosen_high,
        "ppv_target": ppv_target,
        "npv_target": npv_target,
        "ci_level": CI_LEVEL,
        "n_bootstrap": N_BOOTSTRAP,
        "min_complete_calls": MIN_COMPLETE_CALLS,
        "min_not_complete_calls": MIN_NOT_COMPLETE_CALLS,
        "n_complete": n_complete,
        "n_not_complete": n_not_complete,
        "n_uncertain": n_uncertain,
        "coverage_pct": round(
            (n_complete + n_not_complete) / len(proba) * 100, 1
        ),
        "ppv_complete": round(float(y[proba >= chosen_high].mean()), 4)
        if n_complete else None,
        "npv_not_complete": round(float((1 - y[proba <= chosen_low]).mean()), 4)
        if n_not_complete else None,
        "fallback_high": fallback_high,
        "fallback_low": fallback_low,
        "adjusted_gap": adjusted,
        "scan_high": scan_high,
        "scan_low": scan_low,
    }
    return chosen_low, chosen_high, stats


def confirm_thresholds(y, proba, t_low, t_high):
    """Confirmation of the selected thresholds on the validation split."""
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    complete = proba >= t_high
    not_complete = proba <= t_low
    uncertain = ~complete & ~not_complete

    def _ci(values, seed):
        return _bootstrap_lower_bound(values, n_boot=1000, seed=seed)

    ppv = float(y[complete].mean()) if complete.sum() else None
    npv = float((1 - y[not_complete]).mean()) if not_complete.sum() else None
    ppv_lower = _ci(y[complete], SEED + 11) if complete.sum() else None
    npv_lower = _ci(1 - y[not_complete], SEED + 12) if not_complete.sum() else None

    return {
        "n": int(len(y)),
        "n_complete": int(complete.sum()),
        "n_not_complete": int(not_complete.sum()),
        "n_uncertain": int(uncertain.sum()),
        "coverage_pct": round(
            (complete.sum() + not_complete.sum()) / len(y) * 100, 1
        ),
        "ppv_complete": None if ppv is None else round(ppv, 4),
        "ppv_ci_lower": None if ppv_lower is None else round(ppv_lower, 4),
        "npv_not_complete": None if npv is None else round(npv, 4),
        "npv_ci_lower": None if npv_lower is None else round(npv_lower, 4),
        "passes_ppv_rule": bool(
            ppv_lower is not None and ppv_lower >= PPV_TARGET
        ),
        "passes_npv_rule": bool(
            npv_lower is not None and npv_lower >= NPV_TARGET
        ),
    }


# ── Persistence helpers ──────────────────────────────────────────────────
def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.str_, np.bytes_)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_app_defaults(df, train_idx, feature_cols=None):
    """Save clinically plausible defaults for the app (training split only)."""
    if feature_cols is None:
        feature_cols = get_feature_columns()

    train_df = df.iloc[train_idx]
    values = {}
    for col in feature_cols:
        if col not in train_df.columns:
            values[col] = None
            continue
        series = train_df[col]
        non_null = series.dropna()
        if non_null.empty:
            values[col] = None
            continue
        # Discrete columns (binary codes, stages, small ordinal scales) use
        # the mode so that the app default is always a valid widget option.
        if non_null.nunique() <= 10:
            values[col] = non_null.mode().iloc[0]
        elif pd.api.types.is_numeric_dtype(series):
            values[col] = float(non_null.median())
        else:
            values[col] = non_null.mode().iloc[0]

    payload = {
        "values": _jsonable(values),
        "all_feature_cols": list(feature_cols),
        "ml_feature_cols": list(get_ml_feature_columns()),
        "missing_cols": list(MISSING_COLS),
        "seed": SEED,
    }
    APP_DIR.mkdir(parents=True, exist_ok=True)
    path = APP_DIR / "defaults.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


# ── Main ─────────────────────────────────────────────────────────────────
def run_training():
    print("=" * 60)
    print("TRAINING: rules -> model selection -> calibration -> thresholds")
    print("=" * 60)

    # ── Load and split ───────────────────────────────────────────────
    print("\n[1/7] Loading data and splitting...")
    df, train_idx, val_idx, test_idx = load_and_split()
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    rule_stats = report_hard_stop_stats(df)
    print(f"  Hard stops: {rule_stats['_combined']['n_stopped']} patients; "
          f"ML remaining: {rule_stats['_combined']['n_remaining']}")

    train_idx_ml = filter_hard_stops(df, train_idx)
    val_idx_ml = filter_hard_stops(df, val_idx)
    print(f"  ML train: {len(train_idx_ml)}  ML val: {len(val_idx_ml)}")

    # ── Prepare features ─────────────────────────────────────────────
    ml_feature_cols = get_ml_feature_columns()
    X_train, y_train, X_val, y_val, X_test, y_test, all_cols = prepare_splits(
        df, train_idx_ml, val_idx_ml, test_idx, ml_feature_cols
    )
    print(f"  ML features: {len(ml_feature_cols)} (+{len(all_cols) - len(ml_feature_cols)} missing indicators)")

    # ── Baselines on validation ──────────────────────────────────────
    print("\n[2/7] Baselines on validation (training-only fitting)...")
    majority = int(y_train.mode().iloc[0])
    b0_pred = np.full(len(y_val), majority)
    b0_acc = float((b0_pred == y_val.values).mean())
    print(f"  B0 majority ({majority}): accuracy={b0_acc:.4f}")

    b1_val = rule_based_protocol(df.iloc[val_idx_ml])
    b1_auroc = float(roc_auc_score(y_val, b1_val))
    b1_auprc = float(average_precision_score(y_val, b1_val))
    b1_ppv = float(precision_score(y_val, b1_val, zero_division=0))
    b1_npv = float(recall_score(1 - y_val, 1 - b1_val, zero_division=0))
    print(f"  B1 rule protocol: AUROC={b1_auroc:.4f}  AUPRC={b1_auprc:.4f}  "
          f"PPV={b1_ppv:.4f}  NPV={b1_npv:.4f}")

    b2_features = [
        "mri_mrTRG", "mri_volume_reduction_pct",
        "endo_pct_wall_involved", "mri_wall_thickness_mm",
        "mri_DWI_restriction", "mri_tumor_volume_ml",
        "dre_finding", "endo_finding",
    ]
    X_train_b2, y_train_b2, X_val_b2, y_val_b2, _, _, _ = prepare_splits(
        df, train_idx_ml, val_idx_ml, test_idx, b2_features
    )
    pipe_b2 = Pipeline([
        ("preprocessor", build_preprocessor(b2_features)),
        ("model", build_base_model("logistic_l2")),
    ])
    pipe_b2.fit(X_train_b2, y_train_b2)
    b2_val = pipe_b2.predict_proba(X_val_b2)[:, 1]
    b2_auroc = float(roc_auc_score(y_val_b2, b2_val))
    b2_auprc = float(average_precision_score(y_val_b2, b2_val))
    print(f"  B2 logreg 8-feature: AUROC={b2_auroc:.4f}  AUPRC={b2_auprc:.4f}")

    # ── Model comparison ─────────────────────────────────────────────
    print("\n[3/7] Model comparison (repeated 5x5 CV on training)...")
    preprocessor = build_preprocessor(ml_feature_cols)
    candidates = build_candidates()
    cv_results = run_cv_comparison(X_train, y_train, preprocessor, candidates)

    valid_cv = {k: v for k, v in cv_results.items() if "auroc" in v}
    if not valid_cv:
        raise RuntimeError("All candidate models failed cross-validation.")
    best_name = max(
        valid_cv,
        key=lambda k: (valid_cv[k]["auroc"], -valid_cv[k]["brier"]),
    )
    print(f"  Best model: {best_name} (CV AUROC={valid_cv[best_name]['auroc']:.4f})")

    # ── Calibration selection ────────────────────────────────────────
    print("\n[4/7] Calibration comparison (repeated 5x5 CV on training)...")
    calibration_results, best_calibration = run_calibration_comparison(
        X_train, y_train, ml_feature_cols, best_name
    )
    print(f"  Selected calibration: {best_calibration}")

    # ── OOF predictions and thresholds ───────────────────────────────
    print("\n[5/7] OOF predictions and threshold selection...")
    oof_pipe = build_pipeline(ml_feature_cols, best_name, best_calibration)
    oof_probas = _cv_predictions(X_train, y_train, oof_pipe)
    oof_proba = np.mean(oof_probas, axis=0)
    oof_auroc = float(roc_auc_score(y_train, oof_proba))
    oof_brier = float(brier_score_loss(y_train, oof_proba))
    print(f"  OOF AUROC={oof_auroc:.4f}  Brier={oof_brier:.4f}")

    t_low, t_high, threshold_stats = find_thresholds_oof(y_train, oof_proba)
    print(f"  Selected thresholds: t_low={t_low:.3f}  t_high={t_high:.3f}")
    print(f"  OOF coverage={threshold_stats['coverage_pct']}%  "
          f"PPV={threshold_stats['ppv_complete']}  "
          f"NPV={threshold_stats['npv_not_complete']}")

    # Save OOF predictions for auditability.
    oof_out = pd.DataFrame({
        "patient_id": df[ID_COL].iloc[train_idx_ml].values,
        "y_true": y_train.values,
        "oof_probability": oof_proba,
    })
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    oof_out.to_csv(REPORTS_DIR / "oof_predictions.csv", index=False)

    # ── Final model ──────────────────────────────────────────────────
    print("\n[6/7] Training final model on the full ML training split...")
    final_pipe = build_pipeline(ml_feature_cols, best_name, best_calibration)
    final_pipe.fit(X_train, y_train)

    val_proba = final_pipe.predict_proba(X_val)[:, 1]
    val_auroc = float(roc_auc_score(y_val, val_proba))
    val_brier = float(brier_score_loss(y_val, val_proba))
    print(f"  Validation AUROC={val_auroc:.4f}  Brier={val_brier:.4f}")

    confirmation = confirm_thresholds(y_val, val_proba, t_low, t_high)
    print(f"  Validation confirmation: PPV={confirmation['ppv_complete']} "
          f"(lower {confirmation['ppv_ci_lower']}), "
          f"NPV={confirmation['npv_not_complete']} "
          f"(lower {confirmation['npv_ci_lower']}), "
          f"coverage={confirmation['coverage_pct']}%")

    # ── Save ─────────────────────────────────────────────────────────
    print("\n[7/7] Saving model, config and app defaults...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = REPORTS_DIR / "final_model.joblib"
    joblib.dump(final_pipe, model_path)

    config_out = {
        "model_version": MODEL_VERSION,
        "best_model": best_name,
        "calibration": best_calibration,
        "t_low": float(t_low),
        "t_high": float(t_high),
        "threshold_selection": threshold_stats,
        "validation_confirmation": confirmation,
        "cv_results": cv_results,
        "calibration_results": calibration_results,
        "baselines_validation": {
            "B0_majority_class": majority,
            "B0_accuracy": round(b0_acc, 4),
            "B1_rule_auroc": round(b1_auroc, 4),
            "B1_rule_auprc": round(b1_auprc, 4),
            "B1_rule_ppv": round(b1_ppv, 4),
            "B1_rule_npv": round(b1_npv, 4),
            "B2_logreg8_auroc": round(b2_auroc, 4),
            "B2_logreg8_auprc": round(b2_auprc, 4),
        },
        "val_auroc": round(val_auroc, 4),
        "val_brier": round(val_brier, 4),
        "oof_auroc": round(oof_auroc, 4),
        "oof_brier": round(oof_brier, 4),
        "rule_stats": rule_stats,
        "ml_feature_cols": ml_feature_cols,
        "all_feature_cols": all_cols,
        "train_size": len(train_idx_ml),
        "val_size": len(val_idx_ml),
        "train_full_size": len(train_idx),
        "val_full_size": len(val_idx),
        "test_size": len(test_idx),
        "seed": SEED,
        "data_file": str(DATA_FILE),
        "model_file": "final_model.joblib",
    }

    with open(REPORTS_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(config_out), f, indent=2)

    np.savez(
        REPORTS_DIR / "split_indices.npz",
        train=train_idx, val=val_idx, test=test_idx,
        train_ml=train_idx_ml, val_ml=val_idx_ml,
    )

    defaults_path = save_app_defaults(df, train_idx)

    print("\nDone.")
    print(f"  Model: {model_path}")
    print(f"  Config: {REPORTS_DIR / 'train_config.json'}")
    print(f"  OOF: {REPORTS_DIR / 'oof_predictions.csv'}")
    print(f"  App defaults: {defaults_path}")
    print(f"  Thresholds: t_low={t_low:.3f}  t_high={t_high:.3f}")

    return final_pipe, config_out


if __name__ == "__main__":
    run_training()
