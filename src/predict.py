"""
Prediction: single-patient and batch inference with
Layer 1 (rules) -> Layer 2 (ML) -> Layer 3 (abstention + missing-exam policy).
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.config import REPORTS_DIR, MODEL_VERSION
from src.features import get_ml_feature_columns, prepare_dataframe
from src.rules import (
    apply_hard_stops, apply_hard_stops_single, missing_exam_policy,
    decide_from_probability, COMPLETE, NOT_COMPLETE, UNCERTAIN,
)
from src.schema import validate_patient
from src.audit import log_decision


def load_model_and_config():
    """Load the trained model and configuration with clear error messages."""
    model_path = REPORTS_DIR / "final_model.joblib"
    config_path = REPORTS_DIR / "train_config.json"
    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python -m src.train` first."
        )
    pipe = joblib.load(model_path)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return pipe, config


def _predict_proba(patient: dict, pipe, ml_feature_cols: list[str]):
    """Return the model probability for one patient, or None if unavailable."""
    try:
        df_single = pd.DataFrame([patient])
        df_prep = prepare_dataframe(df_single, ml_feature_cols)
        return float(pipe.predict_proba(df_prep)[:, 1][0])
    except Exception:
        return None


def _base_result(decision: str, probability, confidence: str, source: str,
                 rule, abstained: bool, unavailable_exams: list[str],
                 policy_reason: str, t_low: float, t_high: float) -> dict:
    return {
        "decision": decision,
        "probability": None if probability is None else round(float(probability), 4),
        "confidence": confidence,
        "source": source,
        "rule": rule,
        "abstained": abstained,
        "unavailable_exams": unavailable_exams,
        "policy_reason": policy_reason,
        "t_low": float(t_low),
        "t_high": float(t_high),
        "model_version": MODEL_VERSION,
    }


def predict_single(patient: dict, pipe=None, config=None, log: bool = True,
                   validate: bool = False, audit_log_path: Path | None = None):
    """
    Full inference pipeline for a single patient.

    Returns a dict with:
      decision, probability, confidence, source, rule, abstained,
      unavailable_exams, policy_reason, t_low, t_high, model_version
    """
    if pipe is None or config is None:
        pipe, config = load_model_and_config()

    t_low = float(config["t_low"])
    t_high = float(config["t_high"])
    ml_feature_cols = config["ml_feature_cols"]

    patient = dict(patient)

    if validate:
        errors = validate_patient(patient, ml_feature_cols, allow_missing=True)
        if errors:
            raise ValueError("Invalid patient input:\n- " + "\n- ".join(errors))

    # ── Layer 1: deterministic hard stops ────────────────────────────────
    hard_stop = apply_hard_stops_single(patient)
    if hard_stop["fired"]:
        result = _base_result(
            NOT_COMPLETE, None, "HIGH", "rule", hard_stop["rule"],
            False, [], "", t_low, t_high,
        )
        if log:
            log_decision(patient.get("patient_id", "unknown"), result,
                         log_path=audit_log_path)
        return result

    # ── Layer 3a: missing-exam safety policy ─────────────────────────────
    policy = missing_exam_policy(patient)

    # Compute the model probability even when the policy blocks a definite
    # call, so the app can show it for transparency.
    proba = _predict_proba(patient, pipe, ml_feature_cols)

    if policy["blocked"]:
        result = _base_result(
            UNCERTAIN, proba, "LOW", "policy", policy["reason"],
            True, policy["unavailable_exams"], policy["reason"], t_low, t_high,
        )
        if log:
            log_decision(patient.get("patient_id", "unknown"), result,
                         log_path=audit_log_path)
        return result

    # ── Layer 2/3b: model probability and abstention band ────────────────
    if proba is None:
        result = _base_result(
            UNCERTAIN, None, "LOW", "model",
            "Model probability could not be computed; referred to the MDT.",
            True, [], "", t_low, t_high,
        )
        if log:
            log_decision(patient.get("patient_id", "unknown"), result,
                         log_path=audit_log_path)
        return result

    decision = decide_from_probability(proba, t_low, t_high)
    result = _base_result(
        decision, proba, "LOW" if decision == UNCERTAIN else "HIGH",
        "model", None, decision == UNCERTAIN, [], "", t_low, t_high,
    )
    if log:
        log_decision(patient.get("patient_id", "unknown"), result,
                     log_path=audit_log_path)
    return result


def predict_batch(df: pd.DataFrame, pipe=None, config=None, log: bool = False,
                  validate: bool = False, audit_log_path: Path | None = None):
    """
    Batch prediction for a DataFrame of patients.

    Returns the original rows plus:
      decision, probability, confidence, source, rule, abstained,
      unavailable_exams, policy_reason
    """
    if pipe is None or config is None:
        pipe, config = load_model_and_config()

    t_low = float(config["t_low"])
    t_high = float(config["t_high"])
    ml_feature_cols = config["ml_feature_cols"]

    df = df.copy()
    if validate:
        errors = []
        for i, (_, row) in enumerate(df.iterrows()):
            row_errors = validate_patient(
                row.to_dict(), ml_feature_cols, allow_missing=True
            )
            if row_errors:
                errors.append(f"row {i}: " + "; ".join(row_errors))
        if errors:
            raise ValueError("Invalid batch input:\n- " + "\n- ".join(errors[:20]))

    # ── Layer 1: hard stops ──────────────────────────────────────────────
    stops = apply_hard_stops(df)
    hard_stop_fired = stops["hard_stop_fired"].values
    hard_stop_rule = stops["hard_stop_rule"].values

    # ── Layer 2: model probabilities for non-stopped rows ────────────────
    proba = np.full(len(df), np.nan, dtype=float)
    ml_mask = ~hard_stop_fired
    if ml_mask.sum() > 0:
        df_ml = df.loc[ml_mask]
        try:
            df_prep = prepare_dataframe(df_ml, ml_feature_cols)
            proba[ml_mask] = pipe.predict_proba(df_prep)[:, 1]
        except Exception:
            proba[ml_mask] = np.nan

    # ── Layer 3: policy + abstention, row by row ─────────────────────────
    rows = []
    for pos, (_, row) in enumerate(df.iterrows()):
        row_dict = row.to_dict()
        if hard_stop_fired[pos]:
            rows.append({
                "decision": NOT_COMPLETE,
                "probability": None,
                "confidence": "HIGH",
                "source": "rule",
                "rule": hard_stop_rule[pos],
                "abstained": False,
                "unavailable_exams": [],
                "policy_reason": "",
            })
            continue

        policy = missing_exam_policy(row_dict)
        p = None if np.isnan(proba[pos]) else float(proba[pos])

        if policy["blocked"]:
            rows.append({
                "decision": UNCERTAIN,
                "probability": p,
                "confidence": "LOW",
                "source": "policy",
                "rule": policy["reason"],
                "abstained": True,
                "unavailable_exams": policy["unavailable_exams"],
                "policy_reason": policy["reason"],
            })
        elif p is None:
            rows.append({
                "decision": UNCERTAIN,
                "probability": None,
                "confidence": "LOW",
                "source": "model",
                "rule": "Model probability could not be computed; referred to the MDT.",
                "abstained": True,
                "unavailable_exams": [],
                "policy_reason": "",
            })
        else:
            decision = decide_from_probability(p, t_low, t_high)
            rows.append({
                "decision": decision,
                "probability": p,
                "confidence": "LOW" if decision == UNCERTAIN else "HIGH",
                "source": "model",
                "rule": None,
                "abstained": decision == UNCERTAIN,
                "unavailable_exams": [],
                "policy_reason": "",
            })

    results = pd.DataFrame(rows, index=df.index)

    if log:
        for pos, (idx, row) in enumerate(df.iterrows()):
            patient_id = row.get("patient_id", f"BATCH_ROW_{pos}")
            res = results.iloc[pos].to_dict()
            res["model_version"] = MODEL_VERSION
            log_decision(str(patient_id), res, log_path=audit_log_path)

    return pd.concat([df, results], axis=1)


def get_example_patients(df: pd.DataFrame, test_idx, pipe=None, config=None,
                         n_each: int = 2):
    """
    Select example patients from the test set for the app.

    This is a deterministic, model-based selection (not hand-picked):
      - COMPLETE: highest model probabilities among COMPLETE calls
      - NOT COMPLETE: one rule-based and one model-based (lowest probability)
      - UNCERTAIN: probability closest to the centre of the abstention band,
        if any; otherwise the patient closest to the band.
    """
    if pipe is None or config is None:
        pipe, config = load_model_and_config()

    test_df = df.iloc[test_idx].copy()
    batch = predict_batch(test_df, pipe, config, log=False)

    t_low = float(config["t_low"])
    t_high = float(config["t_high"])
    centre = (t_low + t_high) / 2

    examples = []

    complete_mask = batch["decision"] == COMPLETE
    complete_idx = batch[complete_mask].sort_values(
        "probability", ascending=False
    ).index[:n_each]
    for idx in complete_idx:
        examples.append({
            "patient_id": str(test_df.loc[idx, "patient_id"]),
            "index": int(idx),
            "expected_decision": COMPLETE,
            "selection_rule": "highest model probability among COMPLETE calls",
            "data": test_df.loc[idx].to_dict(),
        })

    model_not_complete = batch[
        (batch["decision"] == NOT_COMPLETE) & (batch["source"] == "model")
    ].sort_values("probability", ascending=True)
    n_model_not_complete = max(1, n_each - 1)
    for idx in model_not_complete.index[:n_model_not_complete]:
        examples.append({
            "patient_id": str(test_df.loc[idx, "patient_id"]),
            "index": int(idx),
            "expected_decision": NOT_COMPLETE,
            "selection_rule": "lowest model probability among model NOT COMPLETE calls",
            "data": test_df.loc[idx].to_dict(),
        })

    rule_not_complete = batch[batch["source"] == "rule"]
    if len(rule_not_complete) > 0:
        idx = rule_not_complete.index[0]
        examples.append({
            "patient_id": str(test_df.loc[idx, "patient_id"]),
            "index": int(idx),
            "expected_decision": NOT_COMPLETE,
            "selection_rule": "first rule-based NOT COMPLETE call",
            "data": test_df.loc[idx].to_dict(),
        })

    uncertain = batch[
        (batch["decision"] == UNCERTAIN) & batch["probability"].notna()
    ]
    if len(uncertain) > 0:
        idx = (uncertain["probability"].astype(float) - centre).abs().sort_values().index[0]
        examples.append({
            "patient_id": str(test_df.loc[idx, "patient_id"]),
            "index": int(idx),
            "expected_decision": UNCERTAIN,
            "selection_rule": "closest probability to the centre of the abstention band",
            "data": test_df.loc[idx].to_dict(),
        })
    else:
        # Honest fallback: no test patient fell in the band under the deployed
        # thresholds.  Select the model-scored patient closest to the band and
        # document that the what-if panel (exam unavailable) is the way to
        # demonstrate abstention for this example.  Hard-stop rows have no
        # model probability and must not be selected here.
        p = batch["probability"].astype(float)
        model_rows = p.notna()
        if model_rows.any():
            p_model = p[model_rows]
            distance = np.where(p_model < t_low, t_low - p_model,
                                np.where(p_model > t_high, p_model - t_high, 0.0))
            idx = pd.Series(distance, index=p_model.index).sort_values().index[0]
        else:
            idx = batch.index[0]
        examples.append({
            "patient_id": str(test_df.loc[idx, "patient_id"]),
            "index": int(idx),
            "expected_decision": str(batch.loc[idx, "decision"]),
            "selection_rule": "closest probability to the abstention band (no UNCERTAIN case in the test set)",
            "note": ("No test patient was UNCERTAIN under the deployed thresholds; "
                     "mark an exam unavailable in the what-if panel to see abstention."),
            "what_if_exam": "mri",
            "data": test_df.loc[idx].to_dict(),
        })

    return examples
