"""
Layer 1: Deterministic rules – hard-stop safety overrides and input validation.

No AI here. Each rule has a clinical justification.
If ANY hard stop fires → patient is NOT safe for watch-and-wait (forced NOT COMPLETE).
"""
import pandas as pd
import numpy as np
from src.config import (
    HARD_STOP_RULES, TARGET_COL, REQUIRED_EXAMS,
    EXAM_AVAILABILITY_COLUMNS, EXAM_DISPLAY_NAMES,
)

# Decision labels used by Layer 3 and the app.
COMPLETE = "COMPLETE RESPONSE"
NOT_COMPLETE = "NOT COMPLETE"
UNCERTAIN = "UNCERTAIN"


def apply_hard_stops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply hard-stop rules. Returns df with added columns:
      - hard_stop_fired: bool
      - hard_stop_rule: str (which rule, or '')
      - hard_stop_decision: 0 (NOT COMPLETE) or NaN (proceed to ML)
    """
    df = df.copy()
    df["hard_stop_fired"] = False
    df["hard_stop_rule"] = ""
    df["hard_stop_decision"] = np.nan

    for col, op, val, reason in HARD_STOP_RULES:
        if col not in df.columns:
            continue
        if op == "==":
            mask = df[col] == val
        else:
            raise ValueError(f"Unsupported operator: {op}")

        # Only mark patients not already stopped (first rule wins)
        new_stops = mask & ~df["hard_stop_fired"]
        df.loc[new_stops, "hard_stop_fired"] = True
        df.loc[new_stops, "hard_stop_rule"] = reason
        df.loc[new_stops, "hard_stop_decision"] = 0  # NOT safe

    return df


def apply_hard_stops_single(patient: dict) -> dict:
    """
    Apply hard-stop rules to a single patient dict.
    Returns {"fired": bool, "rule": str, "decision": 0 or None}.
    """
    for col, op, val, reason in HARD_STOP_RULES:
        patient_val = patient.get(col)
        if pd.isna(patient_val):
            continue
        if op == "==" and patient_val == val:
            return {"fired": True, "rule": reason, "decision": 0}
        if op == "==" and str(patient_val) == str(val):
            return {"fired": True, "rule": reason, "decision": 0}
    return {"fired": False, "rule": "", "decision": None}


def decide_from_probability(proba, t_low: float, t_high: float) -> str:
    """Map a calibrated probability to one of the three decisions."""
    if proba is None or pd.isna(proba):
        return UNCERTAIN
    if proba >= t_high:
        return COMPLETE
    if proba <= t_low:
        return NOT_COMPLETE
    return UNCERTAIN


def exam_availability(patient: dict) -> dict[str, bool]:
    """Return {exam: available} for the five required exams.

    An exam is available when its key safety column is present.  For example,
    pelvic MRI is available when mrTRG is recorded; CT is available when
    distant progression status is recorded.  Other missing sub-fields are
    handled by imputation.
    """
    availability = {}
    for exam in REQUIRED_EXAMS:
        cols = EXAM_AVAILABILITY_COLUMNS[exam]
        availability[exam] = all(not pd.isna(patient.get(c)) for c in cols)
    return availability


def missing_exam_policy(patient: dict) -> dict:
    """
    Safety policy: a COMPLETE call requires all five exams to be available.

    If an exam is unavailable, the tool cannot clear that exam's safety check,
    so the case is referred to the MDT (UNCERTAIN) unless another hard stop
    already forces NOT COMPLETE.  This is deliberately conservative.
    """
    availability = exam_availability(patient)
    unavailable = [e for e in REQUIRED_EXAMS if not availability[e]]
    if not unavailable:
        return {"blocked": False, "unavailable_exams": [], "reason": ""}

    names = ", ".join(EXAM_DISPLAY_NAMES.get(e, e) for e in unavailable)
    reason = (
        f"Exam(s) unavailable: {names}. A COMPLETE call requires all five "
        "exams to clear their safety checks; the case is referred to the MDT."
    )
    return {"blocked": True, "unavailable_exams": unavailable, "reason": reason}


def rule_based_protocol(df: pd.DataFrame) -> pd.Series:
    """
    Baseline B1: simple rule-based clinical protocol for complete response.
    A clinician would call 'complete' if ALL of:
      - mrTRG <= 2
      - DRE: no_abnormality or smooth_scar
      - Endoscopy: flat_white_scar or scar_with_telangiectasia
      - Biopsy: not positive (negative, inadequate, or not_done)
      - CT: no distant progression

    Returns a Series of predicted labels (1 = safe, 0 = not safe).
    """
    pred = (
        (df["mri_mrTRG"] <= 2) &
        (df["dre_finding"].isin(["no_abnormality", "smooth_scar"])) &
        (df["endo_finding"].isin(["flat_white_scar", "scar_with_telangiectasia"])) &
        (df["bx_result"] != "positive") &
        (df["ct_distant_progression"] == 0)
    ).astype(int)
    return pred


def report_hard_stop_stats(df: pd.DataFrame) -> dict:
    """Report how many patients each rule decides and agreement with target."""
    stats = {}
    for col, op, val, reason in HARD_STOP_RULES:
        if op == "==":
            mask = df[col] == val
        else:
            mask = pd.Series([False] * len(df), index=df.index)

        n = mask.sum()
        if n > 0 and TARGET_COL in df.columns:
            agree = (df.loc[mask, TARGET_COL] == 0).sum()
            disagree = n - agree
        else:
            agree = disagree = 0

        stats[f"{col}=={val}"] = {
            "n_patients": int(n),
            "correct_not_safe": int(agree),
            "false_not_safe": int(disagree),
            "agreement_pct": round(agree / n * 100, 1) if n > 0 else 0,
            "reason": reason,
        }

    # Combined
    df_stops = apply_hard_stops(df)
    n_stopped = df_stops["hard_stop_fired"].sum()
    if TARGET_COL in df.columns:
        correct = (df.loc[df_stops["hard_stop_fired"], TARGET_COL] == 0).sum()
    else:
        correct = 0

    stats["_combined"] = {
        "n_stopped": int(n_stopped),
        "n_remaining": int(len(df) - n_stopped),
        "correct_not_safe": int(correct),
        "false_not_safe": int(n_stopped - correct),
    }
    return stats
