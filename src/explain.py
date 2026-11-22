"""Exact local contribution explanations for the logistic final model."""

import numpy as np
import pandas as pd

from src.config import EXAM_GROUPS
from src.features import prepare_dataframe


def _original_feature_name(transformed_name):
    name = transformed_name
    for prefix in ("num__", "cat__", "miss__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    if name.endswith("_missing") or "_missing_" in name:
        base = name.split("_missing", 1)[0]
        if base:
            return base + "_missing"

    matches = [col for col in (
        sum(EXAM_GROUPS.values(), [])
    ) if name == col or name.startswith(col + "_")]
    if matches:
        return max(matches, key=len)
    return name


def _exam_for_feature(feature_name):
    original = _original_feature_name(feature_name)
    base = original.replace("_missing", "")
    for exam, columns in EXAM_GROUPS.items():
        if base == columns or base in columns:
            return exam
        if any(base == col or base.startswith(col + "_") for col in columns):
            return exam
    return "other"


def explain_patient(patient, pipe, config):
    """Return intercept, logit, exam totals, and top feature contributions."""
    feature_cols = config["ml_feature_cols"]
    X = prepare_dataframe(pd.DataFrame([patient]), feature_cols)
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)

    names = []
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder":
            continue
        if transformer_name == "num":
            names.extend([f"num__{column}" for column in columns])
        elif transformer_name == "cat":
            if hasattr(transformer, "get_feature_names_out"):
                names.extend([
                    f"cat__{name}"
                    for name in transformer.get_feature_names_out(columns).tolist()
                ])
            else:
                names.extend([f"cat__{column}" for column in columns])
        elif transformer_name == "miss":
            names.extend([f"miss__{column}" for column in columns])
    coefficients = np.asarray(model.coef_[0], dtype=float)
    intercept = float(model.intercept_[0])
    values = transformed[0]
    contributions = values * coefficients
    logit = intercept + float(contributions.sum())

    exam_totals = {exam: 0.0 for exam in EXAM_GROUPS}
    exam_totals["other"] = 0.0
    top = []
    for name, value, coefficient, contribution in zip(
        names, values, coefficients, contributions
    ):
        exam = _exam_for_feature(name)
        exam_totals[exam] = exam_totals.get(exam, 0.0) + float(contribution)
        top.append({
            "feature": _original_feature_name(name),
            "transformed_feature": name,
            "value": float(value),
            "coefficient": float(coefficient),
            "contribution": float(contribution),
            "exam": exam,
        })

    top.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    for item in top:
        item["contribution"] = float(item["contribution"])
        item["coefficient"] = float(item["coefficient"])
        item["value"] = float(item["value"])

    return {
        "intercept": intercept,
        "logit": logit,
        "exam_contributions": {
            key: float(value) for key, value in exam_totals.items()
            if abs(value) > 1e-12
        },
        "top_contributors": top[:12],
    }


def explain_patient_dataframe(patient, pipe, config):
    """Return one-row explanations as a DataFrame for plotting/export."""
    explanation = explain_patient(patient, pipe, config)
    rows = [
        {"exam": exam, "contribution": contribution}
        for exam, contribution in explanation["exam_contributions"].items()
    ]
    rows.append({
        "exam": "intercept",
        "contribution": explanation["intercept"],
    })
    return pd.DataFrame(rows)
