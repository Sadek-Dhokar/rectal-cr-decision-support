"""
Tests: schema validation, no-leakage, rules, abstention, determinism.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path


def test_no_target_in_features():
    """CRITICAL: no target-derived column may be among model inputs."""
    from src.config import TARGET_COLS, ALL_FEATURE_COLS
    from src.features import get_feature_columns

    feature_cols = get_feature_columns()
    target_set = set(TARGET_COLS)
    overlap = set(feature_cols) & target_set
    assert len(overlap) == 0, f"TARGET LEAKAGE: {overlap} in feature columns!"


def test_schema_loads():
    """Data dictionary loads without error."""
    from src.schema import build_schema
    schema = build_schema()
    assert len(schema) > 0
    assert "patient_id" in schema


def test_data_shape():
    """Dataset has expected shape."""
    from src.config import DATA_FILE
    df = pd.read_csv(DATA_FILE)
    assert df.shape == (800, 49), f"Expected (800, 49), got {df.shape}"


def test_target_relationships():
    """Verify target column relationships."""
    from src.config import DATA_FILE
    df = pd.read_csv(DATA_FILE)

    # residual_tumor == local_residual_tumor
    assert (df["residual_tumor"] == df["local_residual_tumor"]).all()

    # response_category == complete iff local_residual_tumor == 0
    assert (
        df["response_category"].eq("complete") == df["local_residual_tumor"].eq(0)
    ).all()

    # safe = (local == 0) AND (ct_distant == 0)
    safe_computed = (
        (df["local_residual_tumor"] == 0) & (df["ct_distant_progression"] == 0)
    ).astype(int)
    assert (safe_computed == df["safe_for_watch_and_wait"]).all()


def test_hard_stops_all_not_safe():
    """Hard-stop patients should overwhelmingly be not safe."""
    from src.config import DATA_FILE, TARGET_COL
    from src.rules import apply_hard_stops
    df = pd.read_csv(DATA_FILE)
    df_stops = apply_hard_stops(df)
    stopped = df_stops[df_stops["hard_stop_fired"]]
    # At least 90% should be target=0
    agreement = (stopped[TARGET_COL] == 0).mean()
    assert agreement >= 0.90, f"Hard stop agreement only {agreement:.1%}"


def test_hard_stops_cover_distant_progression():
    """ct_distant_progression==1 must trigger hard stop."""
    from src.config import DATA_FILE
    from src.rules import apply_hard_stops
    df = pd.read_csv(DATA_FILE)
    df_stops = apply_hard_stops(df)
    distant = df["ct_distant_progression"] == 1
    # All distant progression patients should be stopped
    assert df_stops.loc[distant, "hard_stop_fired"].all()


def test_hard_stops_cover_positive_biopsy():
    """bx_result==positive must trigger hard stop."""
    from src.config import DATA_FILE
    from src.rules import apply_hard_stops
    df = pd.read_csv(DATA_FILE)
    df_stops = apply_hard_stops(df)
    positive = df["bx_result"] == "positive"
    assert df_stops.loc[positive, "hard_stop_fired"].all()


def test_rule_based_protocol():
    """B1 rule-based protocol should produce valid binary predictions."""
    from src.config import DATA_FILE
    from src.rules import rule_based_protocol
    df = pd.read_csv(DATA_FILE)
    preds = rule_based_protocol(df)
    assert set(preds.unique()).issubset({0, 1})
    assert len(preds) == len(df)


def test_preprocessing_no_nan_output():
    """Preprocessor should not produce NaN in output."""
    from src.config import DATA_FILE
    from src.features import (
        get_feature_columns, build_preprocessor, prepare_dataframe
    )
    df = pd.read_csv(DATA_FILE)
    feature_cols = get_feature_columns()
    ml_cols = [c for c in feature_cols if c != "ct_distant_progression"]
    df_prep = prepare_dataframe(df, ml_cols)
    preprocessor = build_preprocessor(ml_cols)
    X_transformed = preprocessor.fit_transform(df_prep)
    assert not np.any(np.isnan(X_transformed)), "NaN in preprocessed output!"


def test_audit_log_no_raw_values():
    """Audit log should not contain raw clinical values."""
    from src.audit import hash_patient_id, log_decision
    import tempfile

    result = {
        "decision": "COMPLETE RESPONSE",
        "probability": 0.95,
        "confidence": "HIGH",
        "source": "model",
        "rule": None,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        log_path = Path(f.name)

    log_decision("PT_0001", result, log_path=log_path)

    with open(log_path) as f:
        entry = json.loads(f.readline())

    # Should have hashed ID, not raw
    assert "PT_0001" not in str(entry)
    assert "patient_hash" in entry
    assert len(entry["patient_hash"]) == 16

    # Clean up
    log_path.unlink()


def test_determinism():
    """Same input should produce same output."""
    from src.rules import apply_hard_stops_single

    patient = {
        "ct_distant_progression": 0,
        "bx_result": "negative",
        "endo_finding": "flat_white_scar",
        "dre_finding": "no_abnormality",
    }
    r1 = apply_hard_stops_single(patient)
    r2 = apply_hard_stops_single(patient)
    assert r1 == r2


def test_abstention_band():
    """A patient with an unavailable exam is referred to the MDT (UNCERTAIN)."""
    from src.predict import predict_single

    patient = {
        "patient_id": "TEST_001",
        "ct_distant_progression": 0,
        "bx_result": "negative",
        "endo_finding": "flat_white_scar",
        "dre_finding": "no_abnormality",
        # no MRI: the missing-exam policy must block a COMPLETE call
    }
    try:
        result = predict_single(patient, log=False)
        assert result["decision"] in ["COMPLETE RESPONSE", "NOT COMPLETE", "UNCERTAIN"]
        assert result["decision"] == "UNCERTAIN"
        assert result["source"] == "policy"
        assert "mri" in result["unavailable_exams"]
        if result["probability"] is not None:
            assert 0 <= result["probability"] <= 1
    except FileNotFoundError:
        pytest.skip("Model not yet trained")


def test_decide_from_probability():
    """Layer 3 mapping from probability to the three decisions."""
    from src.rules import (
        decide_from_probability, COMPLETE, NOT_COMPLETE, UNCERTAIN,
    )
    assert decide_from_probability(0.90, 0.29, 0.71) == COMPLETE
    assert decide_from_probability(0.10, 0.29, 0.71) == NOT_COMPLETE
    assert decide_from_probability(0.50, 0.29, 0.71) == UNCERTAIN


def test_missing_exam_policy():
    """Each required exam, when unavailable, blocks a COMPLETE call."""
    from src.rules import missing_exam_policy

    base = {
        "dre_finding": "no_abnormality",
        "endo_finding": "flat_white_scar",
        "bx_result": "negative",
        "mri_mrTRG": 1,
        "ct_distant_progression": 0,
    }
    assert not missing_exam_policy(base)["blocked"]

    key_cols = {
        "dre": "dre_finding",
        "endoscopy": "endo_finding",
        "biopsy": "bx_result",
        "mri": "mri_mrTRG",
        "ct": "ct_distant_progression",
    }
    for exam, col in key_cols.items():
        patient = dict(base)
        patient[col] = None
        result = missing_exam_policy(patient)
        assert result["blocked"], f"{exam} should be blocked"
        assert exam in result["unavailable_exams"]


def test_feature_matrix_has_no_targets():
    """The actual model input matrix must not contain any target column."""
    from src.config import DATA_FILE, TARGET_COLS, ID_COL
    from src.features import (
        get_ml_feature_columns, prepare_dataframe, build_preprocessor,
    )

    df = pd.read_csv(DATA_FILE).head(50)
    feat = get_ml_feature_columns()
    assert "ct_distant_progression" not in feat

    X = prepare_dataframe(df, feat)
    assert ID_COL not in X.columns
    assert not (set(X.columns) & set(TARGET_COLS))

    pre = build_preprocessor(feat).fit(X)
    input_cols = set(pre.feature_names_in_)
    assert ID_COL not in input_cols
    assert not (input_cols & set(TARGET_COLS))


def test_prepare_dataframe_drops_targets():
    """prepare_dataframe must not leak target or ID columns into X."""
    from src.config import DATA_FILE, TARGET_COLS, ID_COL
    from src.features import get_ml_feature_columns, prepare_dataframe

    df = pd.read_csv(DATA_FILE).head(5)
    X = prepare_dataframe(df, get_ml_feature_columns())
    assert ID_COL not in X.columns
    assert not (set(X.columns) & set(TARGET_COLS))


def test_schema_validation_rejects_bad_values():
    """Out-of-range and malformed values must produce clear errors."""
    from src.schema import validate_patient
    from src.features import get_ml_feature_columns

    feat = get_ml_feature_columns()

    errors = validate_patient({"age": 200}, feat, allow_missing=True)
    assert any("age" in e and "out of range" in e for e in errors)

    errors = validate_patient({"sex": "X"}, feat, allow_missing=True)
    assert any("sex" in e for e in errors)

    errors = validate_patient({"age": "not-a-number"}, feat, allow_missing=True)
    assert any("age" in e for e in errors)

    errors = validate_patient({"age": 60, "sex": "M"}, feat, allow_missing=True)
    assert errors == []


def test_predict_single_hard_stop():
    """A hard stop must return NOT COMPLETE with source=rule."""
    from src.predict import predict_single

    patient = {
        "patient_id": "TEST_HS",
        "ct_distant_progression": 1,
        "bx_result": "negative",
        "endo_finding": "flat_white_scar",
        "dre_finding": "no_abnormality",
    }
    try:
        result = predict_single(patient, log=False)
    except FileNotFoundError:
        pytest.skip("Model not yet trained")

    assert result["decision"] == "NOT COMPLETE"
    assert result["source"] == "rule"
    assert result["probability"] is None


def test_predict_batch_missing_exam_policy():
    """A batch with MRI unavailable must be referred to the MDT."""
    from src.predict import predict_batch
    from src.config import DATA_FILE

    df = pd.read_csv(DATA_FILE)
    eligible = df[
        (df["ct_distant_progression"] == 0)
        & (df["bx_result"] != "positive")
        & (df["endo_finding"] != "visible_tumor")
        & (df["dre_finding"] != "palpable_mass")
    ].head(3).copy()
    eligible["mri_mrTRG"] = np.nan

    try:
        out = predict_batch(eligible, log=False)
    except FileNotFoundError:
        pytest.skip("Model not yet trained")

    assert set(out["decision"]) == {"UNCERTAIN"}
    assert out["source"].eq("policy").all()
    assert out["unavailable_exams"].apply(lambda x: "mri" in x).all()


def test_audit_log_has_no_clinical_values(tmp_path):
    """Audit entries must not contain raw identifiers or clinical values."""
    from src.audit import log_decision

    result = {
        "decision": "COMPLETE RESPONSE",
        "probability": 0.91,
        "confidence": "HIGH",
        "source": "model",
        "rule": None,
    }
    log_path = tmp_path / "audit.jsonl"
    log_decision("PT_0001", result, log_path=log_path)
    text = log_path.read_text(encoding="utf-8")
    assert "PT_0001" not in text
    assert "palpable_mass" not in text
    assert "mri_mrTRG" not in text


def test_threshold_policy_config():
    """The PPV/NPV target and CI rule must be explicit in config."""
    from src.config import (
        PPV_TARGET, NPV_TARGET, CI_LEVEL, N_BOOTSTRAP,
        THRESHOLD_SELECTION,
    )
    assert 0 < PPV_TARGET <= 1
    assert 0 < NPV_TARGET <= 1
    assert 0 < CI_LEVEL < 1
    assert N_BOOTSTRAP >= 1000
    assert THRESHOLD_SELECTION == "oof_bootstrap_lower_bound"


def test_train_config_thresholds():
    """Saved thresholds must be ordered and selected by the OOF rule."""
    from src.config import REPORTS_DIR
    path = REPORTS_DIR / "train_config.json"
    if not path.exists():
        pytest.skip("Model not yet trained")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert 0 < cfg["t_low"] < cfg["t_high"] < 1
    assert cfg["calibration"] in {"none", "sigmoid", "isotonic"}
    assert cfg["threshold_selection"]["method"].startswith("OOF")


def test_bx_mucinous_treatment():
    """bx_mucinous_component=-1 should be treated as a category, not a number."""
    from src.features import prepare_bx_mucinous
    from src.config import BX_MUCINOUS_COL
    df = pd.DataFrame({BX_MUCINOUS_COL: [-1, 0, 1]})
    df_out = prepare_bx_mucinous(df)
    assert df_out[BX_MUCINOUS_COL].tolist() == ["not_applicable", "no", "yes"]


def test_exam_groups_cover_all_features():
    """Every feature should belong to exactly one exam group."""
    from src.config import ALL_FEATURE_COLS, EXAM_GROUPS
    grouped = []
    for cols in EXAM_GROUPS.values():
        grouped.extend(cols)
    assert set(grouped) == set(ALL_FEATURE_COLS), \
        f"Mismatch: {set(ALL_FEATURE_COLS) - set(grouped)} not in groups"


def test_exact_contributions_reproduce_logit():
    """Five exact logistic contributions plus intercept must reproduce decision logits."""
    from src.config import DATA_FILE
    from src.explain import explain_patient
    from src.predict import load_model_and_config
    from src.features import prepare_dataframe

    df = pd.read_csv(DATA_FILE)
    eligible = df[
        (df["ct_distant_progression"] == 0)
        & (df["bx_result"] != "positive")
        & (df["endo_finding"] != "visible_tumor")
        & (df["dre_finding"] != "palpable_mass")
    ].head(5)

    pipe, config = load_model_and_config()
    for _, row in eligible.iterrows():
        explanation = explain_patient(row.to_dict(), pipe, config)
        X = prepare_dataframe(pd.DataFrame([row.to_dict()]), config["ml_feature_cols"])
        model_logit = float(pipe.decision_function(X)[0])
        assert abs(explanation["logit"] - model_logit) < 1e-6
