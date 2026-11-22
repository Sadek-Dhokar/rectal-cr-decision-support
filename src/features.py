"""
Feature engineering: preprocessing pipeline with missing-value indicators,
encoding, scaling. Operates inside CV folds to prevent leakage.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from src.config import (
    ALL_FEATURE_COLS, CATEGORICAL_COLS, MISSING_COLS,
    TARGET_COLS, ID_COL, BX_MUCINOUS_COL, EXAM_GROUPS,
)


def get_feature_columns(include_ct_distant=True):
    """Return the list of feature columns (excludes targets and patient_id)."""
    cols = [c for c in ALL_FEATURE_COLS if c not in TARGET_COLS and c != ID_COL]
    if not include_ct_distant:
        cols = [c for c in cols if c != "ct_distant_progression"]
    return cols


def get_ml_feature_columns():
    """Features available to Layer 2.

    ct_distant_progression is deliberately excluded: distant progression is
    handled by the Layer 1 hard stop, so the ML model must not rely on it.
    """
    return get_feature_columns(include_ct_distant=False)


def get_numeric_cols(feature_cols=None):
    if feature_cols is None:
        feature_cols = get_feature_columns()
    return [c for c in feature_cols
            if c not in CATEGORICAL_COLS and c != BX_MUCINOUS_COL]


def get_categorical_cols(feature_cols=None):
    if feature_cols is None:
        feature_cols = get_feature_columns()
    return [c for c in feature_cols
            if c in CATEGORICAL_COLS or c == BX_MUCINOUS_COL]


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary columns indicating missingness for columns with known NaN."""
    df = df.copy()
    for col in MISSING_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)
    return df


def prepare_bx_mucinous(df: pd.DataFrame) -> pd.DataFrame:
    """Convert bx_mucinous_component: -1 → 'not_applicable', 0 → '0', 1 → '1'."""
    df = df.copy()
    if BX_MUCINOUS_COL in df.columns:
        df[BX_MUCINOUS_COL] = df[BX_MUCINOUS_COL].map(
            {-1: "not_applicable", 0: "no", 1: "yes"}
        ).fillna("not_applicable")
    return df


def build_preprocessor(feature_cols=None):
    """
    Build a sklearn ColumnTransformer for the given feature columns.

    Unknown categorical values at inference are encoded as all zeros
    (handle_unknown="ignore").  This is important for the app: a whole exam
    can be marked unavailable without crashing the encoder.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    num_cols = get_numeric_cols(feature_cols)
    cat_cols = get_categorical_cols(feature_cols)

    # Missing-indicator columns (added before this pipeline)
    miss_indicator_cols = [f"{c}_missing" for c in MISSING_COLS if c in feature_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(
            drop="if_binary", sparse_output=False, handle_unknown="ignore",
            min_frequency=5,
        )),
    ])

    # Missing indicators are already binary, just pass through
    indicator_pipeline = Pipeline([
        ("passthrough", FunctionTransformer(feature_names_out="one-to-one")),
    ])

    transformers = [
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ]
    if miss_indicator_cols:
        transformers.append(("miss", indicator_pipeline, miss_indicator_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def ensure_feature_columns(df: pd.DataFrame, feature_cols=None) -> pd.DataFrame:
    """Return a copy of df with every required feature column present.

    Missing columns are added as NaN.  This lets a single patient (or a batch
    with a whole exam missing) flow through the same preprocessor without a
    column-mismatch crash; the missing-exam policy then decides whether a
    definite COMPLETE call is still allowed.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def prepare_dataframe(df: pd.DataFrame, feature_cols=None,
                      drop_targets: bool = True) -> pd.DataFrame:
    """
    Prepare a DataFrame for the pipeline:
    1. Ensure all required feature columns exist
    2. Add missing indicators
    3. Convert bx_mucinous_component
    4. Keep only model inputs (targets and patient_id are dropped by default)
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    df = ensure_feature_columns(df, feature_cols)
    df = add_missing_indicators(df)
    df = prepare_bx_mucinous(df)

    miss_cols = [f"{c}_missing" for c in MISSING_COLS if c in feature_cols]
    keep_cols = feature_cols + miss_cols
    if not drop_targets:
        for extra in TARGET_COLS + [ID_COL]:
            if extra in df.columns and extra not in keep_cols:
                keep_cols.append(extra)

    return df[[c for c in keep_cols if c in df.columns]]


def get_exam_group_for_feature(feature_name: str) -> str:
    """Map a (possibly transformed) feature name back to its exam group."""
    # Strip prefixes added by ColumnTransformer
    for prefix in ["num__", "cat__", "miss__"]:
        if feature_name.startswith(prefix):
            feature_name = feature_name[len(prefix):]
            break

    # Strip OHE suffixes
    base = feature_name.split("_missing")[0] if "_missing" in feature_name else feature_name

    for group, cols in EXAM_GROUPS.items():
        for col in cols:
            if base.startswith(col):
                return group
    return "other"


def get_ablation_specs():
    """
    Ordered feature sets for the exam ablation study.

    MRI → +endoscopy → +DRE → +biopsy → +CT (all five) → all five plus
    baseline/treatment/post-treatment labs.

    ct_distant_progression is excluded from the CT group because the Layer 1
    hard stop removes every patient with distant progression; within the ML
    population that column is constant and would add no information.
    """
    mri = EXAM_GROUPS["mri"].copy()
    endo = EXAM_GROUPS["endoscopy"].copy()
    dre = EXAM_GROUPS["dre"].copy()
    bx = EXAM_GROUPS["biopsy"].copy()
    ct = [c for c in EXAM_GROUPS["ct"] if c != "ct_distant_progression"]
    baseline = (EXAM_GROUPS["baseline"] + EXAM_GROUPS["treatment"]
                + EXAM_GROUPS["post_labs"])

    specs = {
        "MRI only": {
            "features": mri,
            "n_exams": 1,
            "adds_baseline": False,
        },
        "MRI + Endoscopy": {
            "features": mri + endo,
            "n_exams": 2,
            "adds_baseline": False,
        },
        "MRI + Endo + DRE": {
            "features": mri + endo + dre,
            "n_exams": 3,
            "adds_baseline": False,
        },
        "MRI + Endo + DRE + Biopsy": {
            "features": mri + endo + dre + bx,
            "n_exams": 4,
            "adds_baseline": False,
        },
        "All 5 exams": {
            "features": mri + endo + dre + bx + ct,
            "n_exams": 5,
            "adds_baseline": False,
        },
        "All 5 exams + baseline": {
            "features": mri + endo + dre + bx + ct + baseline,
            "n_exams": 5,
            "adds_baseline": True,
        },
    }
    return specs


def get_ablation_feature_sets():
    """Backwards-compatible name -> feature list mapping."""
    return {name: spec["features"] for name, spec in get_ablation_specs().items()}
