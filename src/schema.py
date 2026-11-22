"""
Schema definition: column types, allowed values, ranges.
Used by Layer 1 (validation) and by the Streamlit app.
"""
import pandas as pd
from pathlib import Path
from src.config import (
    DICT_FILE, CATEGORICAL_COLS, BX_MUCINOUS_COL, ALL_FEATURE_COLS,
)

# ── Parse data dictionary ────────────────────────────────────────────────
def load_data_dictionary(path: Path = DICT_FILE) -> pd.DataFrame:
    """Return the data dictionary as a DataFrame."""
    return pd.read_csv(path)


def build_schema(dict_df: pd.DataFrame | None = None) -> dict:
    """
    Build a schema dict: column -> {dtype, values, range, n_missing}.
    """
    if dict_df is None:
        dict_df = load_data_dictionary()

    schema = {}
    for _, row in dict_df.iterrows():
        col = row["column"]
        vals_raw = str(row["values_or_range"])
        entry = {
            "n_missing": int(row["n_missing"]),
            "is_categorical": col in CATEGORICAL_COLS or col == BX_MUCINOUS_COL,
        }
        # Determine if it's a range (contains '-' with numbers) or enum
        if col in CATEGORICAL_COLS or col == BX_MUCINOUS_COL:
            entry["allowed_values"] = [v.strip() for v in vals_raw.split(";")]
            entry["dtype"] = "category"
        elif "-" in vals_raw and ";" not in vals_raw and col != "patient_id":
            parts = vals_raw.split("-")
            # Handle negative numbers: e.g. "-9-100" => min=-9, max=100
            if vals_raw.startswith("-"):
                # e.g. "-9-100"
                entry["min"] = float("-" + parts[1])
                entry["max"] = float(parts[2])
            else:
                entry["min"] = float(parts[0])
                entry["max"] = float(parts[1])
            entry["dtype"] = "float"
        elif ";" in vals_raw:
            vals = [v.strip() for v in vals_raw.split(";")]
            try:
                entry["allowed_values"] = [float(v) for v in vals]
                entry["dtype"] = "float"
            except ValueError:
                entry["allowed_values"] = vals
                entry["dtype"] = "category"
        else:
            entry["dtype"] = "str"

        schema[col] = entry
    return schema


def validate_patient(patient: dict, feature_cols: list[str] | None = None,
                     allow_missing: bool = False,
                     schema: dict | None = None) -> list[str]:
    """
    Validate only the feature fields of one patient.

    Parameters
    ----------
    patient : dict
        Values keyed by column name.  Extra keys (targets, patient_id) are
        ignored.
    feature_cols : list[str] | None
        Columns to validate.  Defaults to all model features.
    allow_missing : bool
        If True, missing values are accepted for every feature (used for batch
        uploads and for the what-if panel, where the missing-exam policy takes
        over).  If False, missing values are accepted only for columns that
        have documented missingness in the provided data.
    schema : dict | None
        Pre-built schema; loaded from the data dictionary if omitted.

    Returns
    -------
    list[str]
        Human-readable error messages.  Empty list means valid.
    """
    if schema is None:
        schema = build_schema()
    if feature_cols is None:
        feature_cols = list(ALL_FEATURE_COLS)

    errors: list[str] = []
    for col in feature_cols:
        spec = schema.get(col)
        if spec is None:
            continue
        val = patient.get(col)

        if pd.isna(val):
            known_missing = spec.get("n_missing", 0) > 0
            if not (allow_missing or known_missing):
                errors.append(f"{col}: missing value not expected")
            continue

        if "allowed_values" in spec:
            allowed = spec["allowed_values"]
            if val not in allowed and str(val) not in [str(v) for v in allowed]:
                errors.append(f"{col}: value '{val}' not in {allowed}")

        if "min" in spec:
            try:
                v = float(val)
                if v < spec["min"] - 0.01 or v > spec["max"] + 0.01:
                    errors.append(
                        f"{col}: value {v} out of range "
                        f"[{spec['min']}, {spec['max']}]"
                    )
            except (ValueError, TypeError):
                errors.append(f"{col}: cannot convert '{val}' to number")

    return errors


def validate_row(row: dict, schema: dict) -> list[str]:
    """
    Validate a single patient row against the schema.
    Returns list of error messages (empty = valid).
    """
    errors = []
    for col, spec in schema.items():
        if col == "patient_id":
            continue
        val = row.get(col)
        # Allow NaN for columns with known missingness
        if pd.isna(val):
            if spec["n_missing"] == 0 and col not in [
                "residual_tumor", "local_residual_tumor",
                "response_category", "safe_for_watch_and_wait",
            ]:
                errors.append(f"{col}: missing value not expected")
            continue

        if "allowed_values" in spec:
            if val not in spec["allowed_values"] and str(val) not in [str(v) for v in spec["allowed_values"]]:
                errors.append(
                    f"{col}: value '{val}' not in {spec['allowed_values']}"
                )
        if "min" in spec:
            try:
                v = float(val)
                if v < spec["min"] - 0.01 or v > spec["max"] + 0.01:
                    errors.append(
                        f"{col}: value {v} out of range [{spec['min']}, {spec['max']}]"
                    )
            except (ValueError, TypeError):
                errors.append(f"{col}: cannot convert '{val}' to number")
    return errors
