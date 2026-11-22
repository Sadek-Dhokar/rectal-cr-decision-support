"""
Central configuration for rectal-cr pipeline.
All tuneable values in one place.

Design notes
------------
* Layer 1 (rules) uses deterministic safety overrides.  No model is involved.
* Layer 2 (ML) is fitted only on patients who survive Layer 1.
* Layer 3 (abstention) uses two thresholds selected on out-of-fold (OOF)
  predictions from the training split, never on the test split.
* The test split is held out until the single final evaluation run.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_DIR = PROJECT_ROOT / "app"


def _find_data_file(filename: str) -> Path:
    """Locate a provided data file without hard-coding one checkout layout."""
    candidates = [
        PROJECT_ROOT.parent / "data" / filename,
        PROJECT_ROOT / "data" / filename,
        PROJECT_ROOT.parent / "files" / filename,
        PROJECT_ROOT / "files" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_FILE = _find_data_file("rectal_post_crt_5exams_800.csv")
DICT_FILE = _find_data_file("data_dictionary_5exams.csv")
DATA_DIR = DATA_FILE.parent

# ── Reproducibility ─────────────────────────────────────────────────────
SEED = 42

# ── Split ratios ────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Target ───────────────────────────────────────────────────────────────
TARGET_COL = "safe_for_watch_and_wait"
TARGET_COLS = [
    "residual_tumor",
    "local_residual_tumor",
    "response_category",
    "safe_for_watch_and_wait",
]
ID_COL = "patient_id"

# ── Model identity ───────────────────────────────────────────────────────
MODEL_VERSION = "v1.0"
MODEL_NAME = "logistic_l2"

# ── Abstention thresholds and selection rule ────────────────────────────
# A COMPLETE call must be supported by a conservative estimate of precision:
# the lower bound of the 95% bootstrap CI of PPV must meet PPV_TARGET.
# A NOT COMPLETE call must be supported by a conservative estimate of NPV:
# the lower bound of the 95% bootstrap CI of NPV must meet NPV_TARGET.
PPV_TARGET = 0.95
NPV_TARGET = 0.90
CI_LEVEL = 0.95
N_BOOTSTRAP = 2000
MIN_COMPLETE_CALLS = 20
MIN_NOT_COMPLETE_CALLS = 20
THRESHOLD_GRID_STEP = 0.01
THRESHOLD_SELECTION = "oof_bootstrap_lower_bound"
DEFAULT_T_LOW = 0.25       # fallback only; real values come from OOF selection
DEFAULT_T_HIGH = 0.75      # fallback only; real values come from OOF selection

# ── Cross-validation ────────────────────────────────────────────────────
CV_N_REPEATS = 5
CV_N_SPLITS = 5
CV_N_JOBS = 1              # deterministic and reliable on Windows

# ── Audit log ───────────────────────────────────────────────────────────
# Demo salt.  Production must use a secret salt stored outside the repository.
AUDIT_SALT = "tech4health2026-rectal-cr"

# ── Exam groups (for explanation & ablation) ─────────────────────────────
EXAM_GROUPS = {
    "baseline": [
        "age", "sex", "tumor_distance_anal_verge_cm",
        "pre_T_stage", "pre_N_stage", "pre_EMVI", "pre_CRM_threatened",
        "pre_mucinous", "pre_circumferential_extent",
        "pre_mri_tumor_volume_ml", "pre_mri_ADC_mean", "pre_CEA_ng_ml",
    ],
    "treatment": [
        "regimen", "weeks_since_end_treatment",
    ],
    "dre": [
        "dre_finding", "dre_fixed_to_structures",
    ],
    "endoscopy": [
        "endo_finding", "endo_mucosal_irregularity",
        "endo_stenosis", "endo_pct_wall_involved",
    ],
    "biopsy": [
        "bx_done", "bx_n_samples", "bx_result", "bx_mucinous_component",
    ],
    "mri": [
        "mri_mrTRG", "mri_T2_residual_signal", "mri_split_scar",
        "mri_DWI_restriction", "mri_ADC_mean", "mri_tumor_volume_ml",
        "mri_volume_reduction_pct", "mri_wall_thickness_mm",
        "mri_lumen_irregularity", "mri_mucin_pool",
        "mri_suspicious_nodes", "mri_EMVI_post", "mri_MRF_involved",
    ],
    "ct": [
        "ct_lung_lesion", "ct_liver_lesion", "ct_distant_nodes",
        "ct_distant_progression", "ct_rectal_wall_thickening",
        "ct_perirectal_stranding",
    ],
    "post_labs": [
        "post_CEA_ng_ml",
    ],
}

# Flat list of all feature columns (excludes id and targets)
ALL_FEATURE_COLS = []
for _cols in EXAM_GROUPS.values():
    ALL_FEATURE_COLS.extend(_cols)

# Categorical columns
CATEGORICAL_COLS = [
    "sex", "pre_T_stage", "pre_N_stage", "pre_circumferential_extent",
    "regimen", "dre_finding", "endo_finding", "bx_result",
    "mri_T2_residual_signal",
]

# bx_mucinous_component: -1 is "not applicable", treat as category
BX_MUCINOUS_COL = "bx_mucinous_component"

# Columns with known missingness
MISSING_COLS = [
    "pre_mri_ADC_mean", "mri_DWI_restriction", "mri_ADC_mean",
    "mri_wall_thickness_mm", "post_CEA_ng_ml",
]

# ── Hard-stop rules (Layer 1) ───────────────────────────────────────────
# Each rule: (column, operator, value, reason)
HARD_STOP_RULES = [
    ("ct_distant_progression", "==", 1,
     "Distant metastatic progression on CT: watch-and-wait contraindicated"),
    ("bx_result", "==", "positive",
     "Positive biopsy: residual viable tumor confirmed histologically"),
    ("endo_finding", "==", "visible_tumor",
     "Visible tumor on endoscopy: macroscopic residual disease"),
    ("dre_finding", "==", "palpable_mass",
     "Palpable mass on DRE: residual bulky disease"),
]

# ── Missing-exam policy (Layer 3 safety rule) ───────────────────────────
# A COMPLETE call requires that each of the five exams has delivered the key
# finding that the safety rules rely on.  If an exam is unavailable, the tool
# cannot clear its safety check, so the case is referred to the MDT (UNCERTAIN)
# unless a hard stop from another exam already forces NOT COMPLETE.
REQUIRED_EXAMS = ["dre", "endoscopy", "biopsy", "mri", "ct"]

EXAM_AVAILABILITY_COLUMNS = {
    "dre": ["dre_finding"],
    "endoscopy": ["endo_finding"],
    "biopsy": ["bx_result"],
    "mri": ["mri_mrTRG"],
    "ct": ["ct_distant_progression"],
}

EXAM_DISPLAY_NAMES = {
    "dre": "DRE",
    "endoscopy": "Endoscopy",
    "biopsy": "Biopsy",
    "mri": "Pelvic MRI",
    "ct": "Thoraco-abdomino-pelvic CT",
}

# Columns set to missing when an exam is marked unavailable in the what-if panel.
EXAM_UNAVAILABLE_COLUMNS = {
    "dre": EXAM_GROUPS["dre"],
    "endoscopy": EXAM_GROUPS["endoscopy"],
    "biopsy": EXAM_GROUPS["biopsy"],
    "mri": EXAM_GROUPS["mri"],
    "ct": EXAM_GROUPS["ct"],
}
