# Post-chemoradiotherapy low-rectal-cancer dataset (800 rows)

One row per patient. Columns are grouped by examination, identified by prefix:

| Prefix | Examination |
|---|---|
| `age`, `sex`, `tumor_distance_anal_verge_cm`, `pre_*` | Baseline clinical and pre-treatment MRI |
| `regimen`, `weeks_since_end_treatment` | Treatment |
| `dre_` | Digital rectal examination |
| `endo_` | Endoscopy |
| `bx_` | Biopsy |
| `mri_` | Post-treatment pelvic MRI |
| `ct_` | Thoraco-abdomino-pelvic CT |
| `post_CEA_ng_ml` | Post-treatment CEA |

## Targets (never use as model inputs)
- `residual_tumor`, `local_residual_tumor` (1 = tumor remains locally)
- `response_category` (complete / near_complete / incomplete)
- `safe_for_watch_and_wait` (1 = complete local response AND no distant progression on CT)

## Notes
- Missing values (NaN) occur in `pre_mri_ADC_mean`, `mri_DWI_restriction`, `mri_ADC_mean`, `mri_wall_thickness_mm`, `post_CEA_ng_ml`.
- `bx_mucinous_component = -1` means "not applicable" (no biopsy performed), not a measurement.
- Biopsy: a negative result does not exclude residual tumor. DWI and CT wall thickening are noisy (scar and inflammation can mimic tumor).
