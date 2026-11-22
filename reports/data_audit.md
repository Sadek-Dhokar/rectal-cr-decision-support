# Phase 1: Data Audit Report

## 1. Shape and columns
- Rows: 800
- Columns: 49
- Column names: ['patient_id', 'age', 'sex', 'tumor_distance_anal_verge_cm', 'pre_T_stage', 'pre_N_stage', 'pre_EMVI', 'pre_CRM_threatened', 'pre_mucinous', 'pre_circumferential_extent', 'pre_mri_tumor_volume_ml', 'pre_mri_ADC_mean', 'pre_CEA_ng_ml', 'regimen', 'weeks_since_end_treatment', 'dre_finding', 'dre_fixed_to_structures', 'endo_finding', 'endo_mucosal_irregularity', 'endo_stenosis', 'endo_pct_wall_involved', 'bx_done', 'bx_n_samples', 'bx_result', 'bx_mucinous_component', 'mri_mrTRG', 'mri_T2_residual_signal', 'mri_split_scar', 'mri_DWI_restriction', 'mri_ADC_mean', 'mri_tumor_volume_ml', 'mri_volume_reduction_pct', 'mri_wall_thickness_mm', 'mri_lumen_irregularity', 'mri_mucin_pool', 'mri_suspicious_nodes', 'mri_EMVI_post', 'mri_MRF_involved', 'ct_lung_lesion', 'ct_liver_lesion', 'ct_distant_nodes', 'ct_distant_progression', 'ct_rectal_wall_thickening', 'ct_perirectal_stranding', 'post_CEA_ng_ml', 'residual_tumor', 'response_category', 'local_residual_tumor', 'safe_for_watch_and_wait']

## 2. Dtypes
- `patient_id`: object | unique=800 | sample=PT_0001
- `age`: int64 | unique=61 | sample=60
- `sex`: object | unique=2 | sample=M
- `tumor_distance_anal_verge_cm`: float64 | unique=53 | sample=4.5
- `pre_T_stage`: object | unique=3 | sample=T3
- `pre_N_stage`: object | unique=3 | sample=N0
- `pre_EMVI`: int64 | unique=2 | sample=0
- `pre_CRM_threatened`: int64 | unique=2 | sample=0
- `pre_mucinous`: int64 | unique=2 | sample=0
- `pre_circumferential_extent`: object | unique=3 | sample=50-75%
- `pre_mri_tumor_volume_ml`: float64 | unique=377 | sample=37.3
- `pre_mri_ADC_mean`: float64 | unique=76 | sample=0.62
- `pre_CEA_ng_ml`: float64 | unique=204 | sample=8.1
- `regimen`: object | unique=3 | sample=CRT_plus_consolidation
- `weeks_since_end_treatment`: int64 | unique=12 | sample=12
- `dre_finding`: object | unique=4 | sample=irregularity_or_nodule
- `dre_fixed_to_structures`: int64 | unique=2 | sample=0
- `endo_finding`: object | unique=5 | sample=residual_nodule
- `endo_mucosal_irregularity`: int64 | unique=2 | sample=1
- `endo_stenosis`: int64 | unique=2 | sample=0
- `endo_pct_wall_involved`: int64 | unique=61 | sample=3
- `bx_done`: int64 | unique=2 | sample=1
- `bx_n_samples`: int64 | unique=6 | sample=2
- `bx_result`: object | unique=4 | sample=negative
- `bx_mucinous_component`: int64 | unique=3 | sample=0
- `mri_mrTRG`: int64 | unique=5 | sample=3
- `mri_T2_residual_signal`: object | unique=4 | sample=intermediate_signal
- `mri_split_scar`: int64 | unique=2 | sample=1
- `mri_DWI_restriction`: float64 | unique=2 | sample=1.0
- `mri_ADC_mean`: float64 | unique=147 | sample=1.08
- `mri_tumor_volume_ml`: float64 | unique=281 | sample=21.3
- `mri_volume_reduction_pct`: int64 | unique=101 | sample=43
- `mri_wall_thickness_mm`: float64 | unique=129 | sample=5.7
- `mri_lumen_irregularity`: int64 | unique=2 | sample=0
- `mri_mucin_pool`: int64 | unique=2 | sample=0
- `mri_suspicious_nodes`: int64 | unique=2 | sample=1
- `mri_EMVI_post`: int64 | unique=2 | sample=0
- `mri_MRF_involved`: int64 | unique=2 | sample=0
- `ct_lung_lesion`: int64 | unique=2 | sample=0
- `ct_liver_lesion`: int64 | unique=2 | sample=0
- `ct_distant_nodes`: int64 | unique=2 | sample=0
- `ct_distant_progression`: int64 | unique=2 | sample=0
- `ct_rectal_wall_thickening`: int64 | unique=2 | sample=0
- `ct_perirectal_stranding`: int64 | unique=2 | sample=0
- `post_CEA_ng_ml`: float64 | unique=101 | sample=4.0
- `residual_tumor`: int64 | unique=2 | sample=1
- `response_category`: object | unique=3 | sample=near_complete
- `local_residual_tumor`: int64 | unique=2 | sample=1
- `safe_for_watch_and_wait`: int64 | unique=2 | sample=0

## 3. Missing values
- `pre_mri_ADC_mean`: 71 (8.9%)
- `mri_DWI_restriction`: 41 (5.1%)
- `mri_ADC_mean`: 71 (8.9%)
- `mri_wall_thickness_mm`: 31 (3.9%)
- `post_CEA_ng_ml`: 99 (12.4%)

## 4. Class balance (target: safe_for_watch_and_wait)
- 0: 430 (53.8%)
- 1: 370 (46.2%)

### Secondary targets
- `residual_tumor`: {0: 408, 1: 392}
- `local_residual_tumor`: {0: 408, 1: 392}
- `response_category`: {'complete': 408, 'incomplete': 224, 'near_complete': 168}
- `safe_for_watch_and_wait`: {0: 430, 1: 370}

## 5. Target relationship verification
- residual_tumor == local_residual_tumor for all rows: **True**
- response_category=='complete' iff local_residual_tumor==0: **True**
- safe = (local_residual_tumor==0) AND (ct_distant_progression==0): **True**
- response_category distribution: {'complete': 408, 'incomplete': 224, 'near_complete': 168}
- Patients with complete local response but distant progression: 38

## 6. Leakage check
- Feature columns that are also targets: NONE (safe)
- ct_distant_progression is a feature: True
- ct_distant_progression is part of target definition: Yes (safe = local==0 AND distant==0)
  → This is a **legitimate clinical input** (CT finding), not leakage.
  → The hard-stop rule catches distant_progression==1 in Layer 1,
  → so the ML model never sees a case where distant_progression==1.

## 7. Feature ranges (numeric)
- `age`: min=25, max=88, mean=58.73, std=11.30
- `tumor_distance_anal_verge_cm`: min=0.5, max=6.0, mean=3.95, std=1.21
- `pre_EMVI`: min=0, max=1, mean=0.32, std=0.47
- `pre_CRM_threatened`: min=0, max=1, mean=0.31, std=0.46
- `pre_mucinous`: min=0, max=1, mean=0.12, std=0.33
- `pre_mri_tumor_volume_ml`: min=5.0, max=69.5, mean=29.49, std=12.80
- `pre_mri_ADC_mean`: min=0.49, max=1.44, mean=0.95, std=0.15
- `pre_CEA_ng_ml`: min=0.6, max=76.4, mean=8.45, std=7.20
- `weeks_since_end_treatment`: min=5, max=16, mean=9.56, std=2.04
- `dre_fixed_to_structures`: min=0, max=1, mean=0.06, std=0.23
- `endo_mucosal_irregularity`: min=0, max=1, mean=0.37, std=0.48
- `endo_stenosis`: min=0, max=1, mean=0.10, std=0.30
- `endo_pct_wall_involved`: min=0, max=62, mean=15.71, std=16.72
- `bx_done`: min=0, max=1, mean=0.51, std=0.50
- `bx_n_samples`: min=0, max=6, mean=1.97, std=2.21
- `mri_mrTRG`: min=1, max=5, mean=2.69, std=1.31
- `mri_split_scar`: min=0, max=1, mean=0.28, std=0.45
- `mri_DWI_restriction`: min=0.0, max=1.0, mean=0.51, std=0.50
- `mri_ADC_mean`: min=0.34, max=2.26, mean=1.24, std=0.32
- `mri_tumor_volume_ml`: min=0.0, max=54.2, mean=11.79, std=9.35
- `mri_volume_reduction_pct`: min=-9, max=100, mean=60.35, std=24.36
- `mri_wall_thickness_mm`: min=2.0, max=16.9, mean=6.94, std=3.27
- `mri_lumen_irregularity`: min=0, max=1, mean=0.36, std=0.48
- `mri_mucin_pool`: min=0, max=1, mean=0.04, std=0.20
- `mri_suspicious_nodes`: min=0, max=1, mean=0.18, std=0.39
- `mri_EMVI_post`: min=0, max=1, mean=0.14, std=0.34
- `mri_MRF_involved`: min=0, max=1, mean=0.12, std=0.33
- `ct_lung_lesion`: min=0, max=1, mean=0.04, std=0.19
- `ct_liver_lesion`: min=0, max=1, mean=0.04, std=0.19
- `ct_distant_nodes`: min=0, max=1, mean=0.09, std=0.28
- `ct_distant_progression`: min=0, max=1, mean=0.15, std=0.36
- `ct_rectal_wall_thickening`: min=0, max=1, mean=0.43, std=0.50
- `ct_perirectal_stranding`: min=0, max=1, mean=0.24, std=0.43
- `post_CEA_ng_ml`: min=0.5, max=16.8, mean=3.45, std=2.48

## 8. Categorical values
- `sex`: ['F', 'M']
- `pre_T_stage`: ['T2', 'T3', 'T4']
- `pre_N_stage`: ['N0', 'N1', 'N2']
- `pre_circumferential_extent`: ['50-75%', '<50%', '>75%']
- `regimen`: ['CRT_only', 'CRT_plus_consolidation', 'TNT']
- `dre_finding`: ['irregularity_or_nodule', 'no_abnormality', 'palpable_mass', 'smooth_scar']
- `endo_finding`: ['flat_white_scar', 'residual_nodule', 'scar_with_telangiectasia', 'superficial_ulcer', 'visible_tumor']
- `bx_result`: ['inadequate', 'negative', 'not_done', 'positive']
- `mri_T2_residual_signal`: ['intermediate_signal', 'large_intermediate_signal', 'none', 'thin_scar_only']
- `bx_mucinous_component`: [-1, 0, 1] (note: -1 = not applicable)

## 9. Exam group sizes
- baseline: 12 features
- treatment: 2 features
- dre: 2 features
- endoscopy: 4 features
- biopsy: 4 features
- mri: 13 features
- ct: 6 features
- post_labs: 1 features
- Total feature columns: 44

## 10. Hard-stop rule coverage (preview)
- `ct_distant_progression == 1`: 120 patients | target=0 (not safe) in 120/120 → agreement 100.0% 
  Reason: Distant metastatic progression on CT: watch-and-wait contraindicated
- `bx_result == positive`: 121 patients | target=0 (not safe) in 121/121 → agreement 100.0% 
  Reason: Positive biopsy: residual viable tumor confirmed histologically
- `endo_finding == visible_tumor`: 123 patients | target=0 (not safe) in 116/123 → agreement 94.3% 
  Reason: Visible tumor on endoscopy: macroscopic residual disease
- `dre_finding == palpable_mass`: 91 patients | target=0 (not safe) in 86/91 → agreement 94.5% 
  Reason: Palpable mass on DRE: residual bulky disease
- **Any hard stop fires**: 304 patients (38.0%)
  - Of those, target=0: 293 (should be nearly all)
  - Remaining for ML: 496 patients
  - Remaining class balance: {1: 359, 0: 137}

## 11. Feature correlations with target (top 20)
- `mri_volume_reduction_pct`: 0.668
- `endo_pct_wall_involved`: 0.644
- `mri_wall_thickness_mm`: 0.631
- `mri_DWI_restriction`: 0.584
- `mri_mrTRG`: 0.567
- `mri_tumor_volume_ml`: 0.555
- `mri_lumen_irregularity`: 0.552
- `endo_mucosal_irregularity`: 0.548
- `mri_T2_residual_signal`: 0.535
- `post_CEA_ng_ml`: 0.451
- `ct_distant_progression`: 0.390
- `mri_split_scar`: 0.361
- `endo_finding`: 0.338
- `mri_EMVI_post`: 0.330
- `bx_result`: 0.326
- `mri_ADC_mean`: 0.323
- `mri_suspicious_nodes`: 0.293
- `ct_rectal_wall_thickening`: 0.289
- `ct_distant_nodes`: 0.287
- `ct_perirectal_stranding`: 0.277

## 12. bx_mucinous_component distribution
- -1: 395
- 0: 387
- 1: 18

## 13. Summary
- 800 rows, 49 columns as expected.
- Target balance: {0: 430, 1: 370}
- All target relationships verified: ✓
- No target columns in feature set: ✓
- Missing values in 5 columns, as documented.
- Hard-stop rules cover 304 patients with high target agreement.
- Remaining 496 patients for ML model.

---
*Generated by Phase 1 audit script.*