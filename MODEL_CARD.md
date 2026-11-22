# Model Card

## Intended use

Decision support for low rectal cancer after chemoradiotherapy. Users may compare a structured exam-output row with:

- COMPLETE RESPONSE,
- NOT COMPLETE, or
- UNCERTAIN (refer to MDT).

The intended decision owner is the multidisciplinary team. The tool is not autonomous and must not issue treatment instructions.

## Out-of-scope use

- Diagnosis or treatment planning outside post-CRT low rectal cancer.
- Raw MRI, CT or pathology images.
- Survival or recurrence prediction.
- Pregnant, pediatric or otherwise outside-schema populations.
- Standalone clinical use without external validation.

## Data

The provided dataset contains 800 patients and 49 columns. Target counts are 370 safe and 430 not safe. The split is 559 train, 121 validation and 120 test patients. ML-eligible counts are 342 train, 75 validation and 79 test patients.

The target combines local complete response with no distant progression. Hard stops cover 304 patients (38%) and agree with the target for 293 (96.4%); 11 true-safe patients are forced to NOT COMPLETE.

## Evaluation

The test split was scored exactly once. Test metrics below are from `reports/metrics.json`.

| Population | n | AUROC (95% CI) | PPV COMPLETE (95% CI) | NPV NOT COMPLETE (95% CI) | False COMPLETE | Coverage |
|---|---:|---|---|---|---:|---:|
| Full deployed pipeline | 120 | 0.987 (0.958-1.000) | 1.000 (1.000-1.000) | 0.970 (0.923-1.000) | 0 | 100.0% |
| ML-only | 79 | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 0.962 (0.875-1.000) | 0 | 100.0% |

The full-pipeline PPV and NPV include the safety-layer behavior. The ML-only NPV does not rescue the validation target miss.

### Subgroups

Age, sex, regimen, distance, and missingness subgroup estimates are available in `reports/results.md`. They are small-test subgroups and exploratory rather than reliable subgroup guarantees.

### Baselines

The 8-feature B2 logistic baseline achieved test AUROC 0.995 (95% CI 0.982-1.000), essentially at parity with the full ML-only AUROC of 1.000 (95% CI 1.000-1.000). The value of the full pipeline is the safety layer, abstention, per-exam explanation and interface, not a large discrimination gain.

## Failure modes

- False COMPLETE calls can occur from model or probability-threshold error; the test made 0 false-COMPLETE calls, but confidence intervals and small sample size remain important.
- Biopsy false negatives may miss residual viable tumor.
- Scar tissue can mimic tumor on MRI or endoscopy.
- Hard stops can make conservative false NOT COMPLETE calls. The rules made 11 false NOT COMPLETE calls in the provided data.
- Reader, scanner or protocol shifts can change feature distributions and calibration.
- Missing a whole required exam forces UNCERTAIN, which deliberately reduces coverage.

## NPV target miss

The validation ML-only NPV point estimate was 0.950, but the 95% lower bound was 0.824, below the 0.90 target. This is reported and was not tuned away.

## Requirements before clinical use

- External real cases with the same schema, split by patient.
- A prospective silent trial with MDT workflow review.
- Reader/scanner drift monitoring and recalibration.
- External privacy, security and governance review.

The pipeline, decision logic, abstention behaviour and interface are demonstrated on the provided exam-output data; clinical validation requires external real cases with the same schema, split by patient.
