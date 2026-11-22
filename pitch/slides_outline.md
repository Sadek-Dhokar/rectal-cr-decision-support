# Tech4Health Pitch - 8 Slides

Timing: 4-minute pitch plus 2-minute Q&A. Organizers have not confirmed whether the speaking slot is timed; this timing is an internal estimate.

## 1. Problem - 30 seconds
After chemoradiotherapy, deciding whether a low rectal cancer patient may enter watch-and-wait is safety-critical. The tool takes structured outputs from five exams and leaves the final decision to the MDT.

## 2. Data - 25 seconds
The provided data has 800 patients: 370 safe and 430 not safe. The supplied exam-output schema is 49 columns. Splits are 559 train, 121 validation and 120 test patients.

## 3. Architecture - 35 seconds
Layer 1 uses plain rules for CT distant progression, positive biopsy, visible tumor on endoscopy and palpable DRE mass. AI is not optimal for every micro-task. Layer 2 uses uncalibrated `logistic_l2`. Layer 3 abstains outside its thresholds or when an exam is missing.

## 4. Safety and abstention - 35 seconds
Thresholds are 0.29 and 0.71. Hard stops covered 304 patients, or 38%, and agreed with the label in 293 of 304 cases (96.4%). Whole-exam removal creates deliberate UNCERTAIN output rather than a false definite result.

## 5. Demo - 50 seconds
Show one COMPLETE example, one model NOT COMPLETE example, one rule-fired example, MRI unavailable, and the ablation/risk-coverage evidence.

## 6. Results - 35 seconds
Test full-pipeline AUROC is 0.987 (95% CI 0.958-1.000), with 0 false-COMPLETE calls and 100.0% coverage. ML-only test AUROC is 1.000 (95% CI 1.000-1.000).

## 7. Honest limitations - 25 seconds
The 8-feature B2 baseline test AUROC is 0.995 (95% CI 0.982-1.000), essentially at parity. Validation ML-only NPV point estimate is 0.950, but the lower 95% bound is 0.824 and misses the 0.90 target.

## 8. Clinical path and ask - 15 seconds
The pipeline, decision logic, abstention behaviour and interface are demonstrated on the provided exam-output data; clinical validation requires external real cases with the same schema, split by patient.
