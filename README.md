# Tech4Health Rectal-CR Decision Support

This local prototype supports discussion after chemoradiotherapy for low rectal cancer. It consumes the structured outputs of DRE, endoscopy, biopsy, pelvic MRI and CT as a 49-column row. It is not an autonomous diagnostic device.

## Three-layer architecture

1. **Plain rules:** validate inputs and apply four hard stops for distant CT progression, positive biopsy, visible tumor on endoscopy, or a palpable DRE mass.
2. **Logistic model:** an uncalibrated `logistic_l2` model scores patients who pass the hard-stop layer.
3. **Abstention policy:** thresholds `t_low=0.29` and `t_high=0.71`, plus a missing-exam policy that forces UNCERTAIN unless all five exams are available.

AI is not used for every micro-task. The deterministic safety rules handle non-negotiable hard stops, while logistic regression handles the remaining tabular discrimination.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m src.train
python -m streamlit run app/app.py
```

The application is intended to run fully locally. It does not call an external API or send patient data over the network.

## Tests

```powershell
python -m pytest -q
```

The test suite covers schema, leakage, hard stops, missing-exam policy, exact explanation logit reproduction, audit hashing and Streamlit app behavior.

## Evaluation limits

- The test population is 120 patients; the ML-eligible population is 79.
- Full-pipeline test AUROC was 0.987 (95% CI 0.958-1.000), with 0 false-COMPLETE calls and 100.0% coverage.
- ML-only test AUROC was 1.000 (95% CI 1.000-1.000).
- The 8-feature B2 baseline had test AUROC 0.995 (95% CI 0.982-1.000), essentially at parity with the full ML model.
- The validation ML-only NPV point estimate was 0.950, but the 95% lower bound was 0.824; this did not meet the 0.90 NPV target.
- Dropping a whole exam made the deployed pipeline answer UNCERTAIN for approximately 65.8%-72.5% in the drop-exam robustness scenarios.
- Hard stops covered 304 patients (38%) and agreed with the label for 293 of 304 (96.4%); 11 true-safe patients were incorrectly forced to NOT COMPLETE.

The pipeline, decision logic, abstention behaviour and interface are demonstrated on the provided exam-output data; clinical validation requires external real cases with the same schema, split by patient.

This tool uses structured exam outputs, not images. It does not replace radiology, pathology or MDT review.

## Privacy and audit

The audit log stores only a salted SHA-256 patient identifier, timestamp, decision, probability, source, rule and model version. No raw clinical values are intended to be stored. The demo salt must be replaced by a secret salt outside the repository before real use.
