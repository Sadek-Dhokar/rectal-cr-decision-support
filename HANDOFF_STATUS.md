# HANDOFF STATUS

- Done: Baseline tree verification, test-split evaluation, exact explanation logic, Streamlit app, docs, pitch kit, security audit, and fresh-venv verification.
- Done: Test suite: 28 tests pass in both the current environment and `.venv`.
- Done: Test split was scored exactly once; `reports/metrics.json` has `test_eval_count=1`, 7 figures exist, and `reports/results.md` matches the metrics.
- Done: Retrained twice in `.venv`; both runs selected `t_low=0.29`, `t_high=0.71`.
- Done: Streamlit started successfully and returned HTTP 200; the final server was stopped.
- Broken: None known. Do not rerun test evaluation without `--force`; a rerun must disclose an incremented `test_eval_count`.
- Exact next demo step:
  1. `cd rectal-cr`
  2. `.venv\Scripts\python.exe -m streamlit run app/app.py --server.headless true --server.port 8502`
  3. Open `http://localhost:8502`
  4. Demo COMPLETE, model NOT COMPLETE, rule-fired CT progression, MRI unavailable, and the pitch evidence slides.

# Metric Number Audit

- 800 patients / 370 safe / 430 not safe: `reports/metrics.json` target counts and data audit.
- 49 columns: test shape check and `data_dictionary_5exams.csv`.
- 559 train / 121 validation / 120 test: `reports/metrics.json` split sizes.
- 342 / 75 / 79 ML-eligible: `reports/metrics.json` split sizes.
- 304 hard stops / 38%: `reports/metrics.json` rule and data audit.
- 293 of 304 (96.4%) correct / 11 false NOT COMPLETE: `reports/metrics.json` combined rule stats.
- 38 local-complete plus distant-progression patients: `reports/results.md` data audit.
- 0.29 / 0.71 thresholds: `reports/metrics.json` model thresholds and `reports/train_config.json`.
- Test full AUROC 0.987 (0.958-1.000): `reports/metrics.json:test/full/auroc`.
- Test full PPV 1.000 / NPV 0.970 / coverage 100.0% / 0 false COMPLETE: `reports/metrics.json:test/full`.
- Test ML AUROC 1.000: `reports/metrics.json:test/ml/auroc`.
- B2 test AUROC 0.995 (0.982-1.000): `reports/metrics.json:baselines.B2_logreg8_ml.auroc`.
- Validation ML NPV 0.950 with lower bound 0.824: `reports/metrics.json:model.validation_confirmation` and validation ML metric.
- Drop-exam UNCERTAIN range 65.8%-72.5%: `reports/results.md` robustness table.

# UI Redesign Status

- Done: Redesign of `app/app.py` only for presentation: local clinical-tech header, sidebar sections, pipeline stepper, decision card, gauge, exam chips, Altair explanation chart, styled batch mode and evidence expander.
- Done: `.streamlit/config.toml` uses the requested light theme.
- Done: Full suite remains green: 28 tests passed.
- Done: Streamlit returned HTTP 200 on port 8503.
- Done: AppTest walkthrough confirmed examples, manual entry, CT progression, MRI unavailable, batch empty state and age handling.
- Done: Browser visual validation captured COMPLETE and UNCERTAIN states.
- Screenshots: browser screenshots were visually attached during validation; no local image save was available through the browser tool.
- Backup: remove `app/app_backup.py` only after the passing UI verification.
