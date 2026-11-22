# 12-Question Q&A Sheet

## 1. Why not AI for every micro-task?
We use plain rules for clear, deterministic safety hard stops. AI handles only the residual tabular ML population. AI is not optimal for every micro-task.

## 2. Why 800 rows?
The provided dataset has 800 rows, with 370 safe and 430 not-safe patients. It is suitable for a prototype, but not large enough to establish clinical-grade external performance.

## 3. Why abstain?
Abstention preserves coverage and safety when thresholds or required exam evidence are uncertain. If a whole required exam is unavailable, the app refuses a definite COMPLETE call.

## 4. What if it wrongly says COMPLETE?
A false COMPLETE could be dangerous. The test result made 0 false-COMPLETE calls, but the small test population and confidence intervals do not make it clinically safe; the MDT must review every result.

## 5. How is privacy handled?
The app is fully local. The audit log stores a salted SHA-256 patient identifier, timestamp, decision, probability, source, rule and model version, not raw clinical values. The demo salt must be replaced by a secret salt outside the repository for real use.

## 6. How would it deploy in Tunisia?
Deployment needs local-language workflow review, MDT integration, secure hosting if electronic data is used, institutional governance, external validation and a prospective silent trial. The current demo is local-only.

## 7. Why these five exams?
The provided schema combines DRE, endoscopy, biopsy, pelvic MRI and CT because together they cover physical, macroscopic, histological and cross-sectional evidence. It does not use raw images.

## 8. Why no LLM in the decision path?
A generative model could introduce untraceable reasoning and is unnecessary for tabular decision thresholds. The current path is deterministic, auditable and fully local.

## 9. Why does the simple model match the full model?
The 8-feature B2 test AUROC was 0.995 (95% CI 0.982-1.000), essentially at parity with the full ML-only AUROC of 1.000 (95% CI 1.000-1.000). The hand-picked features capture much of the signal.

## 10. What is the NPV-target miss?
On validation, ML-only NPV was 0.950, but the 95% lower bound was 0.824, below the 0.90 target. We reported this limitation and did not tune to hide it.

## 11. What happens when one exam is missing?
The missing-exam policy forces UNCERTAIN unless a hard stop fires. Drop-exam robustness showed roughly 65.8%-72.5% UNCERTAIN on the deployed population.

## 12. What is the next required validation?
External real cases with the same schema, split by patient, are required. Then a prospective silent trial should measure workflow impact before clinical use.
