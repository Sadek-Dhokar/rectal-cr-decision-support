# 60-Second Demo Script

## 0:00-0:12 - COMPLETE
Open the app and select the first COMPLETE example. Point out the green COMPLETE card, the model probability, the thresholds 0.29 and 0.71, and the exam-contribution chart.

## 0:12-0:24 - Model NOT COMPLETE
Select the low-probability model NOT COMPLETE example. Explain that no hard stop fired and that the model is one of the three layers.

## 0:24-0:36 - Rule fired
Select the rule-fired example or set CT distant progression to 1 in manual entry. Show red NOT COMPLETE and the exact hard-stop reason.

## 0:36-0:48 - What-if abstention
Mark MRI unavailable for an otherwise complete-looking example. The result becomes amber UNCERTAIN and the MDT is named as the decision owner.

## 0:48-1:00 - Evidence and limits
Show the ablation and risk-coverage figures. State that the test full-pipeline AUROC was 0.987 and the 8-feature baseline achieved 0.995, essentially parity. State that dropping an exam produced approximately 65.8%-72.5% UNCERTAIN in the robustness results.

Optional offline proof: disconnect Wi-Fi, reload the local app and repeat one example. The app and model run without network calls.
