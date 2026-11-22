# 45-Second Problem Script

After chemoradiotherapy for low rectal cancer, clinicians must decide whether watch-and-wait is safe. The decision combines DRE, endoscopy, biopsy, pelvic MRI and CT. A confident mistake can be dangerous.

Our prototype turns 49 structured exam-output fields into three decisions: COMPLETE RESPONSE, NOT COMPLETE, or UNCERTAIN for MDT review. Plain safety rules handle hard stops, while a simple logistic model handles the remaining tabular cases. The tool deliberately abstains when evidence is missing. It does not replace the MDT or validate a treatment plan.
