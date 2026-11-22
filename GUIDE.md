# DEMO DAY CHEAT SHEET (about 50 seconds)

Tool: decision support for low rectal cancer after chemoradiotherapy. Five exam outputs in (DRE, endoscopy, biopsy, pelvic MRI, CT), one decision for the multidisciplinary team (MDT) out: **COMPLETE RESPONSE / NOT COMPLETE / UNCERTAIN**.

---

## A. BEFORE THE PRESENTATION (do this once, 20 minutes before)

### A1. Open PowerShell in the project
```powershell
cd "C:\Industrial Computer Engineering\Clubs\IEEE\TECH4HEALTH\rectal-cr"
```

### A2. Check your demo file (120 test patients)
The file is `demo_batch_test_120.csv` and must sit in the `rectal-cr` folder. Confirm it is there:
```powershell
dir demo_batch_test_120.csv
```
Quick check (run it once now, not during the demo):
```powershell
.venv\Scripts\python.exe -c "import pandas as pd; from src.config import DATA_FILE; from src.predict import predict_batch; a=pd.read_csv(DATA_FILE,nrows=1); b=pd.read_csv('demo_batch_test_120.csv'); print('rows:',len(b)); print('missing columns:',sorted(set(a.columns)-set(b.columns))); print(predict_batch(b,log=False)['decision'].value_counts().to_dict())"
```
Expected: `rows: 120`, `missing columns: []`, then a count of decisions.
If `dir` says the file is not found, either `cd` to the folder where it is, or run the command with the full path, for example `pd.read_csv(r'C:\full\path\demo_batch_test_120.csv')`.

### A3. Start the app
```powershell
.venv\Scripts\python.exe -m streamlit run app/app.py --server.headless true --server.port 8502
```
Leave this window open. Do not close it.

### A4. Check it responds (new PowerShell window)
```powershell
curl.exe -I http://localhost:8502
```
Expect `HTTP/1.1 200 OK`.

### A5. Open the browser
Go to `http://localhost:8502`. Zoom to 110-125% (`Ctrl` and `+`). Full screen with `F11`. Close every other app and mute notifications.

### A6. Rehearse once, then reset
Upload the file once to check it works, then refresh the page (`F5`) so the demo starts clean.

### A7. Safety net
Record a backup video of the full demo (Windows: `Win + Alt + R`). If anything breaks live, play the video.

### If the port is busy
```powershell
.venv\Scripts\python.exe -m streamlit run app/app.py --server.headless true --server.port 8503
```
Then open `http://localhost:8503`.

---

## B. THE DEMO (timed)

| Time | You do | You say |
|---|---|---|
| 0:00-0:07 | Page is open on the header | "Five exam outputs in, one decision for the multidisciplinary team out. Everything runs locally, nothing leaves this computer." |
| 0:07-0:15 | Sidebar: **Input mode: Example**, choose **PT_0269** | "Visible tumor on endoscopy: a plain safety rule fires. No AI involved, and the app shows exactly which rule." |
| 0:15-0:30 | Choose **PT_0308** (green COMPLETE). Then untick **Pelvic MRI** in the sidebar. Card turns amber. | "Here the model says complete. But if the MRI is missing, the tool cannot clear its safety check, so it refers the patient to the MDT instead of guessing." |
| 0:30-0:45 | Input mode: **Batch CSV**. Upload `demo_batch_test_120.csv`. Point to the KPI cards, the outcome chart, then the **Download** button. | "The same logic runs on a whole batch of 120 test patients. Rules decide the clear cases, the model weighs the rest, and the MDT gets a downloadable table." |
| 0:45-0:52 | Back to the header or the Evidence section | "Decision support only. The MDT decides. The pipeline is demonstrated on the provided exam data; clinical validation on real cases is the next step." |

Total: about 50 seconds. If you are running long, shorten the batch step (just upload and point at the KPI cards).

### What to read off the screen (do not memorize numbers)
In the batch view, read the numbers the app shows. Do not quote accuracy or "99%". The test file has no or very few UNCERTAIN cases, which is why the what-if step exists. Say "what-if" out loud for PT_0308 and never claim the model abstained on a real test patient.

---

## C. IF THEY ASK (one-line answers)

- **Why AI at all?** "Rules handle the clear safety cases. The model only weighs the ambiguous remainder, and the rules around it keep it safe."
- **What if it wrongly says COMPLETE?** "Hard stops, an abstention band and a missing-exam policy exist to prevent that, and the MDT reviews every call."
- **Is the data real?** "These are the provided exam outputs. We make no claim of clinical validation: that needs external real cases with the same columns, split by patient."
- **Why are scores so high?** "The dataset is highly separable. That is a dataset property, not clinical-grade performance."
- **Privacy?** "100% local, no cloud, no API calls, and the audit log stores only a hashed patient ID."

---

## D. IF SOMETHING BREAKS

| Problem | Fix |
|---|---|
| Page won't load | Check the PowerShell window running Streamlit is still open. Re-run step A3. |
| Upload shows an error | Upload `demo_batch_test_120.csv` (the raw 49-column file), **not** `rectal_cr_decisions.csv` (that is the output). |
| App froze | Press `F5`, then `Ctrl + C` in the Streamlit window and re-run A3. |
| Live demo impossible | Play the backup video from step A7. |

---

## E. DO NOT RUN ON DEMO DAY
```powershell
# DO NOT: scores the test set again (needs --force and must be disclosed)
python -m src.evaluate --split test

# DO NOT: retrain the model
python -m src.train
```
Also do not edit any code before presenting.

---

## F. OPTIONAL: PROOF IT RUNS OFFLINE (only if you can afford 5 seconds)
Turn Wi-Fi off before you start the demo, then say: "The Wi-Fi is off. Everything you see runs on this laptop." Do this **before** the demo, not during it.

---

## G. QUICK CHECKLIST (tick before going on stage)
- [ ] App running, page loads at `http://localhost:8502`
- [ ] `demo_batch_test_120.csv` exists and uploaded once successfully
- [ ] Page refreshed so the demo starts clean
- [ ] Browser zoom and full screen set, notifications muted
- [ ] Backup video saved and easy to open
- [ ] You know the three moves: PT_0269, PT_0308 + untick MRI, then batch
