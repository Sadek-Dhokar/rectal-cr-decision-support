import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from src.config import (
    APP_DIR,
    EXAM_DISPLAY_NAMES,
    EXAM_GROUPS,
    EXAM_UNAVAILABLE_COLUMNS,
)
from src.explain import explain_patient_dataframe
from src.predict import load_model_and_config, predict_batch, predict_single
from src.rules import exam_availability
from src.schema import build_schema, validate_patient, validate_row


st.set_page_config(page_title="Tech4Health Rectal-CR Decision Support", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"


@st.cache_resource
def get_model():
    return load_model_and_config()


@st.cache_data
def get_examples():
    path = APP_DIR / "examples.json"
    if not path.exists():
        return {"examples": []}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def get_defaults():
    path = APP_DIR / "defaults.json"
    if not path.exists():
        return {"values": {}}
    return json.loads(path.read_text(encoding="utf-8"))


pipe, config = get_model()
schema = build_schema()
defaults = get_defaults()
base_values = dict(defaults.get("values", {}))

st.markdown(
    """
    <style>
    :root{--navy:#0B2545;--deep:#13315C;--primary:#1D5FD6;--mid:#4F8FE8;
      --light:#DCEBFC;--mist:#F2F7FD;--border:#D5E3F5;--muted:#5B6B82;
      --teal:#12906B;--teal-bg:#E6F6F0;--red:#C23B3B;--red-bg:#FCEAEA;
      --amber:#B7791F;--amber-bg:#FFF6E0;--orange:#E07B39;}
    html,body,[class*="css"]{font-family:"Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;}
    .block-container{max-width:1180px!important;padding-top:1.2rem;padding-bottom:1rem;}
    #MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;}
    [data-testid="stSidebar"]{background:var(--mist);border-right:1px solid var(--border);}
    [data-testid="stSidebar"]>div:first-child{padding-top:1.25rem;}
    .header-banner{border-radius:14px;padding:20px 24px;color:white;
      background:linear-gradient(125deg,var(--deep),var(--primary),var(--mid));
      display:flex;align-items:center;justify-content:space-between;
      box-shadow:0 4px 18px rgba(19,49,92,.10);margin-bottom:18px;}
    .header-main{display:flex;align-items:center;gap:14px;}
    .header-logo{width:52px;height:52px;border-radius:12px;background:rgba(255,255,255,.15);padding:7px;}
    .header-title{font-size:30px;font-weight:700;line-height:1.1;}
    .header-subtitle{font-size:14px;opacity:.92;margin-top:6px;}
    .header-pills{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end;max-width:330px;}
    .header-pill{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.26);
      border-radius:999px;padding:7px 10px;font-size:12px;}
    .stepper{display:flex;align-items:center;gap:10px;margin:10px 0 22px;}
    .step{flex:1;border:1px solid var(--border);background:white;border-radius:14px;padding:12px 14px;
      color:var(--muted);box-shadow:0 2px 10px rgba(19,49,92,.04);}
    .step.active{border-color:#90B3ED;background:#EDF4FF;color:var(--deep);
      box-shadow:0 5px 18px rgba(29,95,214,.15);}
    .step-title{font-size:15px;font-weight:700;}
    .step-arrow{color:#91A3BA;font-size:19px;}
    .decision-card{border-radius:14px;border:1px solid var(--border);padding:22px;
      box-shadow:0 4px 18px rgba(19,49,92,.08);animation:fadein 200ms ease-out;}
    .decision-card.complete{background:var(--teal-bg);border-color:#AEDDCF;}
    .decision-card.not_complete{background:var(--red-bg);border-color:#ECC0C0;}
    .decision-card.uncertain{background:var(--amber-bg);border-color:#E8CD92;}
    .decision-icon{font-size:30px;float:right;font-weight:700;}
    .decision-label{font-size:30px;font-weight:750;line-height:1.1;margin:4px 0 8px;}
    .decision-label.complete{color:var(--teal)}.decision-label.not_complete{color:var(--red)}
    .decision-label.uncertain{color:var(--amber)}
    .decision-meaning{font-size:15px;color:var(--navy)}
    .panel-card{border:1px solid var(--border);border-radius:14px;background:white;padding:18px;
      box-shadow:0 4px 18px rgba(19,49,92,.08);}
    .why-callout{border-left:4px solid var(--primary);background:var(--mist);border-radius:8px;
      padding:10px 12px;margin-bottom:10px;color:var(--navy);font-size:13.5px;}
    .kpi-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:8px 0 18px;}
    .kpi-card{border:1px solid var(--border);border-radius:14px;background:white;padding:14px 12px;
      box-shadow:0 2px 12px rgba(19,49,92,.05);}
    .kpi-label{color:var(--muted);font-size:12.5px}.kpi-value{font-size:24px;font-weight:700;
      color:var(--navy);font-variant-numeric:tabular-nums;}
    .exam-strip{display:flex;gap:7px;flex-wrap:wrap;margin:8px 0 12px;}
    .exam-chip{border:1px solid var(--border);border-radius:999px;padding:6px 9px;font-size:12.5px;
      background:white;color:var(--navy);}
    .exam-chip.available{border-color:#AEDDCF;color:#0E7152;background:#F3FBF8;}
    .exam-chip.missing{color:var(--muted);background:#EEF2F7;}
    .gauge-wrap{margin-top:18px}.gauge{position:relative;height:18px;border-radius:999px;overflow:hidden;
      display:flex;border:1px solid var(--border);}
    .gauge-low{width:29%;background:#F7CCCC}.gauge-mid{width:42%;background:#FFE9AF}
    .gauge-high{width:29%;background:#CFEEDF}
    .gauge-pointer{position:absolute;top:-5px;width:4px;height:28px;background:var(--deep);
      border:2px solid white;border-radius:3px;box-shadow:0 1px 5px rgba(11,37,69,.25);
      transition:left 220ms ease;}
    .gauge-labels{display:flex;justify-content:space-between;color:var(--muted);font-size:11.5px;
      margin-top:5px;font-variant-numeric:tabular-nums;}
    .stat-chips{display:flex;gap:7px;margin-top:12px;flex-wrap:wrap;}
    .stat-chip{border:1px solid var(--border);border-radius:999px;padding:6px 9px;background:white;
      color:var(--navy);font-size:12.5px;}
    .rule-state{border:1px solid #F0CD7A;background:#FFF8E7;border-radius:10px;padding:11px 12px;
      color:var(--navy);font-size:13.5px;}
    .error-card{border:1px solid #ECC0C0;border-radius:14px;background:var(--red-bg);
      color:#7D2626;padding:14px 16px;}
    .footer{border-top:1px solid var(--border);margin-top:24px;padding-top:13px;color:var(--muted);
      text-align:center;font-size:12.5px;}
    .empty-state{border:1px dashed var(--border);border-radius:14px;background:var(--mist);
      text-align:center;padding:34px;color:var(--navy);margin:18px 0;}
    .empty-orb{width:64px;height:64px;border-radius:50%;border:10px solid var(--light);
      border-top-color:var(--primary);margin:0 auto 12px;}
    div[data-testid="stTabs"] button{font-size:14px;font-weight:650;}
    div[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:14px;overflow:hidden;}
    @keyframes fadein{from{opacity:.5;transform:translateY(3px)}to{opacity:1;transform:none}}
    @media(max-width:900px){.header-banner{align-items:flex-start}.header-pills{display:none}
      .kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.header-title{font-size:24px}}
    </style>
    <div class="header-banner">
      <div class="header-main">
        <div class="header-logo">
          <svg viewBox="0 0 56 56" aria-hidden="true">
            <circle cx="28" cy="28" r="21" fill="none" stroke="white" stroke-width="3"/>
            <path d="M20 15 C14 24 16 38 28 43 C40 38 42 24 36 15" fill="none" stroke="white" stroke-width="3"/>
            <path d="M22 28 L27 33 L36 23" fill="none" stroke="white" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <div><div class="header-title">Rectal Cancer after Chemoradiotherapy</div>
          <div class="header-subtitle">Complete-response decision support for the multidisciplinary team</div>
        </div>
      </div>
      <div class="header-pills">
        <div class="header-pill">✓ Runs 100% locally</div>
        <div class="header-pill">✓ No data leaves this computer</div>
        <div class="header-pill">! Decision support only</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Patient input")
mode = st.sidebar.radio("Input mode", ["Example", "Manual entry", "Batch CSV"])

if mode == "Example":
    example_payload = get_examples()
    examples = example_payload.get("examples", [])
    if not examples:
        st.warning("No examples are available. Run the test evaluation or use manual entry.")
        patient = dict(base_values)
        patient_id = "DEMO"
    else:
        labels = [
            f"{i + 1}. {e['patient_id']} - {e['expected_decision']}"
            for i, e in enumerate(examples)
        ]
        selected = st.sidebar.selectbox("Choose an example", labels)
        example = examples[labels.index(selected)]
        patient = dict(base_values)
        patient.update(example["data"])
        patient_id = str(patient.get("patient_id", "DEMO"))
elif mode == "Manual entry":
    patient = dict(base_values)
    patient_id = st.sidebar.text_input("Patient ID (optional)", value="DEMO")
    patient["patient_id"] = patient_id
else:
    patient = None

if mode != "Batch CSV":
    st.sidebar.markdown("### Exams available")
    st.sidebar.caption("Untick to simulate a missing exam")
    available = {}
    for exam in ("dre", "endoscopy", "biopsy", "mri", "ct"):
        available[exam] = st.sidebar.checkbox(
            EXAM_DISPLAY_NAMES[exam], value=True, key=f"available_{exam}"
        )
        st.sidebar.caption("✓ available" if available[exam] else "○ missing")

    if mode == "Manual entry":
        complete_exams = sum(1 for exam in ("dre","endoscopy","biopsy","mri","ct") if available[exam])
        tab_names = [
            "DRE", "Endoscopy", "Biopsy", "Pelvic MRI", "CT",
            "Baseline/Treatment",
        ]
        st.markdown(
            f'<div class="stat-chip">{complete_exams} of 5 exams complete</div>',
            unsafe_allow_html=True,
        )
        tabs = st.tabs(tab_names)
        group_names = ["dre", "endoscopy", "biopsy", "mri", "ct"]

        for tab, group in zip(tabs[:5], group_names):
            with tab:
                for column in EXAM_GROUPS[group]:
                    spec = schema.get(column, {})
                    label = column
                    value = patient.get(column)
                    if "allowed_values" in spec:
                        allowed = spec["allowed_values"]
                        index = allowed.index(value) if value in allowed else 0
                        patient[column] = st.selectbox(
                            label, allowed, index=index, key=f"manual_{column}"
                        )
                    elif "min" in spec:
                        missing_allowed = spec.get("n_missing", 0) > 0
                        if missing_allowed:
                            is_missing = st.checkbox(
                                f"{column} - not available",
                                value=pd.isna(value),
                                key=f"missing_{column}",
                            )
                            if is_missing:
                                patient[column] = np.nan
                                continue
                        patient[column] = st.number_input(
                            label,
                            min_value=float(spec["min"]),
                            max_value=float(spec["max"]),
                            value=float(value) if not pd.isna(value) else float(spec["min"]),
                            key=f"manual_{column}",
                        )
                    else:
                        patient[column] = st.text_input(
                            label, value="" if pd.isna(value) else str(value),
                            key=f"manual_{column}",
                        )

        with tabs[5]:
            for group in ("baseline", "treatment", "post_labs"):
                st.subheader(EXAM_DISPLAY_NAMES.get(group, group.title()))
                for column in EXAM_GROUPS[group]:
                    spec = schema.get(column, {})
                    value = patient.get(column)
                    if "allowed_values" in spec:
                        allowed = spec["allowed_values"]
                        index = allowed.index(value) if value in allowed else 0
                        patient[column] = st.selectbox(
                            column, allowed, index=index, key=f"manual_{column}"
                        )
                    elif "min" in spec:
                        missing_allowed = spec.get("n_missing", 0) > 0
                        if missing_allowed:
                            is_missing = st.checkbox(
                                f"{column} - not available",
                                value=pd.isna(value),
                                key=f"missing_{column}",
                            )
                            if is_missing:
                                patient[column] = np.nan
                                continue
                        patient[column] = st.number_input(
                            column,
                            min_value=float(spec["min"]),
                            max_value=float(spec["max"]),
                            value=float(value) if not pd.isna(value) else float(spec["min"]),
                            key=f"manual_{column}",
                        )

    for exam, is_available in available.items():
        if not is_available:
            for column in EXAM_UNAVAILABLE_COLUMNS[exam]:
                patient[column] = np.nan

    errors = validate_patient(
        patient,
        config["ml_feature_cols"],
        allow_missing=True,
    )
    if errors:
        st.error("Validation errors:\n- " + "\n- ".join(errors))

    try:
        result = predict_single(
            patient,
            pipe,
            config,
            log=False,
            validate=False,
        )
    except Exception as exc:
        st.error(f"Could not compute a result: {exc}")
        result = None

    if result is not None:
        decision = result["decision"]
        if decision == "COMPLETE RESPONSE":
            css_class, icon, meaning = "complete", "✓", "Eligible for watch-and-wait discussion"
        elif decision == "NOT COMPLETE":
            css_class, icon, meaning = "not_complete", "✕", "Continue the surgical / MDT pathway"
        else:
            css_class, icon, meaning = "uncertain", "!", "Refer to the multidisciplinary team"

        active_step = {"rule": 0, "model": 1, "policy": 2}.get(result["source"])
        step_labels = ["1 Rules (no AI)", "2 Model", "3 Safety layer (abstain)"]
        step_html = ""
        for index, label in enumerate(step_labels):
            step_html += f'<div class="step{" active" if index == active_step else ""}"><div class="step-title">{label}</div></div>'
            if index < 2:
                step_html += '<div class="step-arrow">→</div>'
        st.markdown(f'<div class="stepper">{step_html}</div>', unsafe_allow_html=True)

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown(
                f'<div class="decision-card {css_class}">'
                f'<div class="decision-icon">{icon}</div>'
                f'<div class="decision-label {css_class}">{decision}</div>'
                f'<div class="decision-meaning">{meaning}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="panel-card" style="margin-top:14px">'
                '<h4 style="margin:0 0 8px">Probability</h4>',
                unsafe_allow_html=True,
            )
            if result["probability"] is not None:
                probability = max(0.0, min(1.0, float(result["probability"])))
                st.markdown(
                    f'<div style="font-size:24px;font-weight:700;font-variant-numeric:tabular-nums">'
                    f'{probability:.3f}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="gauge-wrap"><div class="gauge">'
                    '<div class="gauge-low"></div><div class="gauge-mid"></div><div class="gauge-high"></div>'
                    f'<div class="gauge-pointer" style="left:calc({probability:.6f} * 100% - 2px)"></div>'
                    '</div><div class="gauge-labels"><span>0.00</span>'
                    f'<span>{result["t_low"]:.2f}</span><span>{result["t_high"]:.2f}</span><span>1.00</span></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="rule-state">No model probability: decided by a safety rule</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="stat-chips">'
                f'<span class="stat-chip">Source: {result["source"]}</span>'
                f'<span class="stat-chip">Thresholds: {result["t_low"]:.2f} / {result["t_high"]:.2f}</span>'
                f'<span class="stat-chip">Confidence: {result["confidence"]}</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown('<div class="panel-card"><h4 style="margin:0 0 10px">Why?</h4>', unsafe_allow_html=True)
            rule_text = result["rule"] or "No hard-stop rule fired."
            st.markdown(
                f'<div class="why-callout"><strong>Rule / policy</strong><br>{rule_text}</div>',
                unsafe_allow_html=True,
            )
            availability = exam_availability(patient)
            labels = {"dre":"DRE","endoscopy":"Endoscopy","biopsy":"Biopsy","mri":"MRI","ct":"CT"}
            chips = ""
            for key in ("dre","endoscopy","biopsy","mri","ct"):
                ok = bool(availability.get(key, False))
                cls = "available" if ok else "missing"
                text = "available" if ok else "missing"
                chips += f'<span class="exam-chip {cls}">{labels[key]}: {text}</span>'
            st.markdown(f'<div class="exam-strip">{chips}</div>', unsafe_allow_html=True)

            st.markdown('<h4 style="margin:12px 0 6px">Logit contribution by exam</h4>', unsafe_allow_html=True)
            explanation = explain_patient_dataframe(patient, pipe, config)
            chart_data = explanation[explanation["exam"] != "intercept"].copy()
            chart_data["absolute"] = chart_data["contribution"].abs()
            chart_data = chart_data.sort_values("absolute", ascending=True)
            chart = alt.Chart(chart_data).mark_bar(size=16).encode(
                y=alt.Y("exam:N", sort="-x", title=None),
                x=alt.X("contribution:Q", title="Pushes toward NOT COMPLETE ◄ ► Pushes toward COMPLETE"),
                color=alt.condition(
                    alt.datum.contribution >= 0,
                    alt.value("#1D5FD6"),
                    alt.value("#E07B39"),
                ),
                tooltip=[
                    alt.Tooltip("exam:N"),
                    alt.Tooltip("contribution:Q", format=".4f"),
                ],
            ).properties(height=260)
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
                color="#13315C", strokeWidth=1
            ).encode(x="x:Q")
            st.altair_chart(chart + zero, use_container_width=True)
            st.caption("Positive: toward COMPLETE. Negative: toward NOT COMPLETE.")
            with st.expander("Text fallback: exam contributions"):
                st.dataframe(
                    chart_data[["exam", "contribution"]].sort_values(
                        "contribution", ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Evidence: test results and saved figures"):
        metrics_path = PROJECT_ROOT / "reports" / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            test_full = metrics["eval"]["full"]
            auroc = test_full["auroc"]
            ppv = test_full["ppv_complete"]
            st.markdown(
                '<div class="kpi-grid">'
                f'<div class="kpi-card"><div class="kpi-label">Test AUROC</div><div class="kpi-value">{auroc["point"]:.3f} ({auroc["ci_lower"]:.3f}-{auroc["ci_upper"]:.3f})</div></div>'
                f'<div class="kpi-card"><div class="kpi-label">PPV COMPLETE</div><div class="kpi-value">{ppv["point"]:.3f}</div></div>'
                f'<div class="kpi-card"><div class="kpi-label">False COMPLETE</div><div class="kpi-value">{test_full["n_false_complete"]}</div></div>'
                f'<div class="kpi-card"><div class="kpi-label">Coverage</div><div class="kpi-value">{test_full["coverage_pct"]:.1f}%</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )
        figure_files = [
            ("ablation_by_exam.png", "Ablation by exam"),
            ("risk_coverage.png", "Risk-coverage"),
            ("calibration_ml.png", "Calibration"),
        ]
        cols = st.columns(3)
        for col, (filename, caption) in zip(cols, figure_files):
            path = PROJECT_ROOT / "reports" / "figures" / filename
            if path.exists():
                with col:
                    st.image(str(path), caption=caption, use_container_width=True)
        st.caption(
            "Near-ceiling scores reflect the dataset, not clinical-grade performance."
        )

    st.markdown(
        '<div class="footer">Decision support only; the MDT decides.<br>'
        'Prototype: not a medical device; clinical validation on external real cases is required.</div>',
        unsafe_allow_html=True,
    )

else:
    uploaded = st.file_uploader(
        "Upload a 49-column CSV",
        type=["csv"],
        help="The upload is processed locally. Required columns use the 49-column exam-output schema.",
    )
    if uploaded is None:
        st.markdown(
            '<div class="empty-state"><div class="empty-orb"></div>'
            '<strong>Choose an example or enter a patient to begin</strong><br>'
            '<span style="color:#5B6B82">For batch mode, upload a CSV with the 49-column schema.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as exc:
            st.markdown(
                f'<div class="error-card"><strong>Could not read CSV</strong><br>{str(exc)}</div>',
                unsafe_allow_html=True,
            )
        else:
            missing_columns = [
                column for column in config["ml_feature_cols"]
                if column not in batch.columns
            ]
            if missing_columns:
                st.markdown(
                    '<div class="error-card"><strong>Bad columns</strong><br>'
                    'Missing required columns:<br>- '
                    + "<br>- ".join(missing_columns)
                    + '</div>',
                    unsafe_allow_html=True,
                )
            else:
                schema_errors = []
                for index, row in batch.iterrows():
                    row_errors = validate_row(row.to_dict(), schema)
                    if row_errors:
                        schema_errors.append(
                            f"row {index}: " + "; ".join(row_errors)
                        )
                if schema_errors:
                    st.markdown(
                        '<div class="error-card"><strong>Validation errors</strong><br>- '
                        + "<br>- ".join(schema_errors[:20])
                        + '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    try:
                        output = predict_batch(
                            batch,
                            pipe,
                            config,
                            log=False,
                            validate=False,
                        )
                    except Exception as exc:
                        st.markdown(
                            f'<div class="error-card"><strong>Prediction failed cleanly</strong><br>{str(exc)}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        counts = output["decision"].value_counts()
                        total = len(output)
                        complete_count = int(counts.get("COMPLETE RESPONSE", 0))
                        not_complete_count = int(counts.get("NOT COMPLETE", 0))
                        uncertain_count = int(counts.get("UNCERTAIN", 0))
                        rule_count = int(output["source"].eq("rule").sum())
                        rule_pct = (100.0 * rule_count / total) if total else 0.0

                        st.markdown(
                            '<div class="kpi-grid">'
                            f'<div class="kpi-card"><div class="kpi-label">Total</div><div class="kpi-value">{total}</div></div>'
                            f'<div class="kpi-card"><div class="kpi-label">COMPLETE</div><div class="kpi-value">{complete_count}</div></div>'
                            f'<div class="kpi-card"><div class="kpi-label">NOT COMPLETE</div><div class="kpi-value">{not_complete_count}</div></div>'
                            f'<div class="kpi-card"><div class="kpi-label">UNCERTAIN</div><div class="kpi-value">{uncertain_count}</div></div>'
                            f'<div class="kpi-card"><div class="kpi-label">Decided by rules</div><div class="kpi-value">{rule_pct:.1f}%</div></div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                        count_df = counts.rename_axis("decision").reset_index(name="count")
                        outcome_chart = alt.Chart(count_df).mark_bar(size=34).encode(
                            x=alt.X("decision:N", sort=["COMPLETE RESPONSE","NOT COMPLETE","UNCERTAIN"], title=None),
                            y=alt.Y("count:Q", title="Patients"),
                            color=alt.Color(
                                "decision:N",
                                scale=alt.Scale(
                                    domain=["COMPLETE RESPONSE","NOT COMPLETE","UNCERTAIN"],
                                    range=["#12906B","#C23B3B","#B7791F"],
                                ),
                                legend=None,
                            ),
                            tooltip=["decision:N","count:Q"],
                        ).properties(height=220)
                        st.markdown('<div class="panel-card"><h4 style="margin:0 0 8px">Outcome counts</h4>', unsafe_allow_html=True)
                        st.altair_chart(outcome_chart, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                        def color_decision(value):
                            if value == "COMPLETE RESPONSE":
                                return "background-color:#E6F6F0;color:#12906B;font-weight:700"
                            if value == "NOT COMPLETE":
                                return "background-color:#FCEAEA;color:#C23B3B;font-weight:700"
                            return "background-color:#FFF6E0;color:#B7791F;font-weight:700"

                        styled_output = output.style.map(
                            color_decision,
                            subset=["decision"],
                        )
                        st.dataframe(
                            styled_output,
                            use_container_width=True,
                        )
                        csv_bytes = output.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download decisions CSV",
                            csv_bytes,
                            file_name="rectal_cr_decisions.csv",
                            mime="text/csv",
                        )

    st.markdown(
        '<div class="footer">Decision support only; the MDT decides.<br>'
        'Prototype: not a medical device; clinical validation on external real cases is required.</div>',
        unsafe_allow_html=True,
    )
