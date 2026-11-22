from html import escape


CSS = """
<style>
:root {
  --navy:#0B2545; --deep:#13315C; --primary:#1D5FD6; --mid:#4F8FE8;
  --light:#DCEBFC; --mist:#F2F7FD; --border:#D5E3F5; --muted:#5B6B82;
  --teal:#12906B; --teal-bg:#E6F6F0; --red:#C23B3B; --red-bg:#FCEAEA;
  --amber:#B7791F; --amber-bg:#FFF6E0; --orange:#E07B39;
}
html, body, [class*="css"] {
  font-family:"Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;
}
.block-container {max-width:1180px!important;padding-top:1.2rem;padding-bottom:1rem;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden;}
[data-testid="stSidebar"] {background:var(--mist);border-right:1px solid var(--border);}
[data-testid="stSidebar"] > div:first-child {padding-top:1.25rem;}
.header-banner {
  border-radius:14px;padding:20px 24px;color:white;
  background:linear-gradient(125deg,var(--deep),var(--primary),var(--mid));
  display:flex;align-items:center;justify-content:space-between;
  box-shadow:0 4px 18px rgba(19,49,92,.10);margin-bottom:18px;
}
.header-main {display:flex;align-items:center;gap:14px;}
.header-logo {width:52px;height:52px;border-radius:12px;background:rgba(255,255,255,.15);padding:7px;}
.header-title {font-size:30px;font-weight:700;line-height:1.1;}
.header-subtitle {font-size:14px;opacity:.92;margin-top:6px;}
.header-pills {display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end;max-width:330px;}
.header-pill {background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.26);border-radius:999px;padding:7px 10px;font-size:12px;}
.stepper {display:flex;align-items:center;gap:10px;margin:10px 0 22px;}
.step {flex:1;border:1px solid var(--border);background:white;border-radius:14px;padding:12px 14px;color:var(--muted);box-shadow:0 2px 10px rgba(19,49,92,.04);}
.step.active {border-color:#90B3ED;background:#EDF4FF;color:var(--deep);box-shadow:0 5px 18px rgba(29,95,214,.15);}
.step-title {font-size:15px;font-weight:700;}
.step-arrow {color:#91A3BA;font-size:19px;}
.decision-card {border-radius:14px;border:1px solid var(--border);padding:22px;box-shadow:0 4px 18px rgba(19,49,92,.08);animation:fadein 200ms ease-out;}
.decision-card.complete {background:var(--teal-bg);border-color:#AEDDCF;}
.decision-card.not_complete {background:var(--red-bg);border-color:#ECC0C0;}
.decision-card.uncertain {background:var(--amber-bg);border-color:#E8CD92;}
.decision-icon {font-size:30px;float:right;font-weight:700;}
.decision-label {font-size:30px;font-weight:750;line-height:1.1;margin:4px 0 8px;}
.decision-label.complete {color:var(--teal);} .decision-label.not_complete {color:var(--red);} .decision-label.uncertain {color:var(--amber);}
.decision-meaning {font-size:15px;color:var(--navy);}
.panel-card {border:1px solid var(--border);border-radius:14px;background:white;padding:18px;box-shadow:0 4px 18px rgba(19,49,92,.08);}
.why-callout {border-left:4px solid var(--primary);background:var(--mist);border-radius:8px;padding:10px 12px;margin-bottom:10px;color:var(--navy);font-size:13.5px;}
.kpi-grid {display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:8px 0 18px;}
.kpi-card {border:1px solid var(--border);border-radius:14px;background:white;padding:14px 12px;box-shadow:0 2px 12px rgba(19,49,92,.05);}
.kpi-label {color:var(--muted);font-size:12.5px;}
.kpi-value {font-size:24px;font-weight:700;color:var(--navy);font-variant-numeric:tabular-nums;}
.exam-strip {display:flex;gap:7px;flex-wrap:wrap;margin:8px 0 12px;}
.exam-chip {border:1px solid var(--border);border-radius:999px;padding:6px 9px;font-size:12.5px;background:white;color:var(--navy);}
.exam-chip.available {border-color:#AEDDCF;color:#0E7152;background:#F3FBF8;}
.exam-chip.missing {color:var(--muted);background:#EEF2F7;}
.gauge-wrap {margin-top:18px;}
.gauge {position:relative;height:18px;border-radius:999px;overflow:hidden;display:flex;border:1px solid var(--border);}
.gauge-low {width:29%;background:#F7CCCC;} .gauge-mid {width:42%;background:#FFE9AF;} .gauge-high {width:29%;background:#CFEEDF;}
.gauge-pointer {position:absolute;top:-5px;width:4px;height:28px;background:var(--deep);border:2px solid white;border-radius:3px;box-shadow:0 1px 5px rgba(11,37,69,.25);transition:left 220ms ease;}
.gauge-labels {display:flex;justify-content:space-between;color:var(--muted);font-size:11.5px;margin-top:5px;font-variant-numeric:tabular-nums;}
.stat-chips {display:flex;gap:7px;margin-top:12px;flex-wrap:wrap;}
.stat-chip {border:1px solid var(--border);border-radius:999px;padding:6px 9px;background:white;color:var(--navy);font-size:12.5px;}
.rule-state {border:1px solid #F0CD7A;background:#FFF8E7;border-radius:10px;padding:11px 12px;color:var(--navy);font-size:13.5px;}
.error-card {border:1px solid #ECC0C0;border-radius:14px;background:var(--red-bg);color:#7D2626;padding:14px 16px;}
.footer {border-top:1px solid var(--border);margin-top:24px;padding-top:13px;color:var(--muted);text-align:center;font-size:12.5px;}
.empty-state {border:1px dashed var(--border);border-radius:14px;background:var(--mist);text-align:center;padding:34px;color:var(--navy);margin:18px 0;}
.empty-orb {width:64px;height:64px;border-radius:50%;border:10px solid var(--light);border-top-color:var(--primary);margin:0 auto 12px;}
div[data-testid="stTabs"] button {font-size:14px;font-weight:650;}
div[data-testid="stDataFrame"] {border:1px solid var(--border);border-radius:14px;overflow:hidden;}
@keyframes fadein {from{opacity:.5;transform:translateY(3px)}to{opacity:1;transform:none}}
@media(max-width:900px){.header-banner{align-items:flex-start}.header-pills{display:none}.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.header-title{font-size:24px}}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def header():
    import streamlit as st
    logo = """
    <svg viewBox="0 0 56 56" aria-hidden="true">
      <circle cx="28" cy="28" r="21" fill="none" stroke="white" stroke-width="3"/>
      <path d="M20 15 C14 24 16 38 28 43 C40 38 42 24 36 15" fill="none" stroke="white" stroke-width="3"/>
      <path d="M22 28 L27 33 L36 23" fill="none" stroke="white" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """
    st.markdown(
        f"""
        <div class="header-banner">
          <div class="header-main">
            <div class="header-logo">{logo}</div>
            <div>
              <div class="header-title">Rectal Cancer after Chemoradiotherapy</div>
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


def stepper(decided_by):
    import streamlit as st
    active = {"rule": 0, "model": 1, "policy": 2}.get(decided_by)
    labels = ["1 Rules (no AI)", "2 Model", "3 Safety layer (abstain)"]
    parts = []
    for index, label in enumerate(labels):
        cls = "step active" if index == active else "step"
        parts.append(f'<div class="{cls}"><div class="step-title">{escape(label)}</div></div>')
    st.markdown('<div class="stepper">' + " ".join(
        parts[0:1] + ['<div class="step-arrow">→</div>'] + parts[1:2] +
        ['<div class="step-arrow">→</div>'] + parts[2:]
    ) + "</div>", unsafe_allow_html=True)


def decision_card(decision, probability, source, t_low, t_high):
    if decision == "COMPLETE RESPONSE":
        css, icon, meaning = "complete", "✓", "Eligible for watch-and-wait discussion"
    elif decision == "NOT COMPLETE":
        css, icon, meaning = "not_complete", "✕", "Continue the surgical / MDT pathway"
    else:
        css, icon, meaning = "uncertain", "!", "Refer to the multidisciplinary team"

    import streamlit as st
    st.markdown(
        f"""
        <div class="decision-card {css}">
          <div class="decision-icon">{icon}</div>
          <div class="decision-label {css}">{escape(decision)}</div>
          <div class="decision-meaning">{meaning}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge(probability, t_low, t_high):
    import streamlit as st
    if probability is None:
        st.markdown(
            '<div class="rule-state">No model probability: decided by a safety rule</div>',
            unsafe_allow_html=True,
        )
        return
    p = max(0.0, min(1.0, float(probability)))
    st.markdown(
        f"""
        <div class="gauge-wrap">
          <div class="gauge">
            <div class="gauge-low"></div><div class="gauge-mid"></div><div class="gauge-high"></div>
            <div class="gauge-pointer" style="left:calc({p:.6f} * 100% - 2px)"></div>
          </div>
          <div class="gauge-labels"><span>0.00</span><span>{t_low:.2f}</span><span>{t_high:.2f}</span><span>1.00</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def exam_chips(availability):
    import streamlit as st
    labels = {"dre":"DRE","endoscopy":"Endoscopy","biopsy":"Biopsy","mri":"MRI","ct":"CT"}
    chips = []
    for key in ("dre","endoscopy","biopsy","mri","ct"):
        ok = bool(availability.get(key, False))
        cls = "available" if ok else "missing"
        text = "available" if ok else "missing"
        chips.append(f'<span class="exam-chip {cls}">{escape(labels[key])}: {text}</span>')
    st.markdown('<div class="exam-strip">' + "".join(chips) + "</div>", unsafe_allow_html=True)


def kpi_card(label, value):
    import streamlit as st
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{escape(str(label))}</div>'
        f'<div class="kpi-value">{escape(str(value))}</div></div>',
        unsafe_allow_html=True,
    )


def footer():
    import streamlit as st
    st.markdown(
        '<div class="footer">Decision support only; the MDT decides.<br>'
        'Prototype: not a medical device; clinical validation on external real cases is required.</div>',
        unsafe_allow_html=True,
    )
