# =========================================================
# Medical Diagnosis Decision Support System
# Project: AIN7101 Master Project
# Author: Houssem Eddine Djebbi
# Purpose: Clinical decision support using Fuzzy Logic & ML
# =========================================================

from typing import Dict, List, Tuple
from io import BytesIO
import json
import time
import math
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from fuzzy_diagnosis import FuzzyDiagnosis, predict_label
from ml_decision_tree import train_decision_tree, FEATURES_NUM, FEATURES_CAT


# -------------------------
# Paths & Constants
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "synthetic_patients.csv"
ARCH_PATH = BASE_DIR / "system_architecture.png"
METRICS_PATH = BASE_DIR / "results" / "metrics.json"

RISK_THRESHOLDS = (0.33, 0.66)
RISK_WEIGHTS = {
    "fever": 0.10,
    "cough": 0.08,
    "sore_throat": 0.05,
    "breath_short": 0.15,
    "fatigue": 0.07,
    "wbc": 0.10,
    "crp": 0.15,
    "spo2": 0.20,
    "xray": 0.10,
}

SCENARIOS = [
    {
        "name": "Agreement Case",
        "desc": "Both models agree on the diagnosis (classic flu case).",
        "input": dict(fever=8, cough=7, sore_throat=6, breath_short=2, fatigue=5, wbc=7, crp=10, spo2=98, xray=0),
    },
    {
        "name": "Disagreement Case",
        "desc": "Models disagree due to ambiguous symptoms (borderline pneumonia/allergy).",
        "input": dict(fever=4, cough=5, sore_throat=7, breath_short=6, fatigue=6, wbc=10, crp=40, spo2=93, xray=1),
    },
    {
        "name": "High Uncertainty Case",
        "desc": "Both models show low confidence (mild, mixed symptoms).",
        "input": dict(fever=2, cough=2, sore_throat=2, breath_short=2, fatigue=2, wbc=5, crp=5, spo2=99, xray=0),
    },
]

DISEASE_ICONS: Dict[str, str] = {
    "flu": "(F)",
    "pneumonia": "(P)",
    "allergy": "(A)",
    "bronchitis": "(B)",
}


# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Medical Diagnosis Support", layout="wide")


# -------------------------
# Global Styles
# -------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Manrope:wght@400;500;600;700&display=swap');

    :root {
        --bg-a: #f6f7fb;
        --bg-b: #eef2f7;
        --ink: #0f172a;
        --muted: #6b7280;
        --accent: #0ea5a4;
        --accent-2: #1d4ed8;
        --card: rgba(255, 255, 255, 0.92);
        --border: rgba(15, 23, 42, 0.12);
        --shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
    }

    .stApp {
        font-family: "Manrope", "Segoe UI", sans-serif;
        color: var(--ink);
        background:
            radial-gradient(900px 520px at 0% 0%, rgba(14,165,164,0.10) 0%, transparent 60%),
            radial-gradient(900px 520px at 100% 0%, rgba(29,78,216,0.10) 0%, transparent 60%),
            linear-gradient(135deg, var(--bg-a), var(--bg-b));
    }

    h1, h2, h3, h4 {
        font-family: "Source Serif 4", "Georgia", serif;
        color: var(--ink);
        letter-spacing: 0.2px;
    }

    .shell {
        background: rgba(255,255,255,0.65);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 20px 24px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(8px);
    }

    .hero {
        display: grid;
        gap: 8px;
        padding: 18px 20px;
        border-radius: 20px;
        background: linear-gradient(120deg, rgba(14,165,164,0.12), rgba(29,78,216,0.12));
        border: 1px solid var(--border);
    }

    .hero-title { font-size: 2.1rem; margin: 0; }
    .hero-sub { font-size: 1.0rem; color: var(--muted); margin: 0; }

    .tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
        background: #0ea5a4;
        margin-right: 6px;
    }
    .tag.alt { background: #1d4ed8; }
    .tag.dark { background: #111827; }

    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: var(--shadow);
        padding: 14px 16px;
    }

    .card-title { margin: 0 0 6px 0; font-size: 1.05rem; }
    .muted { color: var(--muted); }

    .stat {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 10px;
        border-radius: 12px;
        background: rgba(14,165,164,0.08);
        border: 1px solid rgba(14,165,164,0.2);
        margin-bottom: 6px;
    }

    .stDataFrame, .stPlotlyChart, .stAltairChart, .stBarChart {
        background: var(--card);
        border-radius: 14px;
        box-shadow: var(--shadow);
        padding: 12px;
        border: 1px solid var(--border);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0ea5a4, #22c1c3, #1d4ed8);
        color: #0b1324;
        border-radius: 999px;
        font-weight: 700;
        border: none;
    }

    @media (max-width: 900px) {
        .hero-title { font-size: 1.6rem; }
        .shell { padding: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------
# UI Helpers
# -------------------------
def show_dataframe(df: pd.DataFrame, **kwargs) -> None:
    kwargs.pop("use_container_width", None)
    try:
        st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        st.dataframe(df, **kwargs)


def show_plotly(fig: go.Figure, **kwargs) -> None:
    kwargs.pop("use_container_width", None)
    try:
        st.plotly_chart(fig, use_container_width=True, **kwargs)
    except TypeError:
        st.plotly_chart(fig, **kwargs)


def show_image(path: str, **kwargs) -> None:
    kwargs.pop("use_container_width", None)
    kwargs.pop("use_column_width", None)
    try:
        st.image(path, use_container_width=True, **kwargs)
    except TypeError:
        st.image(path, use_column_width=True, **kwargs)


def build_score_table(disease_scores: Dict[str, float]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            [
                {"Disease": f"{DISEASE_ICONS.get(d, '')} {d.capitalize()}", "Score": round(v, 3)}
                for d, v in disease_scores.items()
            ]
        )
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )


# -------------------------
# Risk & Metrics Helpers
# -------------------------
def explain_difference(fuzzy_pred: str, tree_pred: str) -> str:
    if fuzzy_pred == tree_pred:
        return (
            "Both approaches reached the same diagnosis. "
            "This agreement increases confidence in the result."
        )
    return (
        "The two approaches produced different diagnoses because they rely on "
        "different reasoning mechanisms. Fuzzy Logic uses expert-defined rules "
        "and handles uncertainty explicitly, while the Decision Tree relies on "
        "patterns learned from data."
    )


def diagnostic_probability(
    fever: float,
    cough: float,
    sore_throat: float,
    shortness_breath: float,
    fatigue: float,
    wbc: float,
    crp: float,
    spo2: float,
    xray: int,
) -> Tuple[float, float]:
    F = fever / 10
    C = cough / 10
    ST = sore_throat / 10
    SB = shortness_breath / 10
    FA = fatigue / 10
    W = (wbc - 3) / 12
    CR = crp / 120
    S = (100 - spo2) / 15
    X = xray

    R = (
        0.10 * F
        + 0.08 * C
        + 0.05 * ST
        + 0.15 * SB
        + 0.07 * FA
        + 0.10 * W
        + 0.15 * CR
        + 0.20 * S
        + 0.10 * X
    )

    P = 1 / (1 + math.exp(-12 * (R - 0.5)))
    return round(P, 3), round(R, 3)


def risk_level(prob: float) -> str:
    if prob < RISK_THRESHOLDS[0]:
        return "Low"
    if prob < RISK_THRESHOLDS[1]:
        return "Moderate"
    return "High"


def build_risk_contributions(patient_data: Dict[str, float]) -> pd.DataFrame:
    fever = patient_data["fever"] / 10
    cough = patient_data["cough"] / 10
    sore_throat = patient_data["sore_throat"] / 10
    breath_short = patient_data["breath_short"] / 10
    fatigue = patient_data["fatigue"] / 10
    wbc = (patient_data["wbc"] - 3) / 12
    crp = patient_data["crp"] / 120
    spo2 = (100 - patient_data["spo2"]) / 15
    xray = patient_data["xray_infiltrate"]

    values = {
        "fever": fever,
        "cough": cough,
        "sore_throat": sore_throat,
        "breath_short": breath_short,
        "fatigue": fatigue,
        "wbc": wbc,
        "crp": crp,
        "spo2": spo2,
        "xray": xray,
    }
    rows = []
    for key, val in values.items():
        contrib = val * RISK_WEIGHTS[key]
        rows.append(
            {
                "Feature": key.replace("_", " ").title(),
                "Normalized": round(float(val), 3),
                "Weight": RISK_WEIGHTS[key],
                "Contribution": round(float(contrib), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("Contribution", ascending=False)


def build_norm_values(patient_data: Dict[str, float]) -> Dict[str, float]:
    return {
        "fever": patient_data["fever"] / 10,
        "cough": patient_data["cough"] / 10,
        "sore_throat": patient_data["sore_throat"] / 10,
        "breath_short": patient_data["breath_short"] / 10,
        "fatigue": patient_data["fatigue"] / 10,
        "wbc": (patient_data["wbc"] - 3) / 12,
        "crp": patient_data["crp"] / 120,
        "spo2": (100 - patient_data["spo2"]) / 15,
        "xray": patient_data["xray_infiltrate"],
    }


def interpret_model_performance(eff: Dict) -> str:
    models = list(eff.keys())
    acc = [eff[m]["accuracy"] for m in models]
    f1 = [eff[m]["f1_macro"] for m in models]
    best = models[f1.index(max(f1))]
    worst = models[f1.index(min(f1))]
    if acc[0] == acc[1] and f1[0] == f1[1]:
        return "Both models perform equally well on this dataset."
    return (
        f"The {best.replace('_', ' ').title()} model outperforms the "
        f"{worst.replace('_', ' ').title()} model with higher F1-score "
        f"({max(f1):.2f} vs {min(f1):.2f}) and accuracy ({max(acc):.2f} vs {min(acc):.2f})."
    )


# -------------------------
# Plot Helpers
# -------------------------
def plot_radar(values: Dict[str, float], title: str) -> go.Figure:
    labels = list(values.keys())
    data = list(values.values())
    if labels:
        labels.append(labels[0])
        data.append(data[0])
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=data,
                theta=labels,
                fill="toself",
                marker=dict(color="#0ea5a4"),
            )
        ]
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
    )
    return fig


def plot_stacked_scores(fz_scores: Dict[str, float], ml_scores: Dict[str, float]) -> go.Figure:
    classes = sorted(set(fz_scores.keys()) | set(ml_scores.keys()))
    fz_vals = [fz_scores.get(c, 0.0) for c in classes]
    ml_vals = [ml_scores.get(c, 0.0) for c in classes]
    fig = go.Figure(
        data=[
            go.Bar(name="Fuzzy Logic", x=classes, y=fz_vals, marker_color="#0ea5a4"),
            go.Bar(name="Decision Tree", x=classes, y=ml_vals, marker_color="#1d4ed8"),
        ]
    )
    fig.update_layout(barmode="stack", title="Stacked Class Scores")
    return fig


def plot_gauge(prob: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#0ea5a4"},
                "steps": [
                    {"range": [0, 0.33], "color": "#e5e7eb"},
                    {"range": [0.33, 0.66], "color": "#fde68a"},
                    {"range": [0.66, 1], "color": "#fecaca"},
                ],
            },
            title={"text": title},
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# -------------------------
# PDF Report
# -------------------------
def build_pdf_report(
    patient_data: Dict[str, float],
    fz_pred: str,
    ml_pred: str,
    fz_scores: Dict[str, float],
    ml_scores: Dict[str, float],
    risk_prob: float,
    risk_score: float,
    fired_rules: list,
) -> bytes:
    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    lines = [
        "Medical Diagnosis Support - Current Patient Report",
        "",
        f"Fuzzy Logic Prediction: {fz_pred}",
        f"Decision Tree Prediction: {ml_pred}",
        f"Risk Probability: {risk_prob:.2f} ({risk_level(risk_prob)})",
        f"Risk Score: {risk_score:.2f}",
        "",
        "Patient Inputs:",
    ]
    for key, val in patient_data.items():
        lines.append(f"  - {key.replace('_', ' ')}: {val}")

    lines.append("")
    lines.append("Top Fuzzy Rules:")
    for rule, strength in fired_rules[:5]:
        lines.append(f"  - {rule} -> {strength:.2f}")

    lines.append("")
    lines.append("Fuzzy Logic Scores:")
    for k, v in sorted(fz_scores.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  - {k}: {v:.3f}")

    lines.append("")
    lines.append("Decision Tree Probabilities:")
    for k, v in sorted(ml_scores.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  - {k}: {v:.3f}")

    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, va="top", family="monospace", fontsize=10)
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# -------------------------
# Data Loading
# -------------------------
try:
    cache_resource = st.cache_resource
except AttributeError:
    cache_resource = st.cache


@cache_resource
def load_and_train_models() -> Tuple[FuzzyDiagnosis, object]:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    try:
        df = pd.read_csv(DATA_PATH)
        fuzzy_model = FuzzyDiagnosis()
        ml_model = train_decision_tree(df)
        return fuzzy_model, ml_model
    except Exception as exc:
        st.error(f"Error loading models: {exc}")
        st.stop()


def load_metrics() -> Dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Could not load metrics from {METRICS_PATH}: {exc}")
        return {}


# -------------------------
# Inference
# -------------------------
def run_inference(
    fuzzy: FuzzyDiagnosis,
    ml_model: object,
    patient_data: Dict[str, float],
) -> Tuple[str, Dict[str, float], float, str, Dict[str, float], float, list]:
    t0 = time.perf_counter()
    fz_res = fuzzy.infer(patient_data)
    fz_pred = predict_label(fz_res.scores)
    fz_time = time.perf_counter() - t0

    x = pd.DataFrame([patient_data])[FEATURES_NUM + FEATURES_CAT]
    t1 = time.perf_counter()
    ml_pred = ml_model.model.predict(x)[0]
    ml_probs = ml_model.model.predict_proba(x)[0]
    classes = ml_model.model.named_steps["clf"].classes_
    ml_scores = {c: float(p) for c, p in zip(classes, ml_probs)}
    ml_time = time.perf_counter() - t1

    return fz_pred, fz_res.scores, fz_time, ml_pred, ml_scores, ml_time, fz_res.fired_rules


# -------------------------
# Input & Validation
# -------------------------
def validate_inputs(values: Dict[str, float]) -> List[str]:
    errors: List[str] = []
    if not (0 <= values["fever"] <= 10):
        errors.append("Fever must be between 0 and 10.")
    if not (0 <= values["cough"] <= 10):
        errors.append("Cough must be between 0 and 10.")
    if not (0 <= values["sore_throat"] <= 10):
        errors.append("Sore throat must be between 0 and 10.")
    if not (0 <= values["breath_short"] <= 10):
        errors.append("Shortness of breath must be between 0 and 10.")
    if not (0 <= values["fatigue"] <= 10):
        errors.append("Fatigue must be between 0 and 10.")
    if not (3 <= values["wbc"] <= 15):
        errors.append("WBC must be between 3 and 15.")
    if not (0 <= values["crp"] <= 120):
        errors.append("CRP must be between 0 and 120.")
    if not (85 <= values["spo2"] <= 100):
        errors.append("SpO2 must be between 85 and 100.")
    if values["xray_infiltrate"] not in [0, 1]:
        errors.append("X-ray infiltrate must be 0 or 1.")
    return errors


def collect_patient_inputs() -> Tuple[Dict[str, float], bool]:
    st.markdown("### Patient Inputs")
    with st.form("patient_form"):
        st.markdown("#### Symptoms")
        fever = st.slider("Fever", 0.0, 10.0, 5.0)
        cough = st.slider("Cough", 0.0, 10.0, 5.0)
        sore_throat = st.slider("Sore throat", 0.0, 10.0, 4.0)
        breath_short = st.slider("Shortness of breath", 0.0, 10.0, 3.0)
        fatigue = st.slider("Fatigue", 0.0, 10.0, 5.0)

        st.markdown("#### Clinical Tests")
        wbc = st.slider("WBC", 3.0, 15.0, 7.5)
        crp = st.slider("CRP", 0.0, 120.0, 18.0)
        spo2 = st.slider("SpO2", 85.0, 100.0, 96.0)
        xray = st.selectbox("X-ray infiltrate", [0, 1])

        submitted = st.form_submit_button("Run Diagnosis")

    values = {
        "fever": fever,
        "cough": cough,
        "sore_throat": sore_throat,
        "breath_short": breath_short,
        "fatigue": fatigue,
        "wbc": wbc,
        "crp": crp,
        "spo2": spo2,
        "xray_infiltrate": xray,
    }
    return values, submitted


# -------------------------
# Layout Start
# -------------------------
st.markdown("<div class='shell'>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='hero'>
        <div>
            <span class='tag'>AIN7101 Master Project</span>
            <span class='tag alt'>Decision Support</span>
            <span class='tag dark'>Fuzzy + ML</span>
        </div>
        <h1 class='hero-title'>Professional Clinical Dashboard</h1>
        <p class='hero-sub'>Organized layout with clear separation of inputs, summary, and analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

if "submitted" not in st.session_state:
    st.session_state.submitted = False


# -------------------------
# Inputs + Summary
# -------------------------
left, mid, right = st.columns([1.0, 1.2, 1.0])

with left:
    patient, submitted = collect_patient_inputs()
    if submitted:
        st.session_state.submitted = True
    st.markdown(
        """
        <div class='card'>
            <h3 class='card-title'>Input Guidance</h3>
            <p class='muted'>Adjust sliders then run diagnosis to refresh results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mid:
    st.markdown("### System Overview")
    if ARCH_PATH.exists():
        show_image(str(ARCH_PATH))
    st.markdown(
        """
        <div class='card'>
            <h3 class='card-title'>Workflow</h3>
            <p class='muted'>Inputs -> Fuzzy inference + Decision tree -> Risk scoring -> Comparison</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("### Status")
    st.markdown(
        """
        <div class='card'>
            <h3 class='card-title'>Model Readiness</h3>
            <p class='muted'>System ready to analyze patient data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Validation
validation_errors = validate_inputs(patient)
if validation_errors:
    st.error("\n".join(validation_errors))
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

if not st.session_state.submitted:
    st.info("Fill the patient inputs and click Run Diagnosis to generate results.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# -------------------------
# Load Data and Run
# -------------------------
fuzzy, ml = load_and_train_models()
metrics = load_metrics()

fz_pred, fz_scores, fz_time, ml_pred, ml_scores, ml_time, fired_rules = run_inference(
    fuzzy, ml, patient
)

risk_prob, risk_score = diagnostic_probability(
    fever=patient["fever"],
    cough=patient["cough"],
    sore_throat=patient["sore_throat"],
    shortness_breath=patient["breath_short"],
    fatigue=patient["fatigue"],
    wbc=patient["wbc"],
    crp=patient["crp"],
    spo2=patient["spo2"],
    xray=patient["xray_infiltrate"],
)


# -------------------------
# Executive Summary
# -------------------------
st.markdown("### Executive Summary")
summary_a, summary_b, summary_c = st.columns(3)

with summary_a:
    st.markdown(
        f"""
        <div class='card'>
            <h3 class='card-title'>Fuzzy Logic</h3>
            <div class='stat'><span>Diagnosis</span><strong>{fz_pred.upper()}</strong></div>
            <div class='stat'><span>Runtime</span><strong>{fz_time*1000:.2f} ms</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_b:
    st.markdown(
        f"""
        <div class='card'>
            <h3 class='card-title'>Decision Tree</h3>
            <div class='stat'><span>Diagnosis</span><strong>{ml_pred.upper()}</strong></div>
            <div class='stat'><span>Runtime</span><strong>{ml_time*1000:.2f} ms</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_c:
    st.markdown(
        """
        <div class='card'>
            <h3 class='card-title'>Confidence</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if fz_pred == ml_pred:
        st.success(f"Both models agree on {fz_pred.upper()}.")
    else:
        st.warning("Models disagree. Review both outputs.")
    st.info("Decision support only. Not a medical diagnosis.")


# -------------------------
# Analysis Workspace
# -------------------------
analysis_tabs = st.tabs(["Model Scores", "Risk Analysis", "Explainability"])

with analysis_tabs[0]:
    left_scores, right_scores = st.columns(2)
    with left_scores:
        st.markdown("### Fuzzy Logic Scores")
        show_dataframe(build_score_table(fz_scores), use_container_width=True)
        with st.expander("Top fired rules"):
            for rule, strength in fired_rules[:6]:
                st.write(f"- {rule} -> {strength:.2f}")

    with right_scores:
        st.markdown("### Decision Tree Probabilities")
        show_dataframe(build_score_table(ml_scores), use_container_width=True)

    st.markdown("### Stacked Class Scores")
    show_plotly(plot_stacked_scores(fz_scores, ml_scores), use_container_width=True)

with analysis_tabs[1]:
    st.markdown("### Risk Assessment")
    st.write(
        f"**Risk probability:** {risk_prob:.2f} ({risk_level(risk_prob)}) | "
        f"**Risk score:** {risk_score:.2f}"
    )
    st.caption("Risk thresholds: < 0.33 = Low, 0.33-0.66 = Moderate, > 0.66 = High.")

    r_left, r_right = st.columns([1, 1.2])
    with r_left:
        show_plotly(plot_gauge(risk_prob, "Risk Probability"), use_container_width=True)
        norm_values = build_norm_values(patient)
        show_plotly(plot_radar(norm_values, "Normalized Patient Profile"), use_container_width=True)

    with r_right:
        st.markdown("### Feature Contributions")
        contrib_df = build_risk_contributions(patient)
        show_dataframe(contrib_df, use_container_width=True)

with analysis_tabs[2]:
    st.markdown("### Interpretation")
    st.write(
        f"""
- Fuzzy Logic result: {fz_pred} based on expert rules.
- Decision Tree result: {ml_pred} based on learned patterns.

Why the results may differ:
{explain_difference(fz_pred, ml_pred)}
"""
    )

    st.markdown("### How the Diagnosis Is Calculated (Simple Explanation)")

    st.markdown(
        """
**Fuzzy Logic (rule-based, explainable)**
- **Step 1: Convert symptoms to degrees.** Each symptom becomes a value like low/medium/high.
- **Step 2: Apply medical rules.** Example: “high cough + low oxygen -> pneumonia.”
- **Step 3: Score each disease.** The strongest rules give the highest score.
- **Final diagnosis:** the disease with the highest fuzzy score.

**Decision Tree (data-driven, predictive)**
- **Step 1: Learn rules from data.** The model learns thresholds (if/else) from past cases.
- **Step 2: Predict probabilities.** For a new patient, it outputs a probability for each disease.
- **Final diagnosis:** the disease with the highest probability.

**Risk Score (severity estimate)**
- **Step 1: Normalize inputs** to a 0–1 range (e.g., fever/10).
- **Step 2: Weighted sum** combines symptoms and test results.
- **Step 3: Convert to probability** using a logistic curve.
- **Risk level:** Low / Moderate / High based on thresholds.
"""
    )

    with st.expander("Why use both methods?"):
        st.markdown(
            """
- **Fuzzy Logic** is easy to explain and matches how doctors think in degrees.
- **Decision Tree** is fast and learns patterns automatically from data.
- **Using both** gives transparency and predictive power.
"""
        )


# -------------------------
# Supporting Sections
# -------------------------
bottom_tabs = st.tabs(["Validation", "Metrics", "Downloads"])

with bottom_tabs[0]:
    st.markdown("### Model Validation: Predefined Patient Scenarios")
    for scenario in SCENARIOS:
        st.markdown(f"**{scenario['name']}** - {scenario['desc']}")
        scenario_patient = scenario["input"].copy()
        if "xray" in scenario_patient:
            scenario_patient["xray_infiltrate"] = scenario_patient.pop("xray")
        scenario_fz_pred, _, _, scenario_ml_pred, _, _, _ = run_inference(
            fuzzy, ml, scenario_patient
        )
        st.write(f"Fuzzy Logic: **{scenario_fz_pred.upper()}** | Decision Tree: **{scenario_ml_pred.upper()}**")
        if scenario_fz_pred == scenario_ml_pred:
            st.success("Agreement: Both models suggest the same diagnosis.")
        else:
            st.warning("Disagreement: Models suggest different diagnoses. Consider clinical review.")
        st.markdown("---")

    st.markdown("### Limitations")
    st.markdown(
        """
- This system uses synthetic patient data and simplified medical rules.
- It is intended for demonstration and educational purposes only.
- Real-world deployment requires extensive clinical validation and regulatory review.
- Always consult a qualified healthcare professional for actual medical decisions.
"""
    )

    st.markdown("### Future Work")
    st.markdown(
        """
- Integrate real clinical datasets to improve realism and generalizability.
- Explore hybrid approaches that combine fuzzy logic and machine learning.
- Build a robust UI for deployment as a clinical decision support tool.
- Conduct prospective clinical validation and seek regulatory approval.
"""
    )

with bottom_tabs[1]:
    st.markdown("### Performance Metrics")

    if metrics and "effectiveness" in metrics:
        st.markdown(
            f"""
        <div class='card'>
            <strong>Interpretation:</strong> {interpret_model_performance(metrics['effectiveness'])}
        </div>
        """,
            unsafe_allow_html=True,
        )

        eff = metrics["effectiveness"]
        best_model = max(eff, key=lambda m: eff[m]["f1_macro"])
        st.markdown(
            f"""
        <div style='margin-top:10px;'>
            <span class='tag'>Best F1-score: {best_model.replace('_',' ').title()} ({eff[best_model]['f1_macro']:.2f})</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        models = list(eff.keys())
        accuracy = [eff[m]["accuracy"] for m in models]
        f1_macro = [eff[m]["f1_macro"] for m in models]
        colors = ["#0ea5a4", "#1d4ed8"]

        acc_fig = go.Figure(
            data=[
                go.Bar(
                    x=models,
                    y=accuracy,
                    marker_color=colors,
                    text=[f"{v:.2f}" for v in accuracy],
                    textposition="auto",
                )
            ],
            layout=go.Layout(title="Model Accuracy Comparison", yaxis_title="Accuracy", xaxis_title="Model", bargap=0.5),
        )
        f1_fig = go.Figure(
            data=[
                go.Bar(
                    x=models,
                    y=f1_macro,
                    marker_color=colors,
                    text=[f"{v:.2f}" for v in f1_macro],
                    textposition="auto",
                )
            ],
            layout=go.Layout(
                title="Model F1-score (Macro) Comparison",
                yaxis_title="F1-score (Macro)",
                xaxis_title="Model",
                bargap=0.5,
            ),
        )
        col1, col2 = st.columns(2)
        with col1:
            show_plotly(acc_fig, use_container_width=True)
        with col2:
            show_plotly(f1_fig, use_container_width=True)

    if metrics and "cross_validation" in metrics:
        cv = metrics["cross_validation"]
        st.subheader("Cross-Validation Summary")
        cv_rows = []
        for key, label in [("fuzzy_logic", "Fuzzy Logic"), ("decision_tree", "Decision Tree")]:
            if key not in cv:
                continue
            model_metrics = cv[key].get("metrics", {})
            if not model_metrics:
                continue
            cv_rows.append(
                {
                    "Model": label,
                    "Accuracy (mean+/-std)": f"{model_metrics['accuracy']['mean']:.3f} +/- {model_metrics['accuracy']['std']:.3f}",
                    "Precision (mean+/-std)": f"{model_metrics['precision_macro']['mean']:.3f} +/- {model_metrics['precision_macro']['std']:.3f}",
                    "Recall (mean+/-std)": f"{model_metrics['recall_macro']['mean']:.3f} +/- {model_metrics['recall_macro']['std']:.3f}",
                    "F1 Macro (mean+/-std)": f"{model_metrics['f1_macro']['mean']:.3f} +/- {model_metrics['f1_macro']['std']:.3f}",
                }
            )
        if cv_rows:
            show_dataframe(pd.DataFrame(cv_rows), use_container_width=True)

        per_class_rows = []
        for key, label in [("fuzzy_logic", "Fuzzy Logic"), ("decision_tree", "Decision Tree")]:
            per_class = cv.get(key, {}).get("per_class_f1", {})
            for cls, stats in per_class.items():
                per_class_rows.append(
                    {"Model": label, "Class": cls, "F1 (mean+/-std)": f"{stats['mean']:.3f} +/- {stats['std']:.3f}"}
                )
        if per_class_rows:
            st.caption("Per-class F1 across folds")
            show_dataframe(pd.DataFrame(per_class_rows), use_container_width=True)

with bottom_tabs[2]:
    st.markdown("### Exports")
    patient_scores = pd.DataFrame(
        [
            {"Model": "Fuzzy Logic", "Diagnosis": fz_pred, "Confidence": round(fz_scores.get(fz_pred, 0.0), 3)},
            {"Model": "Decision Tree", "Diagnosis": ml_pred, "Confidence": round(ml_scores.get(ml_pred, 0.0), 3)},
        ]
    )
    contrib_df = build_risk_contributions(patient)

    patient_csv = patient_scores.to_csv(index=False).encode("utf-8")
    contrib_csv = contrib_df.to_csv(index=False).encode("utf-8")
    report_pdf = build_pdf_report(
        patient,
        fz_pred,
        ml_pred,
        fz_scores,
        ml_scores,
        risk_prob,
        risk_score,
        fired_rules,
    )

    st.download_button(
        label="Download Patient Summary (CSV)",
        data=patient_csv,
        file_name="patient_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download Risk Contributions (CSV)",
        data=contrib_csv,
        file_name="risk_contributions.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download Patient Report (PDF)",
        data=report_pdf,
        file_name="patient_report.pdf",
        mime="application/pdf",
    )

st.markdown("</div>", unsafe_allow_html=True)
