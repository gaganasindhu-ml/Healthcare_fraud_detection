import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Insurance Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');

    /* Dark background */
    .stApp { background-color: #050a0f; }
    section[data-testid="stSidebar"] { background-color: #0d1821; border-right: 1px solid #00c89620; }

    /* Typography */
    html, body, [class*="css"] { color: #e8f4f0; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -1px; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #0d1821;
        border: 1px solid #00c89620;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="metric-container"] label { color: #6b8fa0 !important; font-size: 12px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00c896 !important; font-size: 28px !important; font-weight: 800 !important; }

    /* Inputs */
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background: #132030 !important;
        border: 1px solid #00c89630 !important;
        color: #e8f4f0 !important;
        border-radius: 8px !important;
    }
    .stSlider > div > div > div { background: #00c896 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00c896, #00a878) !important;
        color: #000 !important;
        font-weight: 800 !important;
        font-size: 16px !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 28px !important;
        width: 100% !important;
        letter-spacing: 0.5px;
        transition: all 0.2s;
    }
    .stButton > button:hover { box-shadow: 0 8px 30px #00c89650 !important; transform: translateY(-1px); }

    /* Section headers */
    .section-tag {
        font-size: 11px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #00c896;
        margin: 20px 0 10px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Verdict boxes */
    .verdict-fraud {
        background: #ff4d6d12;
        border: 1px solid #ff4d6d80;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 40px #ff4d6d20;
    }
    .verdict-warning {
        background: #ffd16612;
        border: 1px solid #ffd16680;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 40px #ffd16620;
    }
    .verdict-safe {
        background: #00c89612;
        border: 1px solid #00c89680;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 40px #00c89620;
    }
    .verdict-title { font-family: 'Syne', sans-serif; font-size: 32px; font-weight: 800; letter-spacing: -1px; margin: 8px 0; }
    .verdict-sub { color: #6b8fa0; font-size: 13px; line-height: 1.6; }

    /* Info card */
    .info-card {
        background: #0d1821;
        border: 1px solid #00c89620;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }

    /* Factor items */
    .factor-high { color: #ff4d6d; }
    .factor-med  { color: #ffd166; }
    .factor-low  { color: #00c896; }

    /* Divider */
    hr { border-color: #00c89615 !important; }

    /* Checkbox */
    .stCheckbox label { color: #a0bcc8 !important; font-size: 13px !important; }

    /* Sidebar labels */
    .stSlider label, .stSelectbox label, .stNumberInput label { color: #6b8fa0 !important; font-size: 12px !important; }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("fraud_detection_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def get_race_encoding(race_val):
    return [
        1 if race_val == "Group 1" else 0,
        1 if race_val == "Group 2" else 0,
        1 if race_val == "Group 3" else 0,
        1 if race_val == "Group 5" else 0,
    ]

def predict(features):
    arr = np.array([features])
    prob = model.predict_proba(arr)[0][1]
    pred = model.predict(arr)[0]
    return float(prob), int(pred)

def get_risk_factors(features_dict, prob):
    factors = []
    if features_dict["fraud_ratio"] > 0.3:
        factors.append(("🔴", f"Provider fraud history: {features_dict['fraud_ratio']*100:.0f}% of claims flagged", "high"))
    if features_dict["claim_amt"] > 10000:
        factors.append(("🔴", f"High reimbursement: ${features_dict['claim_amt']:,.0f}", "high"))
    if features_dict["hosp_stay"] > 25:
        factors.append(("🟡", f"Extended hospital stay: {features_dict['hosp_stay']} days", "med"))
    if features_dict["total_claims"] > 200 and features_dict["unique_docs"] < 5:
        factors.append(("🔴", f"High claim volume with few doctors ({features_dict['unique_docs']})", "high"))
    if features_dict["num_diag"] > 8:
        factors.append(("🟡", f"Excessive diagnosis codes: {features_dict['num_diag']}", "med"))
    if features_dict["claim_amt"] > 8000 and features_dict["claim_dur"] < 5:
        factors.append(("🟡", f"High claim for short duration ({features_dict['claim_dur']} days)", "med"))
    if features_dict["avg_claim"] > 7000:
        factors.append(("🟡", f"Provider avg claim high: ${features_dict['avg_claim']:,.0f}", "med"))
    if not factors:
        factors.append(("🟢", "No significant risk indicators detected", "low"))
        factors.append(("🟢", f"Provider fraud ratio within normal range: {features_dict['fraud_ratio']*100:.1f}%", "low"))
    return factors[:5]

def make_gauge(prob):
    color = "#ff4d6d" if prob >= 0.6 else "#ffd166" if prob >= 0.35 else "#00c896"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 42, "color": color, "family": "Syne"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6b8fa0",
                     "tickfont": {"color": "#6b8fa0", "size": 11}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#132030",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "#00c89615"},
                {"range": [35, 60], "color": "#ffd16615"},
                {"range": [60, 100], "color": "#ff4d6d15"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": prob * 100
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="#0d1821",
        plot_bgcolor="#0d1821",
        font={"color": "#e8f4f0"},
    )
    return fig

def make_feature_importance_chart(features_dict):
    labels = ["Claim Amount", "Fraud Ratio", "Hosp. Stay", "Claim Duration",
              "Avg Provider Claim", "Diag Codes", "Proc Codes", "Chronic Conds"]
    values = [
        min(features_dict["claim_amt"] / 10000, 1.0),
        features_dict["fraud_ratio"],
        min(features_dict["hosp_stay"] / 30, 1.0),
        min(features_dict["claim_dur"] / 60, 1.0),
        min(features_dict["avg_claim"] / 10000, 1.0),
        min(features_dict["num_diag"] / 10, 1.0),
        min(features_dict["num_proc"] / 10, 1.0),
        min(features_dict["chronic_count"] / 9, 1.0),
    ]
    colors = ["#ff4d6d" if v > 0.6 else "#ffd166" if v > 0.3 else "#00c896" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v*100:.0f}%" for v in values],
        textposition="outside",
        textfont={"color": "#e8f4f0", "size": 11},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="#0d1821",
        plot_bgcolor="#0d1821",
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False,
               "range": [0, 1.3]},
        yaxis={"color": "#6b8fa0", "tickfont": {"size": 12}},
        bargap=0.3,
    )
    return fig


# ─── HEADER ────────────────────────────────────────────────────────────────────
col_logo, col_status = st.columns([3, 1])
with col_logo:
    st.markdown("""
    <h1 style='font-size:42px; margin-bottom:0; color:#e8f4f0;'>
        🛡️ Fraud<span style='color:#00c896'>Shield</span>
    </h1>
    <p style='color:#6b8fa0; font-size:13px; margin-top:4px; font-family: monospace;'>
        Insurance Fraud Detection · XGBoost Classifier · 17 Features
    </p>
    """, unsafe_allow_html=True)

with col_status:
    if model_loaded:
        st.success("✅ Model Loaded")
    else:
        st.error("❌ Model not found")
        st.caption(f"Place `fraud_detection_model.pkl` in the same folder as `app.py`")

st.divider()

# ─── TOP STATS ────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model Type", "XGBoost")
m2.metric("Features", "17")
m3.metric("Accuracy", "94.2%")
m4.metric("Task", "Binary Classification")

st.markdown("<br>", unsafe_allow_html=True)

# ─── SIDEBAR INPUTS ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <h3 style='color:#00c896; font-family:Syne; font-size:18px; margin-bottom:4px;'>
        📋 Claim Input
    </h3>
    <p style='color:#6b8fa0; font-size:11px; letter-spacing:1px;'>FILL IN CLAIM DETAILS BELOW</p>
    """, unsafe_allow_html=True)
    st.divider()

    # Claim Details
    st.markdown("<div class='section-tag'>── Claim Details</div>", unsafe_allow_html=True)
    claim_amt = st.number_input("Claim Amount Reimbursed ($)", min_value=0, max_value=200000, value=5000, step=100)
    claim_type = st.selectbox("Claim Type", ["Inpatient", "Outpatient"])
    claim_dur = st.slider("Claim Duration (days)", 0, 90, 3)
    hosp_stay = st.slider("Hospital Stay (days)", 0, 120, 5)

    # Patient Info
    st.divider()
    st.markdown("<div class='section-tag'>── Patient Info</div>", unsafe_allow_html=True)
    age = st.slider("Patient Age", 18, 100, 65)
    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race Group", ["Group 1", "Group 2", "Group 3", "Group 5"])
    num_diag = st.slider("Number of Diagnosis Codes", 0, 20, 4)
    num_proc = st.slider("Number of Procedure Codes", 0, 20, 2)

    # Chronic Conditions
    st.divider()
    st.markdown("<div class='section-tag'>── Chronic Conditions</div>", unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)
    conditions = []
    with col_c1:
        if st.checkbox("Heart Failure"): conditions.append(1)
        if st.checkbox("Alzheimer's"):   conditions.append(1)
        if st.checkbox("Cancer"):        conditions.append(1)
        if st.checkbox("COPD"):          conditions.append(1)
        if st.checkbox("Diabetes"):      conditions.append(1)
    with col_c2:
        if st.checkbox("Depression"):    conditions.append(1)
        if st.checkbox("Ischemic Heart"):conditions.append(1)
        if st.checkbox("Osteoporosis"):  conditions.append(1)
        if st.checkbox("Kidney Disease"):conditions.append(1)
    chronic_count = len(conditions)

    # Provider Info
    st.divider()
    st.markdown("<div class='section-tag'>── Provider Info</div>", unsafe_allow_html=True)
    total_claims = st.number_input("Total Provider Claims", min_value=1, max_value=10000, value=20)
    unique_docs = st.number_input("Unique Doctors Used", min_value=1, max_value=500, value=3)
    avg_claim = st.number_input("Provider Avg Claim ($)", min_value=0, max_value=200000, value=4500, step=100)
    fraud_ratio = st.slider("Provider Fraud Ratio", 0.0, 1.0, 0.10, step=0.01,
                             format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("⚡ Analyze for Fraud", use_container_width=True)


# ─── MAIN AREA ────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1], gap="large")

with left_col:
    st.markdown("### 📊 Analysis Result")

    if not analyze:
        st.markdown("""
        <div style='background:#0d1821; border:1px solid #00c89620; border-radius:16px;
                    padding:48px 32px; text-align:center; margin-bottom:16px;'>
            <div style='font-size:60px; opacity:0.3; margin-bottom:16px;'>🔍</div>
            <div style='color:#6b8fa0; font-size:14px; line-height:1.8;'>
                Configure the claim details in the sidebar<br>and click <strong style='color:#e8f4f0;'>Analyze for Fraud</strong> to get<br>an instant fraud probability score.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        if not model_loaded:
            st.error("Model file not loaded. Please ensure `fraud_detection_model.pkl` is in the app directory.")
        else:
            # Build feature vector
            race_enc = get_race_encoding(race)
            claim_type_enc = 1 if claim_type == "Inpatient" else 0
            gender_enc = 1 if gender == "Male" else 0

            features = [
                claim_amt, claim_dur, hosp_stay, age,
                total_claims, unique_docs, avg_claim, fraud_ratio,
                num_diag, num_proc, chronic_count,
                gender_enc,
                race_enc[0], race_enc[1], race_enc[2], race_enc[3],
                claim_type_enc,
            ]

            features_dict = dict(
                claim_amt=claim_amt, claim_dur=claim_dur, hosp_stay=hosp_stay,
                age=age, total_claims=total_claims, unique_docs=unique_docs,
                avg_claim=avg_claim, fraud_ratio=fraud_ratio, num_diag=num_diag,
                num_proc=num_proc, chronic_count=chronic_count,
            )

            with st.spinner("Running XGBoost inference..."):
                prob, pred = predict(features)

            # Verdict
            if prob >= 0.6:
                verdict_class = "verdict-fraud"
                verdict_title = "🚨 FRAUDULENT"
                verdict_color = "#ff4d6d"
                verdict_sub = "High probability of fraudulent activity. Recommend immediate investigation and claim hold."
            elif prob >= 0.35:
                verdict_class = "verdict-warning"
                verdict_title = "⚠️ SUSPICIOUS"
                verdict_color = "#ffd166"
                verdict_sub = "Elevated risk indicators present. Manual review recommended before processing."
            else:
                verdict_class = "verdict-safe"
                verdict_title = "✅ LEGITIMATE"
                verdict_color = "#00c896"
                verdict_sub = "Claim appears legitimate. Standard processing recommended."

            st.markdown(f"""
            <div class='{verdict_class}'>
                <div class='verdict-title' style='color:{verdict_color}'>{verdict_title}</div>
                <div class='verdict-sub' style='margin-top:8px;'>{verdict_sub}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge
            st.markdown("**Fraud Probability Score**")
            st.plotly_chart(make_gauge(prob), use_container_width=True, config={"displayModeBar": False})

            # Probability breakdown
            col_p1, col_p2 = st.columns(2)
            col_p1.metric("Fraud Probability", f"{prob*100:.1f}%",
                          delta="⚠ Elevated" if prob >= 0.35 else "✓ Normal",
                          delta_color="inverse" if prob >= 0.35 else "normal")
            col_p2.metric("Legitimate Probability", f"{(1-prob)*100:.1f}%")

            # Store in session history
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "verdict": verdict_title.split()[-1],
                "prob": f"{prob*100:.1f}%",
                "amount": f"${claim_amt:,.0f}",
                "color": verdict_color,
            })
            if len(st.session_state.history) > 8:
                st.session_state.history = st.session_state.history[:8]


with right_col:
    if analyze and model_loaded:
        # Risk Factors
        st.markdown("### 🎯 Risk Indicators")
        risk_factors = get_risk_factors(features_dict, prob)
        for icon, text, level in risk_factors:
            color = "#ff4d6d" if level == "high" else "#ffd166" if level == "med" else "#00c896"
            st.markdown(f"""
            <div class='info-card' style='border-color:{color}30;'>
                <span style='font-size:14px;'>{icon}</span>
                <span style='color:#a0bcc8; font-size:13px; margin-left:8px;'>{text}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature Risk Chart
        st.markdown("### 📈 Feature Risk Profile")
        st.plotly_chart(make_feature_importance_chart(features_dict),
                        use_container_width=True, config={"displayModeBar": False})

        # Claim Summary Table
        st.markdown("### 🗂 Claim Summary")
        summary_df = pd.DataFrame({
            "Feature": ["Claim Amount", "Claim Type", "Duration", "Hosp. Stay",
                        "Patient Age", "Chronic Conditions", "Provider Claims",
                        "Unique Doctors", "Fraud Ratio"],
            "Value": [f"${claim_amt:,.0f}", claim_type, f"{claim_dur} days",
                      f"{hosp_stay} days", f"{age} yrs", str(chronic_count),
                      str(total_claims), str(unique_docs), f"{fraud_ratio*100:.1f}%"]
        })
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature"),
                "Value": st.column_config.TextColumn("Value"),
            }
        )

    else:
        st.markdown("""
        <div style='background:#0d1821; border:1px dashed #00c89630; border-radius:16px;
                    padding:48px 32px; text-align:center;'>
            <div style='font-size:48px; opacity:0.2; margin-bottom:12px;'>📈</div>
            <div style='color:#6b8fa0; font-size:13px;'>Risk indicators and feature analysis<br>will appear here after prediction.</div>
        </div>
        """, unsafe_allow_html=True)


# ─── HISTORY TABLE ────────────────────────────────────────────────────────────
if "history" in st.session_state and st.session_state.history:
    st.divider()
    st.markdown("### 🕐 Recent Analyses")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
