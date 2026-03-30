import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# page config
st.set_page_config(
    page_title="ChurnShield",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# custom css
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --blue:      #1F70C1;
    --blue-dark: #155196;
    --blue-dim:  #d6e8f8;
    --cream:     #F7F3EE;
    --cream-2:   #EDE8E1;
    --cream-3:   #DDD7CF;
    --text:      #1C2333;
    --muted:     #6B7385;
    --warn:      #C94040;
    --mid:       #C47B1A;
    --ok:        #1A7A52;
    --radius:    10px;
    --shadow:    0 2px 12px rgba(31,112,193,0.08);
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--cream);
    color: var(--text);
}

/* hide sidebar, fix padding */
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 5rem 2rem 1.5rem !important; max-width: 1400px !important; margin: 0 auto !important; }
header[data-testid="stHeader"] { display: none !important; }

/* topbar */
.topbar {
    background: var(--blue);
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    padding: 1rem 2.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    box-shadow: 0 2px 12px rgba(31,112,193,0.18);
}
.topbar-shield {
    display: flex;
    align-items: center;
    flex-shrink: 0;
}
.topbar-logo {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.2rem;
    color: #fff;
    letter-spacing: -0.01em;
}
.topbar-sub { color: rgba(255,255,255,0.55); font-size: 0.78rem; }

/* section labels */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--blue);
    margin: 0 0 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--blue-dim);
}

/* form cards */
.form-card {
    background: #fff;
    border: 1px solid var(--cream-3);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    box-shadow: var(--shadow);
}
.form-card-title {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a4f5a;
    margin-bottom: 1rem;
}

/* result panel */
.result-wrap {
    background: var(--blue);
    border-radius: var(--radius);
    padding: 2rem 1.8rem;
    text-align: center;
    color: #fff;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(31,112,193,0.25);
}
.result-wrap::after {
    content: '';
    position: absolute;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
    top: -60px; right: -60px;
}
.result-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--blue-dim);
    margin-bottom: 0.4rem;
}
.result-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 4.5rem;
    font-weight: 600;
    line-height: 1;
    color: #fff;
    margin: 0.2rem 0 0.5rem;
}
.result-badge {
    display: inline-block;
    padding: 0.25rem 1rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.badge-low    { background: rgba(26,122,82,.25);  color: #7ff5c0; }
.badge-medium { background: rgba(196,123,26,.25); color: #ffd48a; }
.badge-high   { background: rgba(201,64,64,.25);  color: #ffaaaa; }
.result-sub { color: rgba(255,255,255,0.65); font-size: 0.82rem; }

/* info cards */
.info-card {
    background: #fff;
    border: 1px solid var(--cream-3);
    border-radius: var(--radius);
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
}
.info-card-title {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* progress bars */
.bar-row { margin-bottom: 1rem; }
.bar-meta { display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:4px; }
.bar-name { color: var(--text); font-weight: 500; }
.bar-val  { font-family:'JetBrains Mono',monospace; font-size:0.72rem; }
.bar-track { background: var(--cream-2); border-radius:99px; height:7px; }
.bar-fill  { height:7px; border-radius:99px; }
.bar-explain { font-size:0.73rem; color:var(--muted); margin-top:3px; line-height:1.45; }

/* pills */
.pill-grid { display:flex; flex-wrap:wrap; gap:0.45rem; }
.pill {
    background: var(--cream);
    border: 1px solid var(--cream-3);
    border-radius: 6px;
    padding: 0.3rem 0.65rem;
    font-size: 0.76rem;
    color: var(--text);
}

/* rec rows */
.rec-row {
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--cream-2);
}
.rec-row:last-child { border-bottom: none; }
.rec-icon { font-size: 1rem; flex-shrink: 0; margin-top: 2px; }
.rec-text { font-size: 0.83rem; color: var(--muted); line-height: 1.5; }

/* metric tiles */
.metric-tile {
    background: #fff;
    border: 1px solid var(--cream-3);
    border-radius: var(--radius);
    padding: 1rem;
    text-align: center;
    box-shadow: var(--shadow);
}
.metric-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--blue);
}
.metric-label { font-size: 0.7rem; color: var(--muted); margin-top: 0.2rem; }

/* predict button */
.stButton > button {
    background: var(--blue) !important;
    color: #fff !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2.5rem !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    transition: background .2s !important;
}
.stButton > button:hover { background: var(--blue-dark) !important; }

/* widget overrides */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {
    background: var(--cream) !important;
    border-color: var(--cream-3) !important;
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
    border-radius: 7px !important;
}
label[data-testid="stWidgetLabel"] p {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
div[data-testid="stNumberInput"] button {
    background: var(--cream-2) !important;
    border-color: var(--cream-3) !important;
}
h1,h2,h3,h4 { color: var(--text) !important; }
.stAlert { border-radius: var(--radius) !important; }
</style>
""",
    unsafe_allow_html=True,
)


# model loader
def _model_mtime():
    """Returns modification times of pkl files as a cache key."""
    try:
        return (
            os.path.getmtime("churn_model.pkl"),
            os.path.getmtime("model_columns.pkl"),
        )
    except OSError:
        return (0, 0)


@st.cache_resource
def load_model(mtime):  # mtime busts cache when files are retrained
    if os.path.exists("churn_model.pkl") and os.path.exists("model_columns.pkl"):
        model = joblib.load("churn_model.pkl")
        cols = joblib.load("model_columns.pkl")
        return model, cols
    return None, None


@st.cache_resource
def load_explainer(_model, mtime):  # mtime busts cache when model changes
    """build a treeexplainer once and reuse it."""
    if SHAP_AVAILABLE:
        return shap.TreeExplainer(_model, feature_perturbation="interventional")
    return None


_mtime = _model_mtime()
model, model_columns = load_model(_mtime)
explainer = load_explainer(model, _mtime) if model else None


@st.cache_data
def load_zip_density():
    path = "data/uszips.csv"
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, usecols=["zip", "density"], dtype={"zip": str})
    df["zip"] = df["zip"].str.zfill(5)
    return dict(zip(df["zip"], df["density"]))


def zip_to_location_type(zip_code: str) -> str:
    zip_density = load_zip_density()
    density = zip_density.get(str(zip_code).zfill(5))
    if density is None:
        return None
    if density >= 2500:
        return "Urban"
    elif density >= 1000:
        return "Suburban"
    else:
        return "Rural"


# human-readable feature name map
FEATURE_LABELS = {
    "Tenure Months": "Tenure",
    "Monthly Charges": "Monthly charges",
    "Total Charges": "Total charges",
    "Contract_One year": "One-year contract",
    "Contract_Two year": "Two-year contract",
    "Internet Service_Fiber optic": "Fiber optic internet",
    "Internet Service_No": "No internet service",
    "Payment Method_Electronic check": "Electronic check payment",
    "Payment Method_Mailed check": "Mailed check payment",
    "Tech Support_Yes": "Has tech support",
    "Online Security_Yes": "Has online security",
    "Online Backup_Yes": "Has online backup",
    "Device Protection_Yes": "Has device protection",
    "Streaming TV_Yes": "Has streaming TV",
    "Streaming Movies_Yes": "Has streaming movies",
    "Multiple Lines_Yes": "Has multiple lines",
    "Phone Service_Yes": "Has phone service",
    "Paperless Billing_Yes": "Paperless billing",
    "Partner_Yes": "Has partner",
    "Dependents_Yes": "Has dependents",
    "Senior Citizen_Yes": "Senior citizen",
    "Gender_Male": "Male",
    "Location Type_Suburban": "Suburban area",
    "Location Type_Urban": "Urban area",
}


def friendly_name(col):
    return FEATURE_LABELS.get(col, col.replace("_", " "))


# plain-english explanation generator
def explain_factor(col, shap_val, customer_val):
    """return a one-sentence plain-english reason for a shap contribution."""
    direction = "increases" if shap_val > 0 else "reduces"
    name = friendly_name(col)

    # numeric columns
    if col == "Tenure Months":
        adj = (
            "short"
            if customer_val < 12
            else ("moderate" if customer_val < 36 else "long")
        )
        return f"a {adj} tenure of {int(customer_val)} months {direction} churn risk — longer customers are harder to lose."
    if col == "Monthly Charges":
        adj = (
            "high"
            if customer_val > 75
            else ("moderate" if customer_val > 45 else "low")
        )
        return f"{adj} monthly charges of ${customer_val:.0f} {direction} churn risk."
    if col == "Total Charges":
        return f"lifetime spend of ${customer_val:,.0f} {direction} churn risk."

    # boolean/encoded columns — interpret the 1/0 value
    is_active = customer_val == 1
    if col == "Contract_Two year":
        return (
            "two-year contracts strongly reduce churn risk — customers are locked in."
            if is_active
            else "no long-term contract means higher switching freedom."
        )
    if col == "Contract_One year":
        return (
            "one-year contracts moderately reduce churn risk vs. month-to-month."
            if is_active
            else "no annual contract — customer can leave any time."
        )
    if col == "Internet Service_Fiber optic":
        return (
            "fiber optic customers churn more often, possibly due to higher bills and more competition."
            if is_active
            else "non-fiber customers tend to be stickier."
        )
    if col == "Payment Method_Electronic check":
        return (
            "electronic check users churn at a higher rate — often less engaged customers."
            if is_active
            else "automated payment methods correlate with lower churn."
        )
    if col == "Tech Support_Yes":
        return (
            "having tech support reduces frustration and churn."
            if is_active
            else "no tech support increases the risk of service dissatisfaction."
        )
    if col == "Online Security_Yes":
        return (
            "online security subscribers feel more invested in the service."
            if is_active
            else "absence of security add-ons correlates with lower commitment."
        )
    if col == "Paperless Billing_Yes":
        return (
            "paperless billing customers tend to be more digitally engaged — slightly higher churn."
            if is_active
            else "paper billing customers tend to be less likely to actively cancel."
        )

    if col == "Gender_Male":
        return "gender shows no meaningful difference in churn rate in this dataset — this contribution is noise."

    if col == "Location Type_Suburban":
        return (
            "suburban customers face moderate competition — some switching options but higher friction than urban areas."
            if is_active
            else "this customer is in a rural area — fewer ISP alternatives make them stickier."
        )
    if col == "Location Type_Urban":
        return (
            "urban customers have the most ISP competition — easier to comparison-shop and switch providers."
            if is_active
            else "this customer is not in an urban area, reducing competitive switching pressure."
        )

    # fallback generic explanation
    val_desc = "enabled" if is_active else "not enabled"
    return f"{name} ({val_desc}) {direction} this customer's churn probability."


# helper: build feature row
def build_feature_row(inputs: dict, model_columns: list) -> pd.DataFrame:
    """one-hot encode inputs to match training schema."""
    row = inputs.copy()

    # Manually encode Location Type — drop_first=True in training dropped Rural (alphabetically first),
    # keeping Location Type_Suburban and Location Type_Urban. get_dummies on a single row
    # only sees one category and drops everything, so we encode explicitly.
    loc = row.pop("Location Type", None)
    row["Location Type_Suburban"] = 1 if loc == "Suburban" else 0
    row["Location Type_Urban"] = 1 if loc == "Urban" else 0
    # Rural is the reference (both = 0)

    df = pd.DataFrame([row])
    cat_cols = [
        "Senior Citizen",
        "Gender",
        "Partner",
        "Dependents",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
    ]
    df = pd.get_dummies(
        df, columns=[c for c in cat_cols if c in df.columns], drop_first=True
    )
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    return df[model_columns]


# topbar
st.markdown(
    """
<div class="topbar">
  <div class="topbar-shield">
    <svg width="28" height="32" viewBox="0 0 28 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M14 1L2 6V15C2 22.18 7.2 28.9 14 31C20.8 28.9 26 22.18 26 15V6L14 1Z"
            fill="rgba(255,255,255,0.15)" stroke="white" stroke-width="1.5" stroke-linejoin="round"/>
      <path d="M9 16l3.5 3.5L19 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
  <div class="topbar-logo">ChurnShield</div>
</div>
""",
    unsafe_allow_html=True,
)

if model is None:
    st.warning(
        "**no trained model found.** run `model.py` first to generate `churn_model.pkl` and `model_columns.pkl`, then relaunch."
    )


# main layout: form left, results right
form_col, result_col = st.columns([1.15, 1], gap="large")

with form_col:
    st.markdown(
        '<div class="section-label">Customer Profile</div>', unsafe_allow_html=True
    )

    # demographics
    st.markdown(
        '<div class="form-card"><div class="form-card-title">Demographics</div>',
        unsafe_allow_html=True,
    )
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with d2:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with d3:
        partner = st.selectbox("Has Partner", ["Yes", "No"])
    with d4:
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    with d5:
        zip_input = st.text_input("Zip Code *", placeholder="e.g. 90210", max_chars=5)
        zip_is_valid_format = (
            zip_input.strip().isdigit() and len(zip_input.strip()) == 5
        )
        derived_location = (
            zip_to_location_type(zip_input) if zip_is_valid_format else None
        )
        zip_invalid = zip_input.strip() == "" or derived_location is None
        if zip_input.strip() == "":
            st.markdown(
                '<span style="color:var(--muted);font-size:0.72rem">enter zip to auto-detect area type</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<span style="color:#C94040;font-size:0.75rem">⚠ This field is required.</span>',
                unsafe_allow_html=True,
            )
        elif derived_location is None:
            st.markdown(
                '<span style="color:#C94040;font-size:0.72rem">⚠ zip not recognised</span>',
                unsafe_allow_html=True,
            )
        else:
            color = {
                "Urban": "var(--warn)",
                "Suburban": "var(--mid)",
                "Rural": "var(--ok)",
            }[derived_location]
            st.markdown(
                f'<span style="font-size:0.72rem;font-weight:600;color:{color}">📍 {derived_location}</span>',
                unsafe_allow_html=True,
            )
        location_type = derived_location
    st.markdown("</div>", unsafe_allow_html=True)

    # account details
    st.markdown(
        '<div class="form-card"><div class="form-card-title">Account Details</div>',
        unsafe_allow_html=True,
    )
    a1, a2 = st.columns(2)
    with a1:
        tenure = st.number_input(
            "Tenure (months) *",
            min_value=0,
            max_value=72,
            value=None,
            step=1,
            placeholder="e.g. 12",
        )
        if tenure is None:
            st.markdown(
                '<span style="color:#C94040;font-size:0.75rem">⚠ This field is required.</span>',
                unsafe_allow_html=True,
            )
    with a2:
        monthly_charges = st.number_input(
            "Monthly Charges ($) *",
            min_value=0.0,
            max_value=200.0,
            value=None,
            step=0.5,
            placeholder="e.g. 65.00",
        )
        if monthly_charges is None:
            st.markdown(
                '<span style="color:#C94040;font-size:0.75rem">⚠ This field is required.</span>',
                unsafe_allow_html=True,
            )
    total_charges = (
        (tenure * monthly_charges)
        if (tenure is not None and monthly_charges is not None)
        else 0.0
    )
    a4, a5, a6 = st.columns(3)
    with a4:
        contract = st.selectbox(
            "Contract Type", ["Month-to-month", "One year", "Two year"]
        )
    with a5:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
    with a6:
        paperless_bill = st.selectbox("Paperless Billing", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)

    # phone services
    st.markdown(
        '<div class="form-card"><div class="form-card-title">Phone Services</div>',
        unsafe_allow_html=True,
    )
    p1, p2 = st.columns(2)
    with p1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with p2:
        if phone_service == "No":
            multiple_lines = "No phone service"
            st.selectbox(
                "Multiple Lines",
                ["No phone service"],
                disabled=True,
                help="Locked — no phone service",
            )
        else:
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)

    # internet services
    st.markdown(
        '<div class="form-card"><div class="form-card-title">Internet Services</div>',
        unsafe_allow_html=True,
    )
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        internet_svc = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    no_internet = internet_svc == "No"
    with i2:
        if no_internet:
            online_security = "No internet service"
            st.selectbox(
                "Online Security",
                ["No internet service"],
                disabled=True,
                help="Locked — no internet service",
            )
        else:
            online_security = st.selectbox("Online Security", ["Yes", "No"])
    with i3:
        if no_internet:
            online_backup = "No internet service"
            st.selectbox(
                "Online Backup",
                ["No internet service"],
                disabled=True,
                help="Locked — no internet service",
            )
        else:
            online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    with i4:
        if no_internet:
            device_prot = "No internet service"
            st.selectbox(
                "Device Protection",
                ["No internet service"],
                disabled=True,
                help="Locked — no internet service",
            )
        else:
            device_prot = st.selectbox("Device Protection", ["Yes", "No"])
    i5, i6, i7, _ = st.columns(4)
    with i5:
        if no_internet:
            tech_support = "No internet service"
            st.selectbox(
                "Tech Support",
                ["No internet service"],
                disabled=True,
                help="Locked — no internet service",
            )
        else:
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    with i6:
        if no_internet:
            streaming_tv = "No internet service"
            st.selectbox(
                "Streaming TV",
                ["No internet service"],
                disabled=True,
                help="Locked — no internet service",
            )
        else:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    with i7:
        if no_internet:
            streaming_movies = "No internet service"
            st.selectbox(
                "Streaming Movies",
                ["No internet service"],
                disabled=True,
                help="Locked — no internet service",
            )
        else:
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)

    # predict button
    predict_btn = st.button("Predict Churn Risk")


# results column
with result_col:
    st.markdown(
        '<div class="section-label">Risk Assessment</div>',
        unsafe_allow_html=True,
    )

    if not predict_btn:
        st.markdown(
            """
        <div class="info-card" style="text-align:center;padding:3rem 1.5rem;">
          <div style="font-size:2.8rem;margin-bottom:0.9rem">📋</div>
          <div style="font-weight:600;color:var(--text);margin-bottom:0.5rem;font-size:1rem">Ready to score</div>
          <div style="color:var(--muted);font-size:0.84rem;line-height:1.7">
            Fill in the customer profile on the left<br>and click <strong style="color:var(--blue)">Predict Churn Risk</strong>.
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    else:
        # validate required fields
        if tenure is None or monthly_charges is None or zip_invalid:
            red_css = ""
            if tenure is None:
                red_css += """
div[data-testid="stNumberInput"]:has(input[aria-label="Tenure (months) *"]) input { border-color: #C94040 !important; box-shadow: 0 0 0 1px #C94040 !important; }"""
            if monthly_charges is None:
                red_css += """
div[data-testid="stNumberInput"]:has(input[aria-label="Monthly Charges ($) *"]) input { border-color: #C94040 !important; box-shadow: 0 0 0 1px #C94040 !important; }"""
            if zip_invalid:
                red_css += """
div[data-testid="stTextInput"]:has(input[aria-label="Zip Code *"]) input { border-color: #C94040 !important; box-shadow: 0 0 0 1px #C94040 !important; }"""
            st.markdown(f"<style>{red_css}</style>", unsafe_allow_html=True)
            st.stop()

        # build input dict
        inputs = {
            "Gender": gender,
            "Senior Citizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Location Type": location_type,
            "Tenure Months": tenure,
            "Phone Service": phone_service,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_svc,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection": device_prot,
            "Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Contract": contract,
            "Paperless Billing": paperless_bill,
            "Payment Method": payment_method,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges,
        }

        # predict
        if model is None:
            st.error("model not found. run `model.py` first.")
            st.stop()

        X_input = build_feature_row(inputs, model_columns)
        prob = float(model.predict_proba(X_input)[0, 1])
        pred = int(model.predict(X_input)[0])

        pct = int(round(prob * 100))
        if prob < 0.35:
            badge_cls, risk_label, bar_color = "badge-low", "Low Risk", "#1A7A52"
        elif prob < 0.65:
            badge_cls, risk_label, bar_color = "badge-medium", "Medium Risk", "#C47B1A"
        else:
            badge_cls, risk_label, bar_color = "badge-high", "High Risk", "#C94040"

        # risk meter
        st.markdown(
            f"""
        <div class="result-wrap">
          <div class="result-eyebrow">churn probability</div>
          <div class="result-pct">{pct}%</div>
          <div class="result-badge {badge_cls}">{risk_label}</div>
          <div class="result-sub">{"this customer is likely to churn" if pred else "this customer is expected to stay"}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # summary metric tiles
        factors = {
            "month-to-month contract": contract == "Month-to-month",
            "fiber optic internet": internet_svc == "Fiber optic",
            "no tech support": tech_support == "No",
            "electronic check": payment_method == "Electronic check",
            "low tenure (< 12 mo)": tenure < 12,
            "no online security": online_security == "No",
            "paperless billing": paperless_bill == "Yes",
            "high monthly charges": monthly_charges > 75,
            "urban area (high competition)": location_type == "Urban",
        }
        risk_count = sum(1 for v in factors.values() if v)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-num">{tenure}</div><div class="metric-label">tenure (months)</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-num">${monthly_charges:.0f}</div><div class="metric-label">monthly charges</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-num">${total_charges:,.0f}</div><div class="metric-label">total charges</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-num">{risk_count}/8</div><div class="metric-label">risk signals</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # contributing factors (shap or fallback)
        st.markdown(
            '<div class="info-card"><div class="info-card-title">Contributing Factors</div>',
            unsafe_allow_html=True,
        )

        if SHAP_AVAILABLE and explainer is not None:
            # shap: customer-specific explanations
            shap_vals = explainer.shap_values(X_input, check_additivity=False)
            # handle all output shapes from different shap/sklearn versions:
            #   list of arrays [class0, class1] -> take class1, first sample
            #   3-d array (n_samples, n_features, n_classes) -> take class1, first sample
            #   2-d array (n_samples, n_features) -> take first sample
            import numpy as np

            if isinstance(shap_vals, list):
                sv = np.array(shap_vals[1])[0]
            else:
                sv = np.array(shap_vals)
                if sv.ndim == 3:  # (n_samples, n_features, n_classes)
                    sv = sv[0, :, 1]
                elif sv.ndim == 2:  # (n_samples, n_features)
                    sv = sv[0]
                else:  # already 1-d
                    sv = sv

            shap_series = pd.Series(sv, index=model_columns)
            top_shap = shap_series.abs().nlargest(8)
            top_shap_signed = shap_series[top_shap.index]
            abs_max = top_shap.max()

            st.markdown(
                '<p style="color:var(--muted);font-size:0.74rem;margin:-0.3rem 0 1rem">'
                "🔍 customer-specific · each bar shows how strongly a factor pushes this customer toward or away from churning.</p>",
                unsafe_allow_html=True,
            )

            # convert shap values to % impact relative to their total absolute pull
            total_abs = top_shap.sum()
            bars_html = ""
            for feat in top_shap.index:
                sv_val = float(top_shap_signed[feat])
                cust_val = float(X_input[feat].iloc[0])
                width = int(abs(sv_val) / abs_max * 100)
                impact_pct = (
                    int(round(abs(sv_val) / total_abs * 100)) if total_abs > 0 else 0
                )
                color = "#C94040" if sv_val > 0 else "#1A7A52"
                arrow = "▲" if sv_val > 0 else "▼"
                direction_word = "raises risk" if sv_val > 0 else "lowers risk"
                label = friendly_name(feat)
                sentence = explain_factor(feat, sv_val, cust_val)

                bars_html += f"""
                <div class="bar-row">
                  <div class="bar-meta">
                    <span class="bar-name">{arrow} {label}</span>
                    <span class="bar-val" style="color:{color}">{impact_pct}% impact · {direction_word}</span>
                  </div>
                  <div class="bar-track">
                    <div class="bar-fill" style="width:{width}%;background:{color};"></div>
                  </div>
                  <div class="bar-explain">{sentence}</div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

        else:
            # fallback: global feature importances
            if not SHAP_AVAILABLE:
                st.caption(
                    "💡 install `shap` for customer-specific explanations: `pip install shap`"
                )
            fi = pd.Series(model.feature_importances_, index=model_columns)
            top_fi = fi.nlargest(10)
            fi_max = top_fi.max()
            fi_total = top_fi.sum()
            bars_html = ""
            for feat, imp in top_fi.items():
                short = friendly_name(feat)[:40]
                width = int(imp / fi_max * 100)
                impact_pct = int(round(imp / fi_total * 100)) if fi_total > 0 else 0
                bars_html += f"""
                <div class="bar-row">
                  <div class="bar-meta">
                    <span class="bar-name">{short}</span>
                    <span class="bar-val" style="color:var(--blue)">{impact_pct}% influence</span>
                  </div>
                  <div class="bar-track">
                    <div class="bar-fill" style="width:{width}%;background:var(--blue);"></div>
                  </div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # active risk signals
        st.markdown(
            '<div class="info-card"><div class="info-card-title">Active Risk Signals</div>',
            unsafe_allow_html=True,
        )
        present = [k for k, v in factors.items() if v]
        if present:
            pills = "".join(f'<div class="pill">⚠ {f}</div>' for f in present)
            st.markdown(f'<div class="pill-grid">{pills}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color:var(--muted);font-size:0.84rem;margin:0">no major risk signals detected.</p>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # recommendations
        st.markdown(
            '<div class="info-card"><div class="info-card-title">Recommended Actions</div>',
            unsafe_allow_html=True,
        )

        recs = []
        if contract == "Month-to-month":
            recs.append(
                (
                    "📋",
                    "Offer a discounted 1- or 2-year contract upgrade to increase switching cost.",
                )
            )
        if internet_svc == "Fiber optic" and tech_support == "No":
            recs.append(
                (
                    "🛡️",
                    "Upsell the tech support bundle — fiber users without support churn at 2× the rate.",
                )
            )
        if payment_method == "Electronic check":
            recs.append(
                (
                    "💳",
                    "Encourage auto-pay via bank transfer or credit card; echeck correlates strongly with churn.",
                )
            )
        if tenure < 12:
            recs.append(
                (
                    "🎁",
                    "Activate the early loyalty program — first-year customers are highest-risk.",
                )
            )
        if monthly_charges > 75 and contract == "Month-to-month":
            recs.append(
                (
                    "💰",
                    "Propose a loyalty discount or value-add (e.g. free streaming add-on) to justify charges.",
                )
            )
        if online_security == "No" and internet_svc != "No":
            recs.append(
                (
                    "🔒",
                    "Offer a 3-month free trial of online security — improves satisfaction and stickiness.",
                )
            )
        if not recs:
            recs.append(
                (
                    "✅",
                    "No urgent interventions needed. continue standard engagement cadence.",
                )
            )

        recs_html = "".join(
            f'<div class="rec-row"><span class="rec-icon">{icon}</span><span class="rec-text">{text}</span></div>'
            for icon, text in recs
        )
        st.markdown(recs_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Feature Logic Explorer
fle_intro = (
    '<p style="color:var(--muted);font-size:0.83rem;line-height:1.7;margin:0 0 1.25rem">'
    "Each feature below was deliberately chosen — or dropped — based on its predictive signal, "
    "data quality, and interpretability. Expand any feature to see what values raise or lower churn risk, "
    "and why the model weights it the way it does."
    "</p>"
)
FEATURE_LOGIC = {
    "📅 Tenure": {
        "why_kept": "Tenure is one of the single strongest predictors of churn in subscription businesses. It captures relationship depth — the longer a customer has stayed, the higher their switching cost (learned UI, bundled services, loyalty discounts) and the more satisfied they presumably are.",
        "why_others_dropped": "Raw signup date was dropped — absolute dates leak time-of-year bias and don't generalise across cohorts. Tenure (months) is a clean, comparable numeric signal.",
        "values": [
            (
                "0 – 11 months",
                "🔴 Highest risk",
                "New customers haven't yet experienced full service value. Buyer's remorse, onboarding friction, and competitor offers hit hardest in the first year. Churn rate in this band is typically 2–3× the average.",
            ),
            (
                "12 – 35 months",
                "🟡 Moderate risk",
                "Customer has cleared the critical early window but hasn't reached deep loyalty. Price sensitivity remains — a competitor offer can still flip them.",
            ),
            (
                "36 + months",
                "🟢 Low risk",
                "Long-tenure customers have high inertia. They've integrated the service into their routines, likely have bundled add-ons, and rarely shop around actively.",
            ),
        ],
    },
    "💳 Contract Type": {
        "why_kept": "Contract type directly encodes commitment and exit cost. A month-to-month customer can leave with zero penalty at any time; a two-year customer faces early termination fees. This is the clearest structural churn lever in the dataset.",
        "why_others_dropped": "No comparable variable was dropped — contract type is unique. Billing cycle (monthly vs annual billing) overlaps significantly and was absorbed into contract type.",
        "values": [
            (
                "Month-to-month",
                "🔴 Highest risk",
                "No lock-in whatsoever. The customer can cancel the same day they see a competitor ad. This group accounts for the majority of all churns in the dataset.",
            ),
            (
                "One year",
                "🟡 Moderate risk",
                "Moderate switching cost — an early termination fee acts as a brake. Customers are more likely to wait out the contract than actively churn mid-term.",
            ),
            (
                "Two year",
                "🟢 Lowest risk",
                "Strong financial commitment. Churn in this group is rare and usually driven by relocation or extreme service failure, not preference.",
            ),
        ],
    },
    "💰 Monthly Charges": {
        "why_kept": "Monthly charges capture current price pressure. Total charges are auto-calculated as tenure × monthly charges — a proxy for cumulative spend — and fed to the model automatically without requiring manual input.",
        "why_others_dropped": "CLTV (Customer Lifetime Value) was available in some dataset versions but is a derived field computed from charges and tenure — including it would be near-circular. It was excluded to keep features causal.",
        "values": [
            (
                "Monthly < $45",
                "🟢 Low pressure",
                "Budget tier customers have little financial motivation to switch — competitors would offer similar or higher prices. Price is not their pain point.",
            ),
            (
                "Monthly $45 – $75",
                "🟡 Moderate pressure",
                "Mid-range customers are comparison-shopping territory. A competitor offering the same bundle at $10 less could trigger a switch.",
            ),
            (
                "Monthly > $75",
                "🔴 High pressure",
                "Premium-tier customers scrutinise their bill. If they don't perceive commensurate value (fast internet, good support), they churn. Especially dangerous on month-to-month contracts.",
            ),
        ],
    },
    "🌐 Internet Service": {
        "why_kept": "Internet service type is both a product tier signal and a satisfaction proxy. Fiber optic customers pay more and expect more — making them more likely to churn when expectations aren't met. DSL customers are often lower-expectation, stickier subscribers.",
        "why_others_dropped": "Raw bandwidth/speed metrics were not available. Internet service type is the best available proxy for the technology tier a customer is on.",
        "values": [
            (
                "No internet",
                "🟢 Low risk",
                "Phone-only customers have simpler needs, fewer competitors to switch to, and lower bills. Churn is rare.",
            ),
            (
                "DSL",
                "🟡 Moderate risk",
                "Older technology, lower cost, lower expectations. Customers are generally satisfied with 'good enough' service and churn less than fiber users.",
            ),
            (
                "Fiber optic",
                "🔴 Higher risk",
                "Counterintuitively, fiber customers churn more. They pay premium prices, have high expectations, and operate in a more competitive market. Without supporting services (security, tech support), dissatisfaction builds quickly.",
            ),
        ],
    },
    "🛡️ Tech Support & Security": {
        "why_kept": "These add-ons act as satisfaction anchors. A customer who has set up tech support has a direct service relationship with the provider — they're less likely to want to re-establish that relationship elsewhere. Security add-ons create a sense of dependency and trust.",
        "why_others_dropped": "Streaming TV and Movies were kept as features but have weaker individual signal — they contribute to bundle depth. Customer satisfaction scores (CSAT/NPS) would be ideal but weren't in the dataset.",
        "values": [
            (
                "Tech Support: Yes",
                "🟢 Reduces risk",
                "Customers with tech support have an active human touchpoint with the provider. Resolving issues builds loyalty. These customers churn at roughly half the rate of those without.",
            ),
            (
                "Tech Support: No",
                "🔴 Raises risk",
                "No support relationship means problems go unresolved. For fiber optic customers especially, lack of support is a top churn driver.",
            ),
            (
                "Online Security: Yes",
                "🟢 Reduces risk",
                "Security-subscribed customers feel the service is protecting something important. Switching means giving that up and re-configuring protection elsewhere — a real friction cost.",
            ),
            (
                "Online Security: No",
                "🟡 Mild risk",
                "Lower commitment to the service ecosystem. No security subscription doesn't guarantee churn, but removes a loyalty anchor.",
            ),
        ],
    },
    "💳 Payment Method": {
        "why_kept": "Payment method is a strong proxy for customer engagement and intentionality. Automated payment customers have opted in more deliberately — they've handed over banking details, signalling higher commitment. Manual payment methods correlate with passive or disengaged customers.",
        "why_others_dropped": "Specific bank or card provider data was not available. Payment method category is the best available signal for payment intentionality.",
        "values": [
            (
                "Electronic check",
                "🔴 Highest risk",
                "The highest-churn payment group. Electronic check users are often one-time setup customers who never fully committed. It's the easiest method to set up and the easiest to cancel.",
            ),
            (
                "Mailed check",
                "🟡 Moderate risk",
                "Slightly lower churn than e-check — these customers engage each month by physically mailing a payment. But it's still manual and non-committed.",
            ),
            (
                "Bank transfer (auto)",
                "🟢 Low risk",
                "Auto-pay via bank transfer signals intentional long-term commitment. Cancelling requires active effort — the customer must go in and stop the transfer.",
            ),
            (
                "Credit card (auto)",
                "🟢 Low risk",
                "Similar to bank transfer. Auto-pay customers are systematically more engaged. Credit card users also tend to be in higher income brackets with fewer price objections.",
            ),
        ],
    },
    "📍 Location Type": {
        "why_kept": "Rather than encoding 1,100+ individual California cities, we engineered a single Location Type feature using US Census population density thresholds. Urban customers have more ISP competitors and higher churn tendency; rural customers have fewer alternatives and are stickier.",
        "why_others_dropped": "Raw City and Zip Code were dropped — one-hot encoding 1,131 cities from ~7,000 rows creates severe sparsity, with most cities having only 2-5 training examples. Population density compresses this into three meaningful, generalisable categories.",
        "values": [
            (
                "Urban",
                "🔴 Higher risk",
                "Dense markets have the most ISP competition. Customers can easily switch to a rival offering a better deal. Urban customers are more digitally engaged and more likely to comparison-shop.",
            ),
            (
                "Suburban",
                "🟡 Moderate risk",
                "Moderate competition. Customers have some alternatives but switching friction is higher than in urban areas. Price and service quality are the main churn drivers.",
            ),
            (
                "Rural",
                "🟢 Lower risk",
                "Fewer competitor options mean customers often stay by default. However, service quality issues hit harder since alternatives are limited — extreme dissatisfaction can still trigger churn.",
            ),
        ],
    },
    "👤 Demographics": {
        "why_kept": "Demographics are kept for model completeness and equity auditing, but they carry low individual predictive weight. The most signal comes from Dependents and Senior Citizen — not gender or partner status.",
        "why_others_dropped": "Age (continuous) was not available — Senior Citizen (binary 65+) is the available proxy. Race and income were not in the dataset and were not used.",
        "values": [
            (
                "Senior Citizen: Yes",
                "🟡 Slight risk",
                "Senior customers may be less comfortable navigating cancellation, which slightly reduces churn. However, on fixed incomes they can be price-sensitive. Effect is modest.",
            ),
            (
                "Dependents: Yes",
                "🟢 Slightly reduces risk",
                "Customers with dependents have more people relying on the service. Disrupting connectivity for a household is a higher-friction decision than for a single user.",
            ),
            (
                "Partner: Yes",
                "🟢 Slightly reduces risk",
                "Shared services are harder to cancel. Also correlates loosely with household stability.",
            ),
            (
                "Gender",
                "⚪ Neutral",
                "Gender shows no statistically meaningful difference in churn rate in this dataset. It is retained for completeness and bias monitoring, not predictive value.",
            ),
        ],
    },
    "📋 Paperless Billing": {
        "why_kept": "Paperless billing is a digital engagement signal. It correlates with customers who are more active online — and paradoxically, more likely to churn, because they're also more likely to comparison-shop online and respond to competitor digital marketing.",
        "why_others_dropped": "No other billing format variables were available. This is the only billing-behaviour signal in the dataset.",
        "values": [
            (
                "Paperless: Yes",
                "🟡 Slightly raises risk",
                "Digital customers are more engaged and more informed — they see competitor ads, read reviews, and compare prices. Churn rate is modestly higher than paper billing customers.",
            ),
            (
                "Paperless: No",
                "🟢 Slightly reduces risk",
                "Paper billing customers tend to be older, less digitally active, and less likely to actively seek out alternative providers. Inertia plays in the provider's favour.",
            ),
        ],
    },
}

accordion_html = "<div style='border:1px solid var(--cream-3);border-radius:var(--radius);overflow:hidden;margin-bottom:2rem;'>"
for label, info in FEATURE_LOGIC.items():
    cards_inner = ""
    for val_label, risk, explanation in info["values"]:
        cards_inner += (
            '<div style="display:flex;justify-content:space-between;align-items:flex-start;'
            'gap:1rem;padding:0.75rem 0;border-bottom:1px solid var(--cream-2);">'
            '<div style="flex:1">'
            f'<div style="font-size:0.83rem;font-weight:600;color:var(--text);margin-bottom:0.25rem">{val_label}</div>'
            f'<div style="font-size:0.8rem;color:var(--muted);line-height:1.6">{explanation}</div>'
            "</div>"
            f'<div style="font-size:0.74rem;font-weight:600;color:var(--muted);white-space:nowrap;padding-top:2px">{risk}</div>'
            "</div>"
        )
    accordion_html += (
        '<details style="border-bottom:1px solid var(--cream-3);">'
        '<summary style="list-style:none;display:flex;justify-content:space-between;'
        "align-items:center;padding:0.85rem 1.1rem;cursor:pointer;font-size:0.83rem;"
        'font-weight:600;color:var(--text);background:#fff;user-select:none;">'
        f"<span>{label}</span>"
        '<span style="font-size:0.7rem;color:var(--muted)">▼</span>'
        "</summary>"
        f'<div style="padding:0.5rem 1.1rem 0.75rem;background:var(--cream);">{cards_inner}</div>'
        "</details>"
    )
accordion_html += "</div>"

outer_html = (
    '<details style="margin-bottom:2rem;border-radius:6px;overflow:hidden;">'
    '<summary style="list-style:none;display:flex;justify-content:space-between;'
    "align-items:center;padding:0.75rem 1rem;cursor:pointer;font-size:0.88rem;"
    'font-weight:600;color:#F7F3EE;background:#1F70C1;border-radius:6px;user-select:none;">'
    "<span>Feature Logic Explorer</span>"
    '<span style="font-size:0.7rem;color:rgba(247,243,238,0.7)">▼</span>'
    "</summary>"
    f'<div style="padding:1.25rem 1rem 1rem;border:1px solid #DDD7CF;border-top:none;box-shadow:0 4px 12px rgba(0,0,0,0.06);'
    f'border-radius:0 0 6px 6px;">{fle_intro}{accordion_html}</div>'
    "</details>"
)
st.markdown(outer_html, unsafe_allow_html=True)
