import streamlit as st
import pandas as pd
import joblib
from typing import Dict, List, Tuple

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Cognitive Limitation Risk Calculator",
    page_icon="🧠",
    layout="wide",
)

# =====================================================
# STYLE
# =====================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #f6f7fb;
    }
    .block-container {
        max-width: 980px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
    }
    .small-muted {
        color: #6b7280;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# LOAD MODEL ARTIFACT
# =====================================================
@st.cache_resource
def load_artifacts() -> Tuple[object, List[str], float]:
    model_artifact = joblib.load("calibrated_gb_model.pkl")

    if isinstance(model_artifact, dict):
        model = model_artifact["model"]
        feature_names = model_artifact["feature_names"]
        threshold = model_artifact.get("threshold", 0.20)
    else:
        model = model_artifact
        feature_names = [
            "phq9_total",
            "sleep_hours",
            "short_sleep",
            "daytime_sleepiness",
            "med_count",
            "polypharmacy",
            "cns_load",
            "psych_med_count",
            "is_opioid",
            "is_antidepressant",
            "anemia",
            "high_rdw",
            "bmi",
            "age",
            "sex_male",
            "income_ratio",
        ]
        threshold = 0.20

    return model, feature_names, threshold


# =====================================================
# INPUT MAPPINGS
# =====================================================
PHQ_OPTIONS = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}

SLEEPINESS_OPTIONS = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Almost always": 4,
}

# Training alignment:
# sex_male = 1 if Male else 0
SEX_TO_MODEL = {
    "Female": 0,
    "Male": 1,
}

INCOME_OPTIONS = {
    "Very limited relative income": 0.5,
    "Below average relative income": 1.0,
    "Moderate relative income": 2.0,
    "Above average relative income": 3.5,
    "Higher relative income": 5.0,
}

MEDICATION_CLASS_HELP = {
    "Antidepressant": "Examples: sertraline, fluoxetine, escitalopram, bupropion",
    "Benzodiazepine": "Examples: alprazolam, lorazepam, clonazepam, diazepam",
    "Antipsychotic": "Examples: quetiapine, risperidone, aripiprazole",
    "Sleep medication or sedative": "Examples: zolpidem, eszopiclone, zaleplon",
    "Anticholinergic medication": "Examples: diphenhydramine, hydroxyzine, oxybutynin",
    "Opioid pain medication": "Examples: hydrocodone, oxycodone, tramadol, morphine",
    "Seizure or nerve pain medication": "Examples: gabapentin, pregabalin",
    "Muscle relaxant": "Examples: cyclobenzaprine, tizanidine, methocarbamol",
    "Oral steroid": "Examples: prednisone, methylprednisolone",
    "Stimulant": "Examples: methylphenidate, amphetamine",
}

# =====================================================
# HELPERS
# =====================================================
def risk_category(risk: float) -> str:
    if risk < 0.10:
        return "Low"
    if risk < 0.25:
        return "Moderate"
    if risk < 0.40:
        return "Elevated"
    return "High"


def risk_color(risk: float) -> str:
    if risk < 0.10:
        return "#10b981"
    if risk < 0.25:
        return "#f59e0b"
    if risk < 0.40:
        return "#f97316"
    return "#ef4444"


def input_help(text: str):
    st.caption(text)


def compute_phq_total(user_input: Dict[str, float]) -> int:
    return (
        user_input["phq_little_interest"]
        + user_input["phq_low_mood"]
        + user_input["phq_sleep_change"]
        + user_input["phq_low_energy"]
        + user_input["phq_appetite_change"]
        + user_input["phq_feeling_bad"]
        + user_input["phq_concentration"]
        + user_input["phq_slow_restless"]
        + user_input["phq_self_harm"]
    )


def build_features(user_input: Dict[str, float], feature_names: List[str]) -> pd.DataFrame:
    med_count = user_input["med_count"]
    polypharmacy = int(med_count >= 5)

    cns_load = (
        int(user_input["is_benzo"])
        + int(user_input["is_sedative"])
        + int(user_input["is_opioid"])
        + int(user_input["is_antipsychotic"])
        + int(user_input["is_anticonvulsant"])
        + int(user_input["is_muscle_relaxant"])
    )

    psych_med_count = (
        int(user_input["is_antidepressant"])
        + int(user_input["is_antipsychotic"])
        + int(user_input["is_benzo"])
        + int(user_input["is_stimulant"])
    )

    sleep_hours = user_input["sleep_hours"]
    short_sleep = int(sleep_hours < 6)

    row = {
        "phq9_total": compute_phq_total(user_input),
        "sleep_hours": sleep_hours,
        "short_sleep": short_sleep,
        "daytime_sleepiness": user_input["daytime_sleepiness"],
        "med_count": med_count,
        "polypharmacy": polypharmacy,
        "cns_load": cns_load,
        "psych_med_count": psych_med_count,
        "is_opioid": int(user_input["is_opioid"]),
        "is_antidepressant": int(user_input["is_antidepressant"]),
        "anemia": int(user_input["anemia"]),
        "high_rdw": int(user_input["high_rdw"]),
        "bmi": user_input["bmi"],
        "age": user_input["age"],
        "sex_male": int(user_input["sex_male"]),
        "income_ratio": user_input["income_ratio"],
    }

    X = pd.DataFrame([row])

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    return X[feature_names]


def explain_domains(user_input: Dict[str, float]) -> List[str]:
    reasons = []
    phq_total = compute_phq_total(user_input)

    if phq_total >= 10:
        reasons.append("higher mood symptom burden")
    if user_input["sleep_hours"] < 6:
        reasons.append("short sleep duration")
    if user_input["daytime_sleepiness"] >= 2:
        reasons.append("daytime sleepiness")
    if user_input["med_count"] >= 5:
        reasons.append("polypharmacy")
    if (
        int(user_input["is_benzo"])
        + int(user_input["is_sedative"])
        + int(user_input["is_opioid"])
        + int(user_input["is_antipsychotic"])
        + int(user_input["is_anticonvulsant"])
        + int(user_input["is_muscle_relaxant"])
    ) >= 2:
        reasons.append("higher central nervous system medication burden")
    if int(user_input["anemia"]) == 1:
        reasons.append("anemia-related lab pattern")
    if int(user_input["high_rdw"]) == 1:
        reasons.append("abnormal red cell distribution width")

    return reasons[:4]


def plot_risk_bar(risk: float):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.barh([0], [100], height=0.42)
    ax.barh([0], [risk * 100], height=0.42)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Estimated risk (%)")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    st.pyplot(fig, clear_figure=True)


def plot_domain_bars(user_input: Dict[str, float]):
    import matplotlib.pyplot as plt

    phq_total = compute_phq_total(user_input)

    mood = min(phq_total / 27, 1.0) * 100
    sleep = min(((max(0, 6 - user_input["sleep_hours"])) + user_input["daytime_sleepiness"]) / 6, 1.0) * 100
    meds = min((user_input["med_count"] / 8), 1.0) * 100
    labs = (50 if int(user_input["anemia"]) == 1 else 0) + (25 if int(user_input["high_rdw"]) == 1 else 0)
    labs = min(labs, 100)

    labels = ["Mood", "Sleep", "Medications", "Labs"]
    values = [mood, sleep, meds, labs]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(labels, values)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Relative burden")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    st.pyplot(fig, clear_figure=True)


# =====================================================
# APP HEADER
# =====================================================
model, feature_names, threshold = load_artifacts()

st.title("Cognitive Limitation Risk Calculator")
st.write(
    "Estimate the likelihood of self-reported cognitive limitation based on mood symptoms, sleep patterns, medication burden, lab markers, and overall clinical profile."
)
st.caption(
    "This tool is for education and structured risk estimation. It does not diagnose dementia, memory disorders, or other neurologic conditions."
)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Before you begin")
st.write(
    "The questions below are grouped the way a clinical intake might be organized: mood and concentration, sleep, medications, and general clinical context."
)
st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FORM
# =====================================================
with st.form("clinical_risk_form"):
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Mood and concentration")
        input_help("Choose the response that best reflects how often each symptom has affected you over the last 2 weeks.")

        phq_little_interest = PHQ_OPTIONS[st.selectbox("Little interest or pleasure in doing things", list(PHQ_OPTIONS.keys()))]
        phq_low_mood = PHQ_OPTIONS[st.selectbox("Feeling down, depressed, or hopeless", list(PHQ_OPTIONS.keys()))]
        phq_sleep_change = PHQ_OPTIONS[st.selectbox("Sleeping more than usual or having trouble sleeping", list(PHQ_OPTIONS.keys()))]
        phq_low_energy = PHQ_OPTIONS[st.selectbox("Feeling tired or having low energy", list(PHQ_OPTIONS.keys()))]
        phq_appetite_change = PHQ_OPTIONS[st.selectbox("Poor appetite or overeating", list(PHQ_OPTIONS.keys()))]
        phq_feeling_bad = PHQ_OPTIONS[st.selectbox("Feeling bad about yourself", list(PHQ_OPTIONS.keys()))]
        phq_concentration = PHQ_OPTIONS[st.selectbox("Trouble concentrating on reading, watching TV, or similar tasks", list(PHQ_OPTIONS.keys()))]
        phq_slow_restless = PHQ_OPTIONS[st.selectbox("Moving or speaking slowly, or feeling unusually restless", list(PHQ_OPTIONS.keys()))]
        phq_self_harm = PHQ_OPTIONS[st.selectbox("Thoughts that you would be better off dead or of hurting yourself", list(PHQ_OPTIONS.keys()))]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Sleep")
        input_help("Sleep duration and daytime sleepiness often travel with memory and concentration complaints.")
        sleep_hours = st.slider("Average hours of sleep on weekdays", 2.0, 14.0, 7.0, 0.5)
        daytime_sleepiness = SLEEPINESS_OPTIONS[
            st.selectbox("How often do you feel overly sleepy during the day?", list(SLEEPINESS_OPTIONS.keys()))
        ]
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Medications")
        input_help("Medication burden can matter more than a single medication. Include medications currently used on a regular basis.")
        med_count = st.slider("Total number of prescription medications", 0, 20, 0, 1)

        is_antidepressant = st.checkbox("Antidepressant")
        st.caption(MEDICATION_CLASS_HELP["Antidepressant"])

        is_benzo = st.checkbox("Benzodiazepine")
        st.caption(MEDICATION_CLASS_HELP["Benzodiazepine"])

        is_antipsychotic = st.checkbox("Antipsychotic")
        st.caption(MEDICATION_CLASS_HELP["Antipsychotic"])

        is_sedative = st.checkbox("Sleep medication or sedative")
        st.caption(MEDICATION_CLASS_HELP["Sleep medication or sedative"])

        is_anticholinergic = st.checkbox("Anticholinergic medication")
        st.caption(MEDICATION_CLASS_HELP["Anticholinergic medication"])

        is_opioid = st.checkbox("Opioid pain medication")
        st.caption(MEDICATION_CLASS_HELP["Opioid pain medication"])

        is_anticonvulsant = st.checkbox("Seizure or nerve pain medication")
        st.caption(MEDICATION_CLASS_HELP["Seizure or nerve pain medication"])

        is_muscle_relaxant = st.checkbox("Muscle relaxant")
        st.caption(MEDICATION_CLASS_HELP["Muscle relaxant"])

        is_steroid = st.checkbox("Oral steroid")
        st.caption(MEDICATION_CLASS_HELP["Oral steroid"])

        is_stimulant = st.checkbox("Stimulant")
        st.caption(MEDICATION_CLASS_HELP["Stimulant"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Clinical profile")
        age = st.slider("Age", 18, 90, 45, 1)
        bmi = st.slider("Body mass index (BMI)", 15.0, 50.0, 26.0, 0.5)
        input_help("BMI reflects weight relative to height and can travel with sleep, cardiometabolic, and inflammatory burden.")

        sex_label = st.selectbox("Sex", list(SEX_TO_MODEL.keys()))
        st.caption("This is mapped internally to the same binary feature used in the trained model: male = 1, female = 0.")

        income_label = st.selectbox("Relative income level", list(INCOME_OPTIONS.keys()))
        st.caption("Income ratio reflects household income relative to family size and poverty thresholds. It is included as a broad context variable.")

        anemia = st.checkbox("History or lab pattern consistent with anemia")
        high_rdw = st.checkbox("Known abnormal red cell distribution width (RDW)")
        st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Estimate risk", use_container_width=True)

# =====================================================
# RESULT
# =====================================================
if submitted:
    user_input = {
        "phq_little_interest": phq_little_interest,
        "phq_low_mood": phq_low_mood,
        "phq_sleep_change": phq_sleep_change,
        "phq_low_energy": phq_low_energy,
        "phq_appetite_change": phq_appetite_change,
        "phq_feeling_bad": phq_feeling_bad,
        "phq_concentration": phq_concentration,
        "phq_slow_restless": phq_slow_restless,
        "phq_self_harm": phq_self_harm,
        "sleep_hours": sleep_hours,
        "daytime_sleepiness": daytime_sleepiness,
        "med_count": med_count,
        "is_antidepressant": int(is_antidepressant),
        "is_benzo": int(is_benzo),
        "is_antipsychotic": int(is_antipsychotic),
        "is_sedative": int(is_sedative),
        "is_anticholinergic": int(is_anticholinergic),
        "is_opioid": int(is_opioid),
        "is_anticonvulsant": int(is_anticonvulsant),
        "is_muscle_relaxant": int(is_muscle_relaxant),
        "is_steroid": int(is_steroid),
        "is_stimulant": int(is_stimulant),
        "bmi": bmi,
        "age": age,
        "sex_male": SEX_TO_MODEL[sex_label],
        "income_ratio": INCOME_OPTIONS[income_label],
        "anemia": int(anemia),
        "high_rdw": int(high_rdw),
    }

    X_user = build_features(user_input, feature_names)

    risk = float(model.predict_proba(X_user)[:, 1][0])
    category = risk_category(risk)
    color = risk_color(risk)
    reasons = explain_domains(user_input)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    col_a, col_b = st.columns([1.1, 1])

    with col_a:
        st.subheader("Estimated risk")
        st.markdown(f"<h1 style='margin-bottom:0;color:{color};'>{risk*100:.1f}%</h1>", unsafe_allow_html=True)
        st.write(f"**Risk category:** {category}")

        if risk >= threshold:
            st.write("This profile falls in a range associated with elevated likelihood of self-reported cognitive limitation.")
        else:
            st.write("This profile falls below the model's elevated-risk flag threshold.")

        if reasons:
            st.write("**Main contributing domains**")
            for reason in reasons:
                st.write(f"• {reason}")

    with col_b:
        plot_risk_bar(risk)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Clinical burden overview")
    st.caption("This chart is a simple summary of input burden across major domains. It is a structured overview, not a diagnostic breakdown of the model.")
    plot_domain_bars(user_input)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("How to interpret this result")
    st.write(
        "A higher score means the profile is more similar to people in the training data who reported being limited by memory or confusion symptoms. This tool should be used as a structured estimate, not a diagnosis."
    )
    st.markdown("</div>", unsafe_allow_html=True)
