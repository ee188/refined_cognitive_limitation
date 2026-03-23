import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(
    page_title="Brain Fog Risk Estimate",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------------
# Styling
# -----------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.block-container {
    max-width: 780px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    letter-spacing: -0.01em;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 650;
    margin-bottom: 0.35rem;
    color: #222;
}

.subtext {
    font-size: 1rem;
    line-height: 1.65;
    color: #666;
    margin-bottom: 1.5rem;
}

.section-label {
    font-size: 0.92rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #888;
    margin-top: 2rem;
    margin-bottom: 0.75rem;
    font-weight: 600;
}

.divider {
    border-top: 1px solid #ececec;
    margin: 1.5rem 0 0.5rem 0;
}

.note {
    color: #666;
    font-size: 0.96rem;
    line-height: 1.6;
}

.result-wrap {
    margin-top: 1rem;
    padding-top: 0.25rem;
    padding-bottom: 0.25rem;
}

.result-title {
    font-size: 1.35rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
}

.result-prob {
    font-size: 1rem;
    color: #555;
    margin-bottom: 0.6rem;
}

.low {
    color: #2e7d32;
}

.moderate {
    color: #9a6700;
}

.high {
    color: #b42318;
}

.small {
    color: #777;
    font-size: 0.9rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Load model and columns
# -----------------------------------
model = joblib.load("brain_fog_model_v3.pkl")
model_columns = joblib.load("brain_fog_model_columns_v3.pkl")

# -----------------------------------
# Helpers
# -----------------------------------
def map_income_category(cat: str) -> float:
    mapping = {
        "Below poverty line": 0.8,
        "Around poverty line": 1.0,
        "Lower income": 1.5,
        "Middle income": 2.5,
        "Higher income": 4.0
    }
    return mapping[cat]

def map_phq_category(cat: str) -> int:
    mapping = {
        "Minimal (0–4)": 2,
        "Mild (5–9)": 7,
        "Moderate (10–14)": 12,
        "Moderately severe (15–19)": 17,
        "Severe (20–27)": 23
    }
    return mapping[cat]

def build_input_df(
    age,
    income_ratio,
    bmi,
    sleep_hours,
    phq9_total,
    med_count,
    sex,
    education,
    is_benzo,
    is_antidepressant,
    is_antipsychotic,
    is_sedative,
    is_anticholinergic,
    is_opioid,
    is_anticonvulsant,
    is_muscle_relaxant,
    is_steroid,
    is_stimulant
):
    polypharmacy = 1 if med_count >= 5 else 0
    short_sleep = 1 if sleep_hours < 7 else 0
    sex_male = 1 if sex == "Male" else 0

    input_data = {
        "age": age,
        "income_ratio": income_ratio,
        "bmi": bmi,
        "sleep_hours": sleep_hours,
        "short_sleep": short_sleep,
        "phq9_total": phq9_total,
        "med_count": med_count,
        "polypharmacy": polypharmacy,
        "is_benzo": int(is_benzo),
        "is_antidepressant": int(is_antidepressant),
        "is_antipsychotic": int(is_antipsychotic),
        "is_sedative": int(is_sedative),
        "is_anticholinergic": int(is_anticholinergic),
        "is_opioid": int(is_opioid),
        "is_anticonvulsant": int(is_anticonvulsant),
        "is_muscle_relaxant": int(is_muscle_relaxant),
        "is_steroid": int(is_steroid),
        "is_stimulant": int(is_stimulant),
        "sex_Male": sex_male,
        "education_2.0": 1 if education == "9th–11th grade" else 0,
        "education_3.0": 1 if education == "High school / GED" else 0,
        "education_4.0": 1 if education == "Some college / Associate degree" else 0,
        "education_5.0": 1 if education == "College graduate or above" else 0,
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    return input_df

def risk_label_and_message(prob: float):
    if prob < 0.20:
        return (
            "Low likelihood",
            "low",
            "The model estimates a relatively low likelihood of reported confusion or memory difficulty."
        )
    elif prob < 0.50:
        return (
            "Moderate likelihood",
            "moderate",
            "The model estimates a moderate likelihood of reported confusion or memory difficulty."
        )
    else:
        return (
            "High likelihood",
            "high",
            "The model estimates a higher likelihood of reported confusion or memory difficulty."
        )

# -----------------------------------
# Header
# -----------------------------------
st.markdown('<div class="main-title">Brain Fog Risk Estimate</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtext">This educational tool estimates the likelihood of reported confusion or memory difficulty using health, sleep, mood, and medication-related information. It is based on a machine learning model trained on public survey data.</div>',
    unsafe_allow_html=True
)

with st.expander("About this tool"):
    st.write(
        "This is an educational prototype. It does not diagnose brain fog, identify a cause, or recommend treatment."
    )
    st.write(
        "The result reflects a statistical pattern found in survey data and should not be used as medical advice."
    )

# -----------------------------------
# Main form
# -----------------------------------
st.markdown('<div class="section-label">Basic information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 80, 40)
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
    bmi = st.slider("BMI", 14.0, 60.0, 28.0, 0.1)

with col2:
    education = st.selectbox(
        "Highest education level",
        [
            "Less than 9th grade",
            "9th–11th grade",
            "High school / GED",
            "Some college / Associate degree",
            "College graduate or above"
        ]
    )
    income_category = st.selectbox(
        "Household income level",
        [
            "Below poverty line",
            "Around poverty line",
            "Lower income",
            "Middle income",
            "Higher income"
        ]
    )
    med_count = st.slider("Number of current medications", 0, 22, 1)

income_ratio = map_income_category(income_category)

st.markdown('<div class="section-label">Sleep and mood</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    sleep_hours = st.slider("Average sleep per night (hours)", 2.0, 14.0, 7.0, 0.5)
with col4:
    phq_category = st.selectbox(
        "Depression symptom level",
        [
            "Minimal (0–4)",
            "Mild (5–9)",
            "Moderate (10–14)",
            "Moderately severe (15–19)",
            "Severe (20–27)"
        ]
    )

phq9_total = map_phq_category(phq_category)

st.markdown(
    '<div class="note">PHQ-9 is a common depression symptom questionnaire. Higher categories reflect greater symptom burden.</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="section-label">Medication classes</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    is_benzo = st.checkbox("Benzodiazepine")
    is_antidepressant = st.checkbox("Antidepressant")
    is_antipsychotic = st.checkbox("Antipsychotic")
    is_sedative = st.checkbox("Sedative / sleep medication")
    is_anticholinergic = st.checkbox("Anticholinergic medication")

with c2:
    is_opioid = st.checkbox("Opioid pain medication")
    is_anticonvulsant = st.checkbox("Anticonvulsant / nerve pain medication")
    is_muscle_relaxant = st.checkbox("Muscle relaxant")
    is_steroid = st.checkbox("Steroid")
    is_stimulant = st.checkbox("Stimulant")

with st.expander("Examples of medication classes"):
    st.write("**Benzodiazepines:** lorazepam, alprazolam, clonazepam")
    st.write("**Antidepressants:** sertraline, fluoxetine, duloxetine")
    st.write("**Antipsychotics:** quetiapine, olanzapine, risperidone")
    st.write("**Sedatives / sleep meds:** zolpidem, eszopiclone")
    st.write("**Anticholinergics:** diphenhydramine, oxybutynin")
    st.write("**Opioids:** hydrocodone, tramadol, oxycodone")
    st.write("**Anticonvulsants / nerve pain meds:** gabapentin, pregabalin")
    st.write("**Muscle relaxants:** cyclobenzaprine, methocarbamol")
    st.write("**Steroids:** prednisone, methylprednisolone")
    st.write("**Stimulants:** methylphenidate, amphetamine")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

predict_clicked = st.button("See my result", use_container_width=True)

# -----------------------------------
# Result
# -----------------------------------
if predict_clicked:
    input_df = build_input_df(
        age=age,
        income_ratio=income_ratio,
        bmi=bmi,
        sleep_hours=sleep_hours,
        phq9_total=phq9_total,
        med_count=med_count,
        sex=sex,
        education=education,
        is_benzo=is_benzo,
        is_antidepressant=is_antidepressant,
        is_antipsychotic=is_antipsychotic,
        is_sedative=is_sedative,
        is_anticholinergic=is_anticholinergic,
        is_opioid=is_opioid,
        is_anticonvulsant=is_anticonvulsant,
        is_muscle_relaxant=is_muscle_relaxant,
        is_steroid=is_steroid,
        is_stimulant=is_stimulant,
    )

    prob = model.predict_proba(input_df)[0][1]
    risk_label, color_class, message = risk_label_and_message(prob)

    st.markdown('<div class="section-label">Your result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="result-wrap">
            <div class="result-title {color_class}">{risk_label}</div>
            <div class="result-prob">Estimated probability: {prob:.1%}</div>
            <div class="note">{message}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="small">This estimate is based on survey patterns, not a clinical evaluation. It should not be used to diagnose memory problems or guide treatment decisions.</div>',
        unsafe_allow_html=True
    )

    with st.expander("Show the values used by the model"):
        st.dataframe(input_df, use_container_width=True)

else:
    st.markdown(
        '<div class="small">Enter values above and click <strong>See my result</strong>.</div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="small">Educational healthcare informatics prototype built with public NHANES survey data and machine learning.</div>',
    unsafe_allow_html=True
)
