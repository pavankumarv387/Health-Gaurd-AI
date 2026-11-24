import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Disease Risk Prediction", page_icon="🩺", layout="centered")


@st.cache_resource
def load_models():
    return {(d, m): joblib.load(f"models/{d}_{m}.joblib")
            for d in ("diabetes", "heart") for m in ("lr", "rf")}


models = load_models()


def level(p):
    return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"


st.title("🩺 Combined Diabetes & Heart Disease Risk")
st.caption("Educational risk estimate, not a medical diagnosis. Consult a doctor.")

algo_name = st.radio("Model", ["Random Forest", "Logistic Regression"], horizontal=True)
algo = "rf" if algo_name == "Random Forest" else "lr"

st.subheader("Common details")
c1, c2 = st.columns(2)
age = c1.number_input("Age", 1, 100, 45)
sex = c2.selectbox("Sex", ["Male", "Female"])
glucose = st.number_input("Fasting glucose (mg/dL)", 40, 400, 110)

with st.expander("Diabetes details", expanded=True):
    a, b = st.columns(2)
    preg = a.number_input("Pregnancies", 0, 20, 0)
    dbp = b.number_input("Diastolic BP (mmHg)", 30, 140, 75)
    skin = a.number_input("Skin thickness (mm)", 5, 100, 25)
    insulin = b.number_input("Insulin (mu U/ml)", 0, 900, 80)
    bmi = a.number_input("BMI", 10.0, 70.0, 27.0)
    dpf = b.number_input("Diabetes pedigree function", 0.05, 2.5, 0.4)

with st.expander("Heart details", expanded=True):
    a, b = st.columns(2)
    cp = a.selectbox("Chest pain type (1-4)", [1, 2, 3, 4], index=3)
    sbp = b.number_input("Resting systolic BP (mmHg)", 80, 220, 130)
    chol = a.number_input("Cholesterol (mg/dL)", 100, 600, 220)
    ecg = b.selectbox("Resting ECG (0-2)", [0, 1, 2])
    hr = a.number_input("Max heart rate", 60, 220, 150)
    exang = b.selectbox("Exercise-induced angina", ["No", "Yes"])
    oldpeak = a.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0)
    slope = b.selectbox("ST slope (1-3)", [1, 2, 3], index=1)
    ca = a.selectbox("Major vessels (0-3)", [0, 1, 2, 3])
    thal = b.selectbox("Thalassemia (3 normal, 6 fixed, 7 reversible)", [3, 6, 7])

if st.button("Predict risk", type="primary"):
    # Diabetes model
    d_row = pd.DataFrame([[preg, glucose, dbp, skin, insulin, bmi, dpf, age]],
                         columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    p_d = models[("diabetes", algo)].predict_proba(d_row)[0, 1]

    # Heart model: fasting blood sugar is derived from the glucose entered above (the link)
    fbs = int(glucose > 120)
    h_cols = ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
              "fasting_blood_sugar", "resting_ecg", "max_heart_rate", "exercise_angina",
              "oldpeak", "slope", "num_major_vessels", "thalassemia"]

    def heart_prob(fbs_value):
        row = pd.DataFrame([[age, int(sex == "Male"), cp, sbp, chol, fbs_value, ecg, hr,
                             int(exang == "Yes"), oldpeak, slope, ca, thal]], columns=h_cols)
        return models[("heart", algo)].predict_proba(row)[0, 1]

    p_h = heart_prob(fbs)

    m1, m2 = st.columns(2)
    m1.metric("Diabetes risk", f"{p_d:.0%}", level(p_d))
    m2.metric("Heart disease risk", f"{p_h:.0%}", level(p_h))

    st.subheader("How diabetes (high blood sugar) affects heart risk")
    p_hi, p_lo = heart_prob(1), heart_prob(0)
    st.write(f"Same person, only blood sugar changed: heart risk is **{p_lo:.0%}** with normal "
             f"fasting sugar vs **{p_hi:.0%}** with high fasting sugar "
             f"(difference {p_hi - p_lo:+.0%}).")
    st.progress(min(max(p_hi, 0.0), 1.0))

    if p_d >= 0.5 and p_h >= 0.5:
        st.error("Elevated risk for BOTH conditions. Please consult a doctor soon.")
    elif p_d >= 0.5 or p_h >= 0.5:
        st.warning("Elevated risk for one condition. Consider a medical check-up.")
    else:
        st.success("Risk looks low on both, keep up regular check-ups and a healthy lifestyle.")

    st.caption("Diabetes and heart models are trained on separate datasets, so the link shown "
               "is an association learned by the model, not proof of cause.")
