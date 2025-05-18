
import streamlit as st
import pandas as pd
import joblib

clf = joblib.load("neet_selection_model.pkl")
reg = joblib.load("neet_score_model.pkl")
features = joblib.load("model_features.pkl")

st.title("NEET Selection Predictor")

study_hours = st.slider("Daily Study Hours", 1.0, 15.0, 6.0)
consistency = st.slider("Consistency (%)", 50, 100, 80)
subject = st.selectbox("Best Subject", ["Biology", "Physics", "Chemistry"])
institute = st.selectbox("Coaching Institute", [
    "Allen Career Institute", "Aakash Institute", "Resonance",
    "Narayana", "FIITJEE", "Motion Education",
    "Unacademy", "PW (Physics Wallah)", "BYJU'S", "Career Point"
])

input_dict = {
    'Daily_Study_Hours': [study_hours],
    'Consistency (%)': [consistency],
    'Subject_Strength_Biology': [1 if subject == "Biology" else 0],
    'Subject_Strength_Chemistry': [1 if subject == "Chemistry" else 0],
    'Subject_Strength_Physics': [1 if subject == "Physics" else 0],
}
for inst in ["Allen Career Institute", "Aakash Institute", "Resonance", "Narayana", "FIITJEE",
             "Motion Education", "Unacademy", "PW (Physics Wallah)", "BYJU'S", "Career Point"]:
    input_dict[f"Coaching_Institute_{inst}"] = [1 if institute == inst else 0]

for col in features:
    if col not in input_dict:
        input_dict[col] = [0]

input_df = pd.DataFrame(input_dict)
input_df = input_df[features]  # Align columns with model training

if st.button("Predict"):
    selection = clf.predict(input_df)[0]
    expected_score = reg.predict(input_df)[0]

    st.markdown(f"🎯 **Prediction**: {'✅ Likely to be Selected' if selection else '❌ Not Likely'}")
    st.markdown(f"📊 **Expected NEET Marks**: {round(expected_score)} / 720")
