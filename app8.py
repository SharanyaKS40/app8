import streamlit as st
import pandas as pd
import pickle
import os

base_path = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(base_path, "attrition_model88.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(base_path, "scaler 88.pkl"), "rb"))
columns = pickle.load(open(os.path.join(base_path, "columns88.pkl"), "rb"))

st.title("HR Attrition Prediction App")

age = st.number_input("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 2)
work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 2)
years_at_company = st.number_input("Years at Company", 0, 40, 5)
overtime = st.selectbox("OverTime", ["Yes", "No"])

input_dict = {
    "Age": age,
    "MonthlyIncome": monthly_income,
    "JobSatisfaction": job_satisfaction,
    "WorkLifeBalance": work_life_balance,
    "YearsAtCompany": years_at_company,
    "OverTime_Yes": 1 if overtime == "Yes" else 0,
}

input_df = pd.DataFrame([input_dict])

for col in columns:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[columns]
input_scaled = scaler.transform(input_df)

if st.button("Predict Attrition"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("Employee likely to leave")
    else:
        st.success("Employee likely to stay")
