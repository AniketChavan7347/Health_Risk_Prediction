import streamlit as st
import joblib
import numpy as np
import pandas as pd     
import plotly.express as px


model = joblib.load(open("Healthcare Risk_predict.pkl", "rb"))
encoders = joblib.load(open("Label_encoders.pkl", "rb"))

st.title("🩺 Health Risk Predictor")


age = st.slider("Age", min_value=18, max_value=80, value=22)
diet = st.selectbox("Diet Quality", ['Poor', 'Average', 'Good'])
exercise = st.select_slider("Exercise days per week", options=list(range(8)), value=3)
sleep = st.slider("Sleep Hours", min_value=2, max_value=12, value=6)
stress = st.selectbox("Stress Level", ['Low', 'Medium', 'High'])
bmi = st.number_input("BMI", min_value=10.0, max_value=40.0, value=22.0, step=0.1)
smoking = st.selectbox("Smoking", ["Yes", "No"])
alcohol = st.selectbox("Alcohol Consumption", ['Low', 'Medium', 'High'])
family_history = st.selectbox("Family History of Disease", ["Yes", "No"])

encoded_input = [
    age,
    encoders['diet'].transform([diet])[0],
    exercise,
    sleep,
    encoders['stress'].transform([stress])[0],
    bmi,
    encoders['smoking'].transform([smoking])[0],
    encoders['alcohol'].transform([alcohol])[0],
    encoders['family_history'].transform([family_history])[0]
]

if st.button("Predict Risk"):

    prediction = model.predict([encoded_input])
    risk_label = encoders['risk_level'].inverse_transform(prediction)[0]

    
    if risk_label == "Low":
        st.success("🟢 Low Health Risk")
    elif risk_label == "Medium":
        st.warning("🟡 Medium Health Risk")
    else:
        st.error("🔴 High Health Risk")

    
    proba = model.predict_proba([encoded_input])[0]

   
    proba_dict = {
        "High": round(proba[0]*100,2),
        "Low": round(proba[1]*100,2),
        "Medium": round(proba[2]*100,2)
    }

    df_proba = pd.DataFrame({
        "Risk": list(proba_dict.keys()),
        "Probability": list(proba_dict.values())
    })

    fig_proba = px.pie(
        df_proba,
        names="Risk",
        values="Probability",
        color="Risk",
        color_discrete_map={"High":"red", "Medium":"orange", "Low":"green"},
        title="Risk Probability Distribution (%)"
    )
    st.plotly_chart(fig_proba, use_container_width=True)

    st.subheader("Health Recommendations")
    if risk_label == "High":
        st.write("• Improve diet quality")
        st.write("• Stop smoking")
        st.write("• Reduce alcohol consumption")
        st.write("• Increase physical activity")
        st.write("• Consult a healthcare professional")
    elif risk_label == "Medium":
        st.write("• Maintain balanced diet")
        st.write("• Exercise regularly")
        st.write("• Manage stress properly")
    else:
        st.write("• Continue healthy lifestyle")
        st.write("• Maintain regular exercise")

    factors = {
        "Diet": encoders['diet'].transform([diet])[0],
        "Exercise": exercise,
        "Sleep": sleep,
        "Stress": encoders['stress'].transform([stress])[0],
        "BMI": bmi
    }

    bar_fig = px.bar(
        x=list(factors.keys()),
        y=list(factors.values()),
        labels={"x": "Factors", "y": "Value"},
        title="Your Lifestyle Factors"
    )
    st.plotly_chart(bar_fig)