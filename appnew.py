
import joblib
import numpy as np
import streamlit as st
import pandas as pd 

# Load the trained model and preprocessing objects
model = joblib.load('Lung_Cancer_Prediction.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoders.pkl')

# Streamlit app title and description
st.title("Lung Cancer Prediction")
st.write("Enter the patient's information below to predict the likelihood of lung cancer.")

# User inputs for the Streamlit app
gender = st.selectbox("Gender", ["M", "F"])
age = st.slider("Age", 18, 100)
smoking = st.selectbox("Smoking Habit", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
yellow_fingers = st.selectbox("Yellow Fingers", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
anxiety = st.selectbox("Anxiety", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
peer_pressure = st.selectbox("Peer Pressure", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
chronic_disease = st.selectbox("Chronic Disease", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
fatigue = st.selectbox("Fatigue", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
allergy = st.selectbox("Allergy", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
wheezing = st.selectbox("Wheezing", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
alcohol_consuming = st.selectbox("Alcohol Consuming", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
coughing = st.selectbox("Coughing", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
shortness_of_breath = st.selectbox("Shortness of Breath", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
swallowing_difficulty = st.selectbox("Swallowing Difficulty", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
chest_pain = st.selectbox("Chest Pain", [1, 2], format_func=lambda x: "Yes" if x == 2 else "No")

# Process inputs for prediction

# Encode the gender input using the pre-trained encoder
encoded_gender = encoder.transform(pd.DataFrame([gender], columns=["GENDER"]))

# Combine all features (encoded gender + other features)
inputs = np.array([
    *encoded_gender[0],  # Encoded gender
    age,
    smoking,
    yellow_fingers,
    anxiety,
    peer_pressure,
    chronic_disease,
    fatigue,
    allergy,
    wheezing,
    alcohol_consuming,
    coughing,
    shortness_of_breath,
    swallowing_difficulty,
    chest_pain
]).reshape(1, -1)

# Scale the numerical inputs (excluding the categorical gender)
inputs[:, 1:] = scaler.transform(inputs[:, 1:])

# Make prediction
if st.button("Predict Lung Cancer"):               # button to predict result
    prediction = model.predict(inputs)
    st.write("Lung cancer prediction :", prediction[0] )
    
    
    

