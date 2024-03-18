import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = pickle.load(open("rf_model.pkl", "rb"))

# Define a function to preprocess new input data
def preprocess_data(data):
    le = LabelEncoder()
    data["class"] = le.fit_transform(data["class"])

    scaler = StandardScaler()
    columns_to_scale = ['Pr', 'Frate', 'Favrg', 'Time', 'Vtotal', 'Fmax', 'Tmax', 'SNO']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    return data.drop("class", axis=1)  # Drop 'class' column for prediction

# Create a Streamlit app
st.title("Urine Flowmeter Prediction App")

# Collect user input
with st.form("form"):
    Pr = st.slider("Pr", min_value=0, max_value=100, step=1)
    Frate = st.number_input("Frate")
    Favrg = st.number_input("Favrg")
    Time = st.number_input("Time")
    Vtotal = st.number_input("Vtotal")
    Fmax = st.number_input("Fmax")
    Tmax = st.number_input("Tmax")
    SNO = st.number_input("SNO")
    submitted = st.form_submit_button("Predict")

if submitted:
    # Preprocess input data
    new_data = pd.DataFrame([[Pr, Frate, Favrg, Time, Vtotal, Fmax, Tmax, SNO]], columns=['Pr', 'Frate', 'Favrg', 'Time', 'Vtotal', 'Fmax', 'Tmax', 'SNO'])
    processed_data = preprocess_data(new_data)

    # Make prediction
    prediction = model.predict(processed_data)[0]

    # Display the predicted class
    st.write("Predicted Class:", prediction)
