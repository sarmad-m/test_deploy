import streamlit as st
import numpy as np
import pickle

# Load the saved model
with open("best_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title and description
st.title('Predict Urine Flowmeter')
st.markdown('This app predicts the class label based on the input features.')

# Input features
st.header("Input Features")
Pr = st.slider("Pr", min_value=0, max_value=100, step=1)
Frate = st.number_input("Frate")
Favrg = st.number_input("Favrg")
Time = st.number_input("Time")
Vtotal = st.number_input("Vtotal")
Fmax = st.number_input("Fmax")
Tmax = st.number_input("Tmax")
SNO = st.number_input("SNO")

# Prediction button
if st.button("Predict"):
    # Make prediction
    input_data = np.array([['Pr',	'Frate',	'Favrg',	'Time',	'Vtotal',	'Fmax'	,'Tmax',	'SNO']])
    prediction = model.predict(input_data)
    st.write("Predicted Class:", prediction[0])

st.text('')
st.text('')
st.markdown(
    '`Initial code was developed by` [santiviquez](https://twitter.com/santiviquez) | \
         `Code modification and update by:` [Mohamed Alie](https://github.com/Kmohamedalie/iris-streamlit/tree/main)')
