import streamlit as st
import numpy as np
import pickle

# Load the saved model
iris = pickle.load(open('best_rf_model.pkl','rb'))

# Title and description
st.title('Predict Urine Flowmeter')
st.markdown('This app predicts the class label based on the input features.')

# Input features
st.header("Input Features")
Pr =  float(st.number_input("Pr"))
Frate = float(st.number_input("Frate"))
Favrg = float(st.number_input("Favrg"))
Time = float(st.number_input("Time"))
Vtotal = float(st.number_input("Vtotal"))
Fmax = float(st.number_input("Fmax"))
Tmax = float(st.number_input("Tmax"))
SNO = float(st.number_input("SNO"))

# Prediction button
if st.button("Predict"):
    # Make prediction
    input_data = np.array(['Pr',	'Frate',	'Favrg',	'Time',	'Vtotal',	'Fmax'	,'Tmax',	'SNO'])
    prediction = model.predict(input_data)
    st.write("Predicted Class:", prediction[0])

st.text('')
st.text('')
st.markdown(
