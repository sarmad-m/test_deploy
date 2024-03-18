import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoost model
pickle_in = open("classifier.pkl", "rb")
xgb_cl = pickle.load(pickle_in)

# Define the Streamlit app interface
def main():
    st.title("Depression Prediction App")

    # Add input fields for features
    pr = st.number_input("Pr")
    frate = st.number_input("Frate")
    favrg = st.number_input("Favrg")
    time = st.number_input("Time")
    vtotal = st.number_input("Vtotal")
    fmax = st.number_input("Fmax")
    tmax = st.number_input("Tmax")
    sno = st.number_input("SNO")

    # Create a button to trigger prediction
    if st.button("Predict"):
        # Make prediction
        features = np.array([[pr, frate, favrg, time, vtotal, fmax, tmax, sno]])
        prediction = xgb_cl.predict(features)[0]
        
        # Display prediction
        if prediction == 0:
            st.write("Predicted class: Benign Prostatic Hyperplasia (BPH)")
        elif prediction == 1:
            st.write("Predicted class: Benign Prostatic Hyperplasia (Benign Prostatic Hyperplasia (BPH))")
        elif prediction == 2:
            st.write("Predicted class: Bladder tightening procedure")
        elif prediction == 3:
            st.write("Predicted class: Cystectomy")
        elif prediction == 4:
            st.write("Predicted class: Decreased detrusor activity")
        elif prediction == 5:
            st.write("Predicted class: Interstitial cystitis")
        elif prediction == 6:
            st.write("Predicted class: Neurogenic bladder")
        elif prediction == 7:
            st.write("Predicted class: Normal urine flow")
        elif prediction == 8:
            st.write("Predicted class: Prostatectomy")
        elif prediction == 9:
            st.write("Predicted class: Ureter correction")
        elif prediction == 10:
            st.write("Predicted class: Urethral obstruction")
        elif prediction == 11:
            st.write("Predicted class: Urethral stricture")
        elif prediction == 12:
            st.write("Predicted class: Urinary incontinence surgery")
        elif prediction == 13:
            st.write("Predicted class: Vesicoureteral reflux")

if __name__ == "__main__":
    main()
