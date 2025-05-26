"""
Streamlit web application for stroke risk prediction.

This application provides a user interface for users to input patient data
and receive a prediction regarding their likelihood of experiencing a stroke.
It utilizes a pre-trained machine learning pipeline (including BMI imputation,
feature engineering, data preprocessing, and a CatBoost classifier) loaded
from pickled artifact files.

The user inputs demographic, clinical, and lifestyle information, and the app
outputs a binary prediction (High Risk / Low Risk) along with the model's
estimated probability of stroke.
"""

import streamlit as st
import numpy as np

try:
    from model_pipeline import get_stroke_prediction
    MODEL_LOADED_SUCCESSFULLY = True
except ImportError as e:
    st.error(f"Error importing prediction pipeline: {e}. "
             "Make sure model_pipeline.py is in the same directory.")
    MODEL_LOADED_SUCCESSFULLY = False
except Exception as e:
    st.error(f"Error loading model artifacts in model_pipeline.py: {e}")
    MODEL_LOADED_SUCCESSFULLY = False

st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")
st.title(" Stroke Risk Prediction App ")
st.markdown("""
This app predicts the likelihood of a patient experiencing a stroke based on their input parameters. 
Please fill in the patient's details.
""")

if not MODEL_LOADED_SUCCESSFULLY:
    st.warning("The prediction model could not be loaded. Please check the application setup.")
else:
    st.sidebar.header("Patient Input Features")

    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    ever_married = st.sidebar.selectbox("Ever Married?", ["Yes", "No"])
    work_type_options = ["Private", "Self-employed", "Govt_job", "children",
                         "Never_worked"]
    work_type = st.sidebar.selectbox("Work Type", work_type_options)
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status_options = ["formerly smoked", "never smoked",
                              "smokes", "Unknown"]
    smoking_status = st.sidebar.selectbox("Smoking Status", smoking_status_options)

    age = st.sidebar.number_input("Age (in years)", min_value=0.0,
                                  max_value=120.0, value=50.0, step=1.0)
    avg_glucose_level = st.sidebar.number_input("Average Glucose Level (mg/dL)",
                                                min_value=0.0, value=100.0, step=0.1)
    bmi_input_str = st.sidebar.text_input("BMI (Body Mass Index) - "
                                          "leave blank if unknown", value="28.0")

    hypertension = st.sidebar.radio("Hypertension?", (0, 1), format_func=lambda x:
                                    "Yes" if x == 1 else "No", index=0)
    heart_disease = st.sidebar.radio("Heart Disease?", (0, 1), format_func=lambda x:
                                     "Yes" if x == 1 else "No", index=0)

    if st.sidebar.button("Predict Stroke Risk"):
        if bmi_input_str.strip() == "" or bmi_input_str.lower() == "nan":
            bmi = np.nan
        else:
            try:
                bmi = float(bmi_input_str)
                if bmi <= 0:
                    st.error("BMI must be a positive number.")
                    st.stop()
            except ValueError:
                st.error("Invalid BMI input. Please enter a number or leave blank.")
                st.stop()

        patient_data = {
            'gender': gender,
            'age': float(age),
            'hypertension': int(hypertension),
            'heart_disease': int(heart_disease),
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': float(avg_glucose_level),
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        st.subheader("Input Data Summary:")
        st.json(patient_data)

        with st.spinner("Predicting..."):
            prediction_output = get_stroke_prediction(patient_data)

        st.subheader("Prediction Result:")
        if "error" in prediction_output:
            st.error(f"Error during prediction: {prediction_output['error']}")
        else:
            prediction_label = prediction_output['prediction_label']
            probability_stroke = prediction_output['probability_stroke']
            threshold_used = prediction_output['threshold_used']

            if prediction_label == 1:
                st.error("**Prediction: High Risk of Stroke** (Probability: "
                         f"{probability_stroke:.2%})")
                st.warning("This prediction was made using a probability "
                           f"threshold of {threshold_used:.2f}. "
                           "A probability above this threshold is considered high risk.")
            else:
                st.success("**Prediction: Low Risk of Stroke** (Probability: "
                           f"{probability_stroke:.2%})")
                st.info("This prediction was made using a probability "
                        f"threshold of {threshold_used:.2f}. "
                         "A probability below this threshold is considered low risk.")

            st.markdown("---")
