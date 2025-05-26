"""
Handles the end-to-end prediction pipeline for stroke risk assessment.

This module is responsible for:
1. Loading pre-trained machine learning artifacts:
    - A BMI imputation pipeline (bmi_imputer_pipeline.pkl).
    - A final data preprocessor (final_preprocessor.pkl) which includes
      scaling and one-hot encoding.
    - The final trained stroke prediction model (final_stroke_model.pkl),
      which is a CatBoost classifier.
    - The optimal classification threshold (optimal_threshold.pkl) determined
      during model validation.
2. Defining helper functions for:
    - Imputing missing BMI values using the loaded BMI imputer.
    - Creating all necessary engineered features (log transformations, categorical bins,
      interaction terms) that the final preprocessor and model expect.
3. Providing a main prediction function `get_stroke_prediction(raw_patient_data)`:
    - Takes raw patient data as a dictionary.
    - Performs basic cleaning (e.g., handling 'Other' gender).
    - Applies BMI imputation.
    - Applies feature engineering.
    - Transforms the data using the loaded final preprocessor.
    - Obtains a stroke probability from the final CatBoost model.
    - Applies the optimal threshold to determine a binary prediction (stroke/no stroke).
    - Returns the prediction, probability, and threshold used.

The script includes an example usage block (`if __name__ == '__main__':`)
for direct testing with sample patient data. This module is intended to be
imported and used by other applications, such as a web service (e.g., Flask, Streamlit)
for deploying the stroke prediction model.
"""

import pickle
import os

import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

BMI_IMPUTER_PATH = os.path.join(ARTIFACTS_DIR, "bmi_imputer_pipeline.pkl")
FINAL_PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "final_preprocessor.pkl")
FINAL_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_stroke_model.pkl")
OPTIMAL_THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "optimal_threshold.pkl")

try:
    with open(BMI_IMPUTER_PATH, "rb") as f:
        bmi_imputer_pipeline = pickle.load(f)
    with open(FINAL_PREPROCESSOR_PATH, "rb") as f:
        final_data_preprocessor = pickle.load(f)
    with open(FINAL_MODEL_PATH, "rb") as f:
        final_model = pickle.load(f)
    with open(OPTIMAL_THRESHOLD_PATH, "rb") as f:
        OPTIMAL_THRESHOLD = pickle.load(f)
    print("All model artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Critical model artifact not found: {e}")
    print("Please ensure all .pkl files are in the correct location:", ARTIFACTS_DIR)
    bmi_imputer_pipeline = None
    final_data_preprocessor = None
    final_model = None
    OPTIMAL_THRESHOLD = 0.5

if not all([final_model, final_data_preprocessor, bmi_imputer_pipeline]):
    print("DEBUG model_pipeline: One or more artifacts are None after loading attempt.")

def impute_bmi_for_prediction(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing BMI values using the loaded bmi_imputer_pipeline.
    Expects raw features that bmi_imputer_pipeline was trained on.
    """
    if bmi_imputer_pipeline is None:
        raise RuntimeError("BMI Imputer pipeline not loaded.")

    df_processed = df_input.copy()
    missing_bmi_mask = df_processed["bmi"].isna()
    if missing_bmi_mask.any():
        predictor_columns_for_bmi = ['gender',
                                     'age',
                                     'hypertension',
                                     'heart_disease',
                                     'ever_married', 
                                     'work_type',
                                     'Residence_type',
                                     'avg_glucose_level',
                                     'smoking_status'
                                     ]
        features_for_prediction = df_processed.loc[missing_bmi_mask, predictor_columns_for_bmi]

        predicted_bmi_values = bmi_imputer_pipeline.predict(features_for_prediction)
        df_processed.loc[missing_bmi_mask, "bmi"] = predicted_bmi_values

    return df_processed

def create_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates all engineered features required by the final_data_preprocessor.
    """
    data = df.copy()

    if 'bmi' in data.columns:
        data["bmi_log"] = np.log1p(data["bmi"])
    else:
        raise ValueError("'bmi' column missing before log transformation.")

    if 'bmi' in data.columns:
        data["bmi_category"] = pd.cut(data["bmi"],
                                      bins=[0, 18.5, 24.9, 29.9, float('inf')],
                                      labels=["Underweight", "Normal",
                                              "Overweight", "Obese"],
                                      right=False)
    if 'age' in data.columns:
        data["age_group"] = pd.cut(data["age"],
                                   bins=[0, 18, 40, 60, float('inf')],
                                   labels=["Child", "Adult", "Middle_Aged", "Senior"],
                                   right=False)
    if 'avg_glucose_level' in data.columns:
        data['glucose_level_category'] = pd.cut(data['avg_glucose_level'],
                                                bins=[0, 100, 125, float('inf')],
                                                labels=['Normal_glucose',
                                                        'Prediabetes_glucose',
                                                        'Diabetes_glucose'],
                                                right=False)

    if 'hypertension' in data.columns and 'smoking_status' in data.columns:
        data["hypertension_and_smokes"] = ((data['hypertension'] == 1) &
                                           (data['smoking_status'] == 'smokes')).astype(int)
    if 'age' in data.columns and 'heart_disease' in data.columns:
        data["age_X_heart_disease"] = data["age"] * data["heart_disease"]

    return data

def get_stroke_prediction(raw_patient_data: dict) -> dict:
    """
    Takes raw patient data as a dictionary, processes it through the full pipeline,
    and returns the prediction and probability.
    """

    try:
        df_raw = pd.DataFrame([raw_patient_data])
    except Exception as e:
        return {"error": f"Could not create DataFrame from input: {e}"}

    expected_raw_cols = ['gender', 'age', 'hypertension',
                         'heart_disease', 'ever_married',
                         'work_type', 'Residence_type', 'avg_glucose_level',
                         'bmi', 'smoking_status']

    for col in expected_raw_cols:
        if col not in df_raw.columns:
            return {"error": f"Missing required input feature: {col}"}

    if 'gender' in df_raw.columns and "Other" in df_raw["gender"].unique():
        df_raw = df_raw[df_raw["gender"] != "Other"]
        if df_raw.empty:
            return {"prediction_label": 0, "probability_stroke": 0.0,
                    "info": "Input identified as 'Other' gender, excluded."}

    df_imputed = impute_bmi_for_prediction(df_raw)

    df_engineered = create_features_for_model(df_imputed)

    try:
        data_transformed_np = final_data_preprocessor.transform(df_engineered)
    except Exception as e:
        return {"error": f"Error during data preprocessing: {e}. "
                "Ensure input data matches training structure. "
                f"Columns in engineered data: {df_engineered.columns.tolist()}"}

    try:
        probabilities_all_classes = final_model.predict_proba(data_transformed_np)
        probability_stroke = probabilities_all_classes[0, 1]
    except Exception as e:
        return {"error": f"Error during model prediction: {e}"}

    predicted_label = 1 if probability_stroke >= OPTIMAL_THRESHOLD else 0

    return {
        "prediction_label": predicted_label,
        "probability_stroke": float(probability_stroke),
        "threshold_used": float(OPTIMAL_THRESHOLD),
        "info": "Prediction successful"
    }

if __name__ == '__main__':
    sample_patient_data_1 = {
        'gender': 'Female', 'age': 67.0, 'hypertension': 0, 'heart_disease': 1,
        'ever_married': 'Yes', 'work_type': 'Private', 'Residence_type': 'Urban',
        'avg_glucose_level': 228.69, 'bmi': 36.6, 'smoking_status': 'formerly smoked'
    }
    sample_patient_data_2 = {
        'gender': 'Male', 'age': 55.0, 'hypertension': 1, 'heart_disease': 0,
        'ever_married': 'Yes', 'work_type': 'Self-employed', 'Residence_type': 'Rural',
        'avg_glucose_level': 88.5, 'bmi': np.nan, 'smoking_status': 'smokes'
    }
    sample_patient_data_3 = {
        'gender': 'Female', 'age': 22.0, 'hypertension': 0, 'heart_disease': 0,
        'ever_married': 'No', 'work_type': 'Private', 'Residence_type': 'Urban',
        'avg_glucose_level': 75.0, 'bmi': 23.0, 'smoking_status': 'never smoked'
    }

    print("--- Testing Prediction Pipeline ---")
    for i, patient_data in enumerate([sample_patient_data_1,
                                      sample_patient_data_2,
                                      sample_patient_data_3]):
        print(f"\nInput Patient {i+1}: {patient_data}")
        result = get_stroke_prediction(patient_data)
        print(f"Output for Patient {i+1}: {result}")
