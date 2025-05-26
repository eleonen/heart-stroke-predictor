"""
Trains and saves a BMI imputation pipeline for the stroke prediction project.

This script performs the following steps:
1. Loads the raw stroke prediction dataset.
2. Performs an initial train-test split (80/20) and then a further train-validation 
    split (75/25 of the 80%) to isolate a dedicated training partition (60% of original data). 
    The BMI imputer is trained *only* on this 60% training partition to prevent data leakage.
3. From this training partition, it selects rows where BMI is not null.
4. Defines a preprocessing pipeline (ColumnTransformer) to one-hot encode
    categorical features relevant for BMI prediction, passing through numerical features.
5. Combines this preprocessor with a RandomForestRegressor in a scikit-learn Pipeline.
6. Fits this imputation pipeline to predict BMI based on other patient features.
7. Saves the fitted pipeline to 'bmi_imputer_pipeline.pkl' for later use in imputing
    missing BMI values in new or unseen data.
"""

import os

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_SCRIPT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT_SCRIPT, "data", "raw", "healthcare-dataset-stroke-data.csv")
ARTIFACTS_DIR_SCRIPT = os.path.join(PROJECT_ROOT_SCRIPT, "artifacts")

df = pd.read_csv(DATA_PATH).drop(columns=["id"])
df = df[df["gender"]!= "Other"]

X = df.drop(columns=["stroke"])
y = df["stroke"]

X_train_val, _, y_train_val, _ = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)

X_train, _, y_train, _ = train_test_split(X_train_val, y_train_val,
                                          test_size=0.25,
                                          random_state=42,
                                          stratify=y_train_val)

data_with_bmi = X_train.dropna(subset=["bmi"])

X_bmi_train = data_with_bmi.drop(columns=["bmi"])
y_bmi_train = data_with_bmi["bmi"]

categorical_features = ['gender', 'ever_married', 'work_type',
                        'Residence_type', 'smoking_status']
numerical_features = ['age', 'hypertension', 'heart_disease',
                      'avg_glucose_level']

encoder = OneHotEncoder(handle_unknown="ignore", drop="first",
                        sparse_output=False)

bmi_internal_preprocessor = ColumnTransformer([
    ("cat", encoder, categorical_features)
    ],
    remainder="passthrough")

bmi_imputation_pipeline = Pipeline([
    ("preprocessor", bmi_internal_preprocessor),
    ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))
])

print(f"Fitting BMI imputer on {X_bmi_train.shape[0]} samples with BMI.")
bmi_imputation_pipeline.fit(X_bmi_train, y_bmi_train)

os.makedirs(ARTIFACTS_DIR_SCRIPT, exist_ok=True)
bmi_imputer_save_path = os.path.join(ARTIFACTS_DIR_SCRIPT, "bmi_imputer_pipeline.pkl")
with open(bmi_imputer_save_path, "wb") as f:
    pickle.dump(bmi_imputation_pipeline, f)

print(f"BMI imputation pipeline saved successfully to {bmi_imputer_save_path}.")
print(f"It was trained on columns: {X_bmi_train.columns.tolist()}")
