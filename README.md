# Stroke Prediction Analysis

## Project Overview
This repository contains a comprehensive analysis of the "Stroke Prediction Dataset." The primary aim is to develop a machine learning model to predict the likelihood of a patient experiencing a stroke based on demographic and clinical data. The project involves detailed exploratory data analysis (EDA), statistical inference to validate risk factors, feature engineering, and the training and evaluation of multiple machine learning models, with a strong emphasis on interpretability and addressing class imbalance. The final model is then prepared for deployment.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Project Goal & Stakeholders](#project-goal--stakeholders)
3.  [Chosen Metric](#chosen-metric)
4.  [Dataset](#dataset)
5.  [Project Structure](#project-structure)
6.  [Setup](#setup)
7.  [Usage](#usage)
8.  [Analysis Workflow](#analysis-workflow)
9.  [Key Findings & Model Performance](#key-findings--model-performance)
10. [Model Deployment](#model-deployment)
11. [Limitations & Future Work](#limitations--future-work)
12. [Contributors](#contributors)

## Introduction
Stroke is a critical medical condition and a leading cause of death and disability globally. Early identification of individuals at high risk can significantly improve outcomes through timely preventative measures and emergency preparedness. This project leverages machine learning to build a predictive tool that can assist healthcare professionals in this vital task, using a publicly available dataset.

## Project Goal & Stakeholders

### The Goal:
- To develop a robust machine learning model capable of predicting the likelihood of a patient experiencing a stroke.
- To identify key demographic, clinical, and lifestyle factors that significantly contribute to stroke risk.
- To create a predictive tool that can support medical professionals (e.g., at The Johns Hopkins Hospital) in early risk assessment, enabling personalized patient counseling and proactive interventions.

### Stakeholders:
-   **Medical Doctors & Clinicians**: To aid in risk assessment, personalize patient counseling, and prioritize patients for preventative interventions or closer monitoring.
-   **Hospital Administration**: To potentially improve patient outcomes, optimize resource allocation for stroke prevention programs, and enhance the hospital's reputation for proactive care.
-   **Patients and their Families**: To receive early warnings and guidance, empowering them to take necessary actions and be better prepared.
-   **Public Health Researchers**: To gain further insights into stroke risk factors within the studied population.

## Chosen Metric

### Primary Metric: F2-Score
-   **Rationale**: In the context of stroke prediction, failing to identify a patient who will subsequently have a stroke (a **False Negative**) has far more severe consequences than incorrectly identifying a healthy patient as high-risk (a **False Positive**).
-   The **F2-score** is chosen as the primary metric because it weights recall twice as much as precision (beta=2), directly aligning with the critical need to minimize False Negatives and maximize the identification of true stroke cases.

### Secondary Metrics:
-   **Recall (Sensitivity)**: To directly track success in identifying true stroke cases.
-   **Precision**: To ensure predictions are not excessively noisy.
-   **Precision-Recall AUC (PR-AUC)**: Especially useful for imbalanced datasets, providing a threshold-independent measure of performance.
-   **ROC-AUC**: For a general measure of discriminative ability.
-   **Confusion Matrix**: For a detailed breakdown of prediction outcomes.

## Dataset

### Stroke Prediction Dataset
- **Source:** [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Description:** The dataset contains records of individuals with various attributes relevant to stroke risk. It includes demographic information, pre-existing conditions, lifestyle factors, and whether the individual had a stroke.
- **Key Columns:**
    - `gender`: "Male", "Female" or "Other"
    - `age`: Age of the patient
    - `hypertension`: 0 (no) or 1 (yes)
    - `heart_disease`: 0 (no) or 1 (yes)
    - `ever_married`: "No" or "Yes"
    - `work_type`: "children", "Govt_job", "Never_worked", "Private" or "Self-employed"
    - `Residence_type`: "Rural" or "Urban"
    - `avg_glucose_level`: Average glucose level in blood
    - `bmi`: Body mass index
    - `smoking_status`: "formerly smoked", "never smoked", "smokes" or "Unknown"
    - `stroke`: 1 (patient had a stroke) or 0 (not) - **Target Variable**.
- **Size:** Approximately 5110 entries and 11 features (after dropping `id`).

## Project Structure

```text
├── app/
│ ├── streamlit_app.py
│ └── model_pipeline.py
├── artifacts/
│ ├── bmi_imputer_pipeline.pkl
│ ├── final_preprocessor.pkl
│ ├── final_stroke_model.pkl
│ └── optimal_threshold.pkl
├── data/
│ └── raw/
│ └── healthcare-dataset-stroke-data.csv
├── notebooks
│ └── stroke_risk_analysis.ipynb
├── scripts/
│ ├── bmi_imputer.py
│ └── utilities.py
├── .gitignore
├── .dockerignore
├── Dockerfile
├── poetry.lock
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Setup

### Prerequisites
- Python 3.11
- Poetry (for dependency management)
- Docker (for containerization and local testing of the deployed app)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/eleonen/heart-stroke-predictor
    cd heart-stroke-predictor
    ```
2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```
3.  **Activate the virtual environment:**
    ```bash
    poetry shell
    ```

## Usage

### 1. Running the Jupyter Notebook for Analysis (`stroke_risk_analysis.ipynb`):
This notebook contains the full data analysis, EDA, statistical inference, model training, and evaluation.
1.  Ensure you have Jupyter Notebook or JupyterLab installed in your Poetry environment:
    ```bash
    poetry add jupyter notebook
    ```
2.  Start Jupyter:
    ```bash
    jupyter notebook notebooks/stroke_risk_analysis.ipynb 
    ```

### 2. Training the BMI Imputer:
1.  The BMI imputer model artifact is generated by a separate script:
    ```bash
    python scripts/bmi_imputer.py
    ```
2.  This will create/update artifacts/bmi_imputer_pipeline.pkl.

### 3. Running the Streamlit Application Locally (using Docker):
The application uses the pre-trained model artifacts for prediction.

1.  Build the Docker image:
    ```bash
    docker build -t stroke-predictor-app .
    ```
2.  Run the Docker container:
    ```bash
    docker run -p 8501:8501 -it --rm stroke-predictor-app
    ```
3.  Open your web browser and go to: http://localhost:8501

### 4. Running the Prediction Pipeline Script Directly (for testing):
1.  You can test the core prediction logic by running model_pipeline.py.
    ```bash
    python app/model_pipeline.py
    ```
2.  This will use the sample data defined within its if __name__ == '__main__': block.

## Analysis Workflow

The project follows a structured data science workflow:

- **Data Loading and Initial Inspection**: Understanding data types, shapes, and initial quality.
- **Exploratory Data Analysis (EDA)**:
  - Handling missing values (BMI imputation using RandomForest).
  - Analyzing class imbalance of the `stroke` variable.
  - Visualizing feature distributions (histograms, boxplots) segmented by stroke outcome.
  - Investigating correlations between numerical features.
  - Analyzing relationships between categorical features and stroke incidence using cross-tabulations.
- **Statistical Inference**:
  - Formulating hypotheses regarding key risk factors (age, glucose, BMI, heart disease, marital status, gender).
  - Conducting t-tests and Chi-Square tests to assess statistical significance.
- **Feature Engineering & Preprocessing for Modeling**:
  - Creating new features: log transformations (`bmi_log`, `avg_glucose_level_log`), categorical bins (`age_group`, `bmi_category`, `glucose_level_category`), and interaction terms (`age_X_heart_disease`, `hypertension_and_smokes`).
  - Applying a ColumnTransformer pipeline for one-hot encoding categorical features and scaling numerical features.
- **Feature Selection**:
  - Using Permutation Importance (with RandomForest and `average_precision` scoring) and Recursive Feature Elimination (RFE with Logistic Regression) to identify the most impactful features.
  - Selecting a final feature set based on these analyses, dropping features like raw `gender`, `Residence_type`, and continuous `avg_glucose_level_log` in favor of its categorical representation.
- **Model Training and Evaluation**:
  - Establishing baselines: Dummy Classifier, a simple Rule-Based predictor, and Logistic Regression (with VIF checks and feature reduction for LR).
  - Training and tuning advanced models: RandomForest, XGBoost, LightGBM, and CatBoost using GridSearchCV with 5-fold cross-validation, optimizing for the F2-score.
  - Optimizing probability thresholds for each model to maximize F2-score on the validation set.
- **Final Model Selection & Test Set Evaluation**:
  - Selecting the best model (CatBoost) based on validation performance.
  - Retraining the selected model on the combined training and validation data.
  - Evaluating the final model on the unseen test set.
- **Model Interpretability**:
  - Using SHAP values to understand feature contributions for the final CatBoost model.

## Key Findings & Model Performance

### Baselines:
- **Dummy Classifier** (Stratified): F2-score of 0.0826.
- **Simple Rule-Based Predictor**: F2-score of 0.3292.
- **Logistic Regression** (VIF-reduced features): F2-score of 0.4032, PR-AUC 0.1630, ROC-AUC 0.8227. High recall (0.90) but low precision (0.13).

### Advanced Models (Validation Set, Optimal Threshold):
- **CatBoost**: **F2-score 0.4111**, Recall 0.74, Precision 0.1480, PR-AUC 0.1776, ROC-AUC 0.8367. (Optimal Threshold: 0.59)
- **RandomForest**: F2-score 0.4008, Recall 0.80, Precision 0.1338.
- **XGBoost**: F2-score 0.3958, Recall 0.82, Precision 0.1289.
- **LightGBM**: F2-score 0.3920, Recall 0.90, Precision 0.1203.

### Final Model (CatBoost) Performance on Test Set:
- **F2-Score: 0.4283**
- **Recall (Stroke): 0.8** (80% of actual strokes identified)
- **Precision (Stroke): 0.15**
- **PR-AUC: 0.2567**
- **ROC-AUC: 0.8429**
- **Key SHAP Features**: `age`, log-transformed `BMI`, `hypertension`, `smoking_status_never smoked`, `glucose_level_category_Normal_glucose`, `age_group_Senior`.

**Conclusion**: The CatBoost model, after comprehensive preprocessing, feature engineering, and tuning, demonstrated the best performance among the machine learning models. It achieved an F2-score of 0.4283 on the test set, successfully identifying 80% of stroke cases, aligning with the project's goal of prioritizing high recall.

## Model Deployment
The trained model and preprocessing pipeline are saved as pickle files in the `artifacts/` directory.

A Streamlit application (`app/streamlit_app.py`) provides a user interface for making predictions using this pipeline. The application is Dockerized via the Dockerfile.

**Live Demo:**
The application is deployed on Google Cloud Run and can be accessed at:
[https://stroke-predictor-686453789734.europe-west1.run.app](https://stroke-predictor-686453789734.europe-west1.run.app)

*Please note: To manage operational costs associated with cloud hosting, this live demo service may not be available 24/7. If the link is unresponsive, the service might be temporarily paused.*

## Limitations & Future Work

### Limitations:
- The dataset has a significant class imbalance for the target variable `stroke`.
- Missing BMI values were imputed, which might not perfectly reflect true BMI values.
- The "Unknown" category in `smoking_status` limits a full understanding of smoking's impact.
- The dataset's representativeness of a wider global population is not guaranteed.

### Future Work:
- **Advanced Imbalance Handling**: Explore techniques like SMOTE or other over/under-sampling methods more deeply within cross-validation.
- **More Extensive Hyperparameter Tuning**: Utilize tools like Optuna for more sophisticated hyperparameter searches.
- **External Validation**: Test the model on a completely different dataset from another source or hospital to assess true generalizability.
- **Investigate Feature Interactions Further**: Use SHAP interaction values or other techniques to discover more complex relationships.

## Contributors
- [Erikas Leonenka](https://github.com/eleonen)
