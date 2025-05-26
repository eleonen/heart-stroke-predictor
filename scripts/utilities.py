"""
Utility functions for the stroke prediction project.

This module provides helper functions for various stages of the data science
pipeline, including data cleaning, exploratory data analysis (EDA) tasks like
finding missing values and outliers, plotting, statistical testing,
feature engineering, and model evaluation.
"""

from typing import List, Tuple, Dict, Any

import pickle
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, Markdown

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    average_precision_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
from scipy.stats import shapiro, levene, ttest_ind, chi2_contingency

UTILITIES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_UTIL = os.path.dirname(UTILITIES_DIR)
DEFAULT_ARTIFACTS_DIR_UTIL = os.path.join(PROJECT_ROOT_UTIL, "artifacts")

def find_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and counts missing values in a DataFrame,
    including zeroes, empty strings, and NaN values.

    Args:
        df: The DataFrame to analyze.

    Returns:
        A DataFrame with counts of zeroes, empty strings,
        and NaN values for each column.
    """
    zeroes = (df == 0).sum()
    empty_strings = (df.replace(r"^\s*$", "", regex=True) == "").sum()
    nas = df.isna().sum()
    combined_counts = pd.DataFrame(
        {"Zeroes": zeroes, "Empty Strings": empty_strings, "NaN": nas}
    )
    return combined_counts


def find_outliers(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Detects outliers in multiple features using the IQR method.

    Args:
        df: DataFrame containing the data.
        features: List of features to detect outliers in.

    Returns:
        DataFrame containing the outliers for each feature and a DataFrame
        containing analysis for each feature (outlier count, percentage, IQR bounds,
        and flagged values).
    """
    outlier_reports = []
    outlier_indices = set()
    total_rows = len(df)

    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        outlier_count = mask.sum()
        outlier_percentage = (outlier_count / total_rows) * 100

        flagged_values = "None"
        if outlier_count > 0:
            flagged_values = (
                f"[{df[feature][mask].min():.2f}, {df[feature][mask].max():.2f}]"
            )
            outlier_indices.update(df[mask].index)

        outlier_reports.append(
            {
                "Feature Name": feature,
                "Outliers": outlier_count,
                "Percentage": f"{outlier_percentage:.2f}%",
                "IQR Bounds": f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                "Flagged Values": flagged_values,
            }
        )

    outlier_reports_df = pd.DataFrame(outlier_reports)
    outlier_reports_df = outlier_reports_df[outlier_reports_df["Outliers"] > 0]

    if not outlier_reports_df.empty:
        display(Markdown("**Feature-wise Outlier Analysis**"))
        display(outlier_reports_df)
        display(Markdown("**All Outliers**"))
        outliers = df.loc[list(outlier_indices), features]
        display(outliers)
    else:
        print("**No features with outliers detected**")


def plot_corr_matrix(df: pd.DataFrame) -> None:
    """
    Plots a heatmap of the correlation matrix for the numerical
    features in the DataFrame.

    Args:
        df: The input DataFrame containing numerical features.
    """
    correlation = df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, vmax=1, vmin=-1, cmap="vlag", annot=True)
    plt.title("Correlations heatmap")
    plt.show()


def impute_bmi_with_pipeline(
    df_input: pd.DataFrame, pipeline_filename: str = "bmi_imputer_pipeline.pkl"
) -> pd.DataFrame:
    """
    Imputes missing BMI values using the loaded bmi_imputer_pipeline.
    Expects raw features that bmi_imputer_pipeline was trained on.

    Args:
        df_input: Pandas DataFrame with raw features. Must include 'bmi'
                  and all features required by the BMI imputation pipeline.
        pipeline_filename: Name of the pickled BMI imputer pipeline.

    Returns:
        Pandas DataFrame with BMI values imputed.
    """
    full_pipeline_path = os.path.join(DEFAULT_ARTIFACTS_DIR_UTIL, pipeline_filename)

    df_processed = df_input.copy()

    with open(full_pipeline_path, "rb") as f:
        bmi_pipeline = pickle.load(f)

    missing_bmi_mask = df_processed["bmi"].isna()
    if missing_bmi_mask.any():
        predictor_columns = [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "smoking_status",
        ]
        features_for_prediction = df_processed.loc[missing_bmi_mask, predictor_columns]

        predicted_bmi_values = bmi_pipeline.predict(features_for_prediction)
        df_processed.loc[missing_bmi_mask, "bmi"] = predicted_bmi_values

    return df_processed


def check_vif(X_features: pd.DataFrame) -> None:
    """
    Calculates and prints the Variance Inflation Factor (VIF) for each feature
    in the provided feature DataFrame.

    Args:
        X_features: DataFrame containing only the predictor features (already processed).
    """
    X_numeric = X_features.select_dtypes(include=[np.number])
    X_with_const = add_constant(X_numeric)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_numeric.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(1, X_with_const.shape[1])
    ]
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    vif_data["VIF"] = vif_data["VIF"].apply(lambda x: f"{x:.2f}")
    print(vif_data)


def clean_data_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial data cleaning and transformations suitable for EDA.

    Specifically:
    - Removes rows where 'gender' is 'Other'.
    - Maps 'gender' to a binary 'gender_female' (1 for Female, 0 for Male).
    - Maps 'ever_married' from "Yes"/"No" to binary 1/0.
    - Maps 'Residence_type' to a binary 'residence_type_urban' (1 for Urban, 0 for Rural).
    - Drops original 'gender' and 'Residence_type' columns.

    Args:
        df: Input Pandas DataFrame.

    Returns:
        Pandas DataFrame with EDA-specific cleaning applied.
    """
    df = df.copy()
    df = df[df["gender"] != "Other"]
    df["gender_female"] = df["gender"].map({"Female": 1, "Male": 0})
    df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0})
    df["residence_type_urban"] = df["Residence_type"].map({"Urban": 1, "Rural": 0})
    df = df.drop(columns=["gender", "Residence_type"])
    return df

def check_normality(df: pd.DataFrame, feature: str, alpha: float=0.05) -> None:
    """
    Performs Shapiro-Wilk test for normality on a feature, separated by a
    binary target column.

    Prints the test statistic and p-value for each group and indicates if
    normality assumption is met based on the alpha level.

    Args:
        df: Pandas DataFrame containing the data.
        feature: The numerical column name to test for normality.
        alpha: Significance level for the test.
    """
    group_0 = df[df["stroke"] == 0][feature]
    group_1 = df[df["stroke"] == 1][feature]

    stat0, p0 = shapiro(group_0)
    stat1, p1 = shapiro(group_1)

    print("Normality (stroke=0):", f"Statistic={stat0:.3f}, p={p0:.3f}")
    print("Normality (stroke=1):", f"Statistic={stat1:.3f}, p={p1:.3f}")

    if p0 < alpha or p1 < alpha:
        print("At least one group is NOT normally distributed.\n")
    else:
        print("Both groups are normally distributed.\n")


def check_equal_variance(df: pd.DataFrame, feature: str, alpha: float=0.05
) -> None:
    """
    Performs Levene's test for homogeneity of variances on a feature,
    separated by a binary target column.

    Prints the test statistic and p-value and indicates if variances
    are significantly different based on the alpha level.

    Args:
        df: Pandas DataFrame containing the data.
        feature: The numerical column name to test for equal variances.
        alpha: Significance level for the test.
    """
    group_0 = df[df["stroke"] == 0][feature]
    group_1 = df[df["stroke"] == 1][feature]

    stat, p = levene(group_0, group_1)

    print(f"Levene’s test for equal variance: Statistic={stat:.3f}, p={p:.3f}")

    if p < alpha:
        print("Variances are significantly different.")
    else:
        print("Variances are equal.")


def t_test(df: pd.DataFrame, feature: str,
           equal_var: bool=True, alpha:float=0.05) -> None:
    """
    Performs an independent two-sample t-test for the means of a feature,
    separated by a binary target column.

    Prints the t-statistic, p-value, 95% confidence interval for the mean
    difference, and an interpretation of the result.

    Args:
        df: Pandas DataFrame containing the data.
        feature: The numerical column name to compare means for.
        equal_var: If True, perform a standard independent 2 sample test that
                   assumes equal population variances. If False, perform Welch’s
                   t-test, which does not assume equal population variance.
        alpha: Significance level for the test.
    """
    group_0 = df[df["stroke"] == 0][feature]
    group_1 = df[df["stroke"] == 1][feature]

    result = ttest_ind(group_0, group_1, equal_var=equal_var)
    ci = result.confidence_interval()

    print(f"T-test for '{feature}':")
    print(f"  t-statistic = {result.statistic:.3f}")
    print(f"  p-value     = {result.pvalue:.3f}")
    print(
        f"  95% confidence interval for (non-stroke - stroke) "
        f"mean difference: ({ci.low:.2f}, {ci.high:.2f})\n"
    )

    if result.pvalue < alpha:
        print("Means are significantly different.")
        if ci.high < 0:
            print(f"Stroke group has higher {feature} values.")
        elif ci.low > 0:
            print(f"Non-stroke group has higher {feature} values.")
    else:
        print("Means are not significantly different.")
        print("Direction is unclear (confidence interval crosses zero).")


def chi_square_test(df: pd.DataFrame, feature: str, alpha: float=0.05) -> None:
    """
    Performs a Chi-Square test of independence between a categorical feature
    and a binary target column.

    Prints the contingency table, test statistic, p-value, and an interpretation.
    Also calculates and prints proportions of the target's positive class for each
    category of the feature, along with their confidence intervals and the CI
    for the difference in proportions (for binary features).

    Args:
        df: Pandas DataFrame containing the data.
        feature: The categorical column name to test.
        alpha: Significance level for the test.
    """
    contingency_table = pd.crosstab(df[feature], df["stroke"])
    stat, p, _, _ = chi2_contingency(contingency_table)

    print("Contingency Table:")
    display(contingency_table)

    print(f"Chi-square Test of Independence: Statistic={stat:.3f}, p={p:.3f}")

    if p < alpha:
        print(f"There is a significant association between {feature} and stroke.")
    else:
        print(f"No significant association between {feature} and stroke.")

    n_group0 = contingency_table.iloc[0].sum()
    n_group1 = contingency_table.iloc[1].sum()

    n_group0_stroke = contingency_table.iloc[0, 1]
    n_group1_stroke = contingency_table.iloc[1, 1]

    prop_group0 = n_group0_stroke / n_group0
    prop_group1 = n_group1_stroke / n_group1

    ci_group0 = proportion_confint(n_group0_stroke, n_group0, alpha=alpha)
    ci_group1 = proportion_confint(n_group1_stroke, n_group1, alpha=alpha)

    print("\nProportion of stroke cases:")
    print(
        f"  {contingency_table.index[0]}: {prop_group0:.3f} "
        f"(95% CI: {ci_group0[0]:.3f} - {ci_group0[1]:.3f})"
    )
    print(
        f"  {contingency_table.index[1]}: {prop_group1:.3f} "
        f"(95% CI: {ci_group1[0]:.3f} - {ci_group1[1]:.3f})"
    )

    ci_low, ci_high = confint_proportions_2indep(
        count1=n_group0_stroke,
        nobs1=n_group0,
        count2=n_group1_stroke,
        nobs2=n_group1,
        method="wald",
    )

    print(
        f"\n95% Confidence interval for difference in stroke rates "
        f"({contingency_table.index[0]} - {contingency_table.index[1]}): "
        f"({ci_low:.3f}, {ci_high:.3f})"
    )

    if ci_high < 0:
        print(f"Higher proportion of strokes among {contingency_table.index[1]}.")
    elif ci_low > 0:
        print(f"Higher proportion of strokes among {contingency_table.index[0]}.")
    else:
        print("Direction is unclear (confidence interval crosses zero).")


def evaluate_model_performance(
    y_true: pd.Series,
    y_pred_labels: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str="Model",
    dataset_name: str="Validation"
) -> Dict[str, Any]:
    """
    Prints classification report, key scores, and plots a confusion matrix.
    Returns a dictionary of scores.

    Args:
        y_true: True labels.
        y_pred_labels: Predicted class labels (0s and 1s).
        y_pred_proba: Predicted probabilities for the positive class.
        model_name: Name of the model for titles and print statements.
        dataset_name: Name of the dataset (e.g., "Validation", "Test")
                    for titles and print statements.
    """
    print(f"--- {model_name} ({dataset_name} Set) ---")
    print(
        classification_report(
            y_true, y_pred_labels, target_names=["No Stroke", "Stroke"], zero_division=0
        )
    )

    f2 = fbeta_score(y_true, y_pred_labels, beta=2, pos_label=1, zero_division=0)
    print(f"F2 Score: {f2:.4f}")

    pr_auc_val = 0.0
    roc_auc_val = 0.0

    if y_pred_proba is not None:
        pr_auc_val = average_precision_score(y_true, y_pred_proba)
        roc_auc_val = roc_auc_score(y_true, y_pred_proba)
        print(f"PR-AUC: {pr_auc_val:.4f}")
        print(f"ROC-AUC: {roc_auc_val:.4f}")
    else:
        pr_auc_val = average_precision_score(y_true, y_pred_labels)
        roc_auc_val = roc_auc_score(y_true, y_pred_labels)
        print(f"PR-AUC (from 0/1 preds): {pr_auc_val:.4f}")
        print(f"ROC-AUC (from 0/1 preds): {roc_auc_val:.4f}")

    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted No Stroke", "Predicted Stroke"],
        yticklabels=["Actual No Stroke", "Actual Stroke"],
    )
    plt.title(f"{model_name} Confusion Matrix ({dataset_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    report_dict = classification_report(
        y_true,
        y_pred_labels,
        target_names=["No Stroke", "Stroke"],
        zero_division=0,
        output_dict=True,
    )
    recall_stroke = report_dict["Stroke"]["recall"]
    precision_stroke = report_dict["Stroke"]["precision"]

    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "F2": f2,
        "PR_AUC": pr_auc_val,
        "ROC_AUC": roc_auc_val,
        "Recall_Stroke": recall_stroke,
        "Precision_Stroke": precision_stroke,
        "ConfusionMatrix": cm,
    }


def simple_rule_predictor(X_data_input: pd.DataFrame) -> np.ndarray:
    """
    A simple rule-based predictor for stroke.

    Args:
        X_data_input: Pandas Dataframe to predict on.
    """
    predictions = []
    for _, row in X_data_input.iterrows():
        predicted_stroke = 0
        if row["age"] > 65 and row["hypertension"] == 1:
            predicted_stroke = 1
        elif row["avg_glucose_level"] > 200:
            predicted_stroke = 1
        predictions.append(predicted_stroke)
    return np.array(predictions)


def optimize_threshold_for_f2(y_true: pd.Series,
                              y_pred_proba: np.ndarray,
                              model_name: str="Model") -> Tuple[float, float]:
    """
    Finds the optimal probability threshold to maximize the F2-score.

    Iterates through a range of thresholds, calculates the F2-score for each,
    prints the optimal threshold and corresponding F2-score, and plots the
    F2-score vs. threshold.

    Args:
        y_true: Pandas Series containing the true binary labels (0 or 1).
        y_pred_proba: NumPy array containing the predicted probabilities for the
                      positive class (class 1).
        model_name: A string name for the model, used in print statements and plot title.

    Returns:
        A tuple containing:
            - optimal_threshold (float): The probability threshold that maximizes the F2-score.
            - max_f2_score (float): The maximum F2-score achieved at the optimal threshold.
    """
    print(f"--- {model_name}: Optimizing threshold for F2 on validation set ---")
    thresholds = np.arange(0.01, 1.00, 0.01)
    f2_scores_at_thresholds = [
        fbeta_score(
            y_true,
            (y_pred_proba >= t).astype(int),
            beta=2,
            pos_label=1,
            zero_division=0,
        )
        for t in thresholds
    ]

    optimal_idx = np.argmax(f2_scores_at_thresholds)
    optimal_threshold = thresholds[optimal_idx]
    max_f2_score = f2_scores_at_thresholds[optimal_idx]

    print(f"Optimal threshold for {model_name} on validation: {optimal_threshold:.2f}")
    print(f"Max F2-score at this threshold on validation: {max_f2_score:.4f}")

    plot_data = pd.DataFrame(
        {"Threshold": thresholds, "F2 Score": f2_scores_at_thresholds}
    )

    sns.lineplot(
        x="Threshold",
        y="F2 Score",
        data=plot_data,
        marker=".",
        label="F2 Score per Threshold",
        errorbar=None,
    )
    plt.scatter(
        optimal_threshold,
        max_f2_score,
        color="red",
        s=100,
        zorder=5,
        label=f"Optimal F2: {max_f2_score:.4f}\nat Thresh: {optimal_threshold:.2f}",
    )
    plt.axvline(
        optimal_threshold,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Optimal Threshold ({optimal_threshold:.2f})",
    )
    plt.title(f"F2 Score vs. Threshold for {model_name} (Validation Set)")
    plt.xlabel("Probability Threshold")
    plt.ylabel("F2 Score")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(np.arange(0, 1.01, 0.1))
    min_y_val = 0
    max_y_val = max(f2_scores_at_thresholds) if f2_scores_at_thresholds else 0.1
    padding_y = (max_y_val - min_y_val) * 0.1
    plt.ylim(bottom=min_y_val, top=max_y_val + padding_y if max_y_val > 0 else 0.1)
    plt.tight_layout()
    plt.show()

    return optimal_threshold, max_f2_score


def train_tune_evaluate_model(
    model_instance: Any,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name_str: str,
    scorer: Any,
    scorer_name: str="F2 score",
    cv_folds: int=5,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains a model, tunes its hyperparameters using GridSearchCV, evaluates it
    on a validation set using default and F2-optimized thresholds.

    The function performs cross-validated hyperparameter search, reports the best
    parameters and CV score. It then evaluates the best model on the validation
    set, first with its default prediction threshold and then after finding an
    optimal threshold to maximize the F2-score.

    Args:
        model_instance: An unfitted scikit-learn compatible model instance.
        param_grid: Dictionary with parameters names (str) as keys and lists
                    of parameter settings to try as values.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_val: Validation feature DataFrame.
        y_val: Validation target Series.
        model_name_str: String identifier for the model (e.g., "RandomForest").
        scorer: The scikit-learn scorer object to use for GridSearchCV
                (e.g., f2_scorer).
        scorer_name: A string name for the scorer, used in print statements.
        cv_folds: Number of folds for StratifiedKFold cross-validation.

    Returns:
        A tuple containing:
            - best_model: The best estimator found by GridSearchCV,
              refitted on the full X_train.
            - all_model_run_scores: A dictionary containing
              hyperparameter tuning results, and evaluation scores (from
              `evaluate_model_performance`) for both default and optimal
              thresholds on the validation set.
    """
    all_model_run_scores = {}

    print(f"--- {model_name_str} ---")
    start_time = time.time()

    cv_strat = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model_instance,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv_strat,
        n_jobs=-1,
        verbose=1,
    )

    print(f"Starting {model_name_str} GridSearchCV ({cv_folds}-fold CV)")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    hpt_duration = time.time() - start_time
    print(f"GridSearchCV for {model_name_str} completed in {hpt_duration:.2f} seconds.")
    print(f"Best {model_name_str} params: {grid_search.best_params_}")
    print(f"Best score ({scorer_name}) from CV: {grid_search.best_score_:.4f}\n")
    all_model_run_scores["hpt_best_params"] = grid_search.best_params_
    all_model_run_scores["hpt_cv_score"] = grid_search.best_score_

    y_val_pred_labels = best_model.predict(X_val)
    y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    default_thresh_scores = evaluate_model_performance(
        y_val,
        y_val_pred_labels,
        y_val_pred_proba,
        model_name=f"Tuned {model_name_str} (default thresh)",
    )
    all_model_run_scores["val_default_thresh"] = default_thresh_scores

    optimal_threshold, max_f2_val = optimize_threshold_for_f2(
        y_val, y_val_pred_proba, model_name=model_name_str
    )
    all_model_run_scores["val_optimal_F2_threshold"] = optimal_threshold
    all_model_run_scores["val_optimal_F2_score"] = max_f2_val

    y_val_pred_optimal_labels = (y_val_pred_proba >= optimal_threshold).astype(int)
    optimal_thresh_scores = evaluate_model_performance(
        y_val,
        y_val_pred_optimal_labels,
        y_val_pred_proba,
        model_name=f"Tuned {model_name_str} (Optimal Thresh F2={max_f2_val:.4f})",
    )
    all_model_run_scores["val_optimal_thresh"] = optimal_thresh_scores

    return best_model, all_model_run_scores


def create_model_comparison_table(model_results_summary_list: list) -> pd.DataFrame:
    """
    Creates a summary DataFrame from a list of model result dictionaries.

    Args:
        model_results_summary_list: A list where each element is a dictionary
                                     containing results for a model run, as returned by
                                     train_tune_evaluate_model.

    Returns:
        A Pandas DataFrame summarizing model performance, sorted by the
        F2 score at the optimal threshold.
    """
    summary_data_for_df = []
    for result in model_results_summary_list:
        model_name_default = result["val_default_thresh"].get("Model", "Unknown Model")
        model_name_optimal = result.get("val_optimal_thresh", {}).get(
            "Model", model_name_default
        )

        base_model_name = model_name_default.replace("Tuned ", "").replace(
            " (default thresh)", ""
        )
        if "(Optimal Thresh F2=" in model_name_optimal:
            base_model_name = model_name_optimal.split(" (Optimal Thresh F2=")[
                0
            ].replace("Tuned ", "")

        optimal_f2_score = result.get(
            "val_optimal_F2_score", result["val_default_thresh"].get("F2", np.nan)
        )
        optimal_recall = result.get("val_optimal_thresh", {}).get(
            "Recall_Stroke", result["val_default_thresh"].get("Recall_Stroke", np.nan)
        )
        optimal_precision = result.get("val_optimal_thresh", {}).get(
            "Precision_Stroke",
            result["val_default_thresh"].get("Precision_Stroke", np.nan),
        )

        summary_data_for_df.append(
            {
                "Model": base_model_name,
                "CV F2 Score (HPO)": result.get("hpt_cv_score", np.nan),
                "Val F2 (Default Thresh)": result["val_default_thresh"].get(
                    "F2", np.nan
                ),
                "Val PR-AUC (Default Thresh)": result["val_default_thresh"].get(
                    "PR_AUC", np.nan
                ),
                "Val ROC-AUC (Default Thresh)": result["val_default_thresh"].get(
                    "ROC_AUC", np.nan
                ),
                "Val Optimal F2 Threshold": result.get(
                    "val_optimal_F2_threshold", np.nan
                ),
                "Val F2 (Optimal Thresh)": optimal_f2_score,
                "Val Recall (Optimal Thresh)": optimal_recall,
                "Val Precision (Optimal Thresh)": optimal_precision,
            }
        )

    comparison_df = pd.DataFrame(summary_data_for_df)

    comparison_df = comparison_df.sort_values(
        by="Val F2 (Optimal Thresh)", ascending=False
    )

    return comparison_df


def evaluate_final_model_performance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimal_threshold: float,
    model_name: str="Final Model",
    dataset_name: str="Test"
) -> None:
    """
    Evaluates the final (retrained) model on the test set using a pre-determined optimal threshold.

    Prints a classification report and key performance metrics (F2, PR-AUC, ROC-AUC)
    for the test set evaluation. Also displays a confusion matrix.

    Args:
        model: The pre-fitted final machine learning model.
        X_test: Test feature DataFrame (already processed).
        y_test: Test target Series.
        optimal_threshold: The probability threshold to use for converting probabilities
                           to class labels, typically determined from validation set.
        model_name: String name for the model, used in print statements and plot titles.
        dataset_name: String name of the dataset being evaluated (e.g., "Test").
    """
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred_labels = (y_test_pred_proba >= optimal_threshold).astype(int)

    test_set_scores = evaluate_model_performance(
        y_test,
        y_test_pred_labels,
        y_test_pred_proba,
        model_name=model_name,
        dataset_name=dataset_name,
    )

    print("\n--- Summary of Final CatBoost Model on TEST SET ---")
    print(f"Selected Optimal Threshold (from validation): {optimal_threshold:.2f}")
    print(f"F2 Score: {test_set_scores['F2']:.4f}")
    print(f"Recall (Stroke): {test_set_scores['Recall_Stroke']:.4f}")
    print(f"Precision (Stroke): {test_set_scores['Precision_Stroke']:.4f}")
    print(f"PR-AUC: {test_set_scores['PR_AUC']:.4f}")
    print(f"ROC-AUC: {test_set_scores['ROC_AUC']:.4f}")


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new features from an input DataFrame.

    This includes log-transforming 'avg_glucose_level' and 'bmi', creating
    categorical bins for 'bmi', 'age', and 'avg_glucose_level', and
    generating interaction terms like 'hypertension_and_smokes' and
    'age_X_heart_disease'.

    Args:
        df: Pandas DataFrame containing the original features. Expected columns
            include 'avg_glucose_level', 'bmi', 'age', 'hypertension',
            'smoking_status', 'heart_disease'.

    Returns:
        Pandas DataFrame with the new engineered features added. Original columns
        used for creating new features (like raw 'bmi' or 'avg_glucose_level' if
        their log versions are created) are also retained in the output.
    """
    data = df.copy()

    data["avg_glucose_level_log"] = np.log1p(data["avg_glucose_level"])
    data["bmi_log"] = np.log1p(data["bmi"])

    data["bmi_category"] = pd.cut(
        data["bmi"],
        bins=[0, 18.5, 24.9, 29.9, float("inf")],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        right=False,
    )

    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 18, 40, 60, float("inf")],
        labels=["Child", "Adult", "Middle_Aged", "Senior"],
        right=False,
    )

    data["glucose_level_category"] = pd.cut(
        data["avg_glucose_level"],
        bins=[0, 100, 125, float("inf")],
        labels=["Normal_glucose", "Prediabetes_glucose", "Diabetes_glucose"],
        right=False,
    )

    data["hypertension_and_smokes"] = (
        (data["hypertension"] == 1) & (data["smoking_status"] == "smokes")
    ).astype(int)

    data["age_X_heart_disease"] = data["age"] * data["heart_disease"]

    return data


def plot_bmi(df: pd.DataFrame, title: str) -> None:
    """
    Plots the distribution of BMI, colored by the target variable.

    Uses a histogram with a Kernel Density Estimate (KDE).

    Args:
        df: Pandas DataFrame containing 'bmi' and the target column.
        title: The title for the plot.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="bmi", kde=True, bins=30, hue="stroke")
    plt.title(title)
    plt.xlabel("BMI")
    plt.ylabel("Frequency")
    plt.show()


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """
    Plots histograms for all features in the DataFrame,
    separated by the target variable's classes.

    Args:
        df: Pandas DataFrame containing features and the target column.
    """
    cols = [col for col in df.columns if col != "stroke"]
    rows = len(cols)

    _, axes = plt.subplots(rows, 1, figsize=(10, rows * 4))

    for i, column in enumerate(cols):
        sns.histplot(
            data=df,
            x=column,
            hue="stroke",
            edgecolor="black",
            bins=15,
            ax=axes[i],
            legend=(i == 0),
        )
        axes[i].set_title(f"Distribution of {column} by Stroke")

    plt.tight_layout()
    plt.show()


def plot_outliers(df: pd.DataFrame, outlier_cols: List) -> None:
    """
    Generates boxplots for specified numerical columns, separated by the target variable.

    Args:
        df: Pandas DataFrame containing the data.
        outlier_cols: List of numerical column names to create boxplots for.
    """
    rows = len(outlier_cols)
    fig, axes = plt.subplots(rows, 1, figsize=(10, rows * 4))

    for i, column in enumerate(outlier_cols):
        sns.boxplot(data=df, y=column, hue="stroke", ax=axes[i])

    fig.suptitle("Boxplots of Outliers by Stroke Status", y=0.96)
    plt.tight_layout()
    plt.show()


def feature_importance_permutation(X_train: pd.DataFrame, y_train: pd.Series,
                                   X_val: pd.DataFrame, y_val: pd.Series) -> None:
    """
    Calculates and plots permutation importance for features using a RandomForestClassifier.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_val: Validation feature DataFrame.
        y_val: Validation target Series.
    """
    rf_for_importance = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_for_importance.fit(X_train, y_train)

    perm_importance_result = permutation_importance(
        rf_for_importance,
        X_val,
        y_val,
        n_repeats=10,
        random_state=42,
        scoring="average_precision",
    )
    perm_sorted_idx = perm_importance_result.importances_mean.argsort()

    perm_importances = pd.Series(
        perm_importance_result.importances_mean[perm_sorted_idx],
        index=X_train.columns[perm_sorted_idx],
    )
    perm_importances = perm_importances.sort_values(ascending=False)

    print("\nPermutation Importances (on validation set, using RandomForest):")
    print(perm_importances.head(20))

    plt.figure(figsize=(10, 8))
    sns.barplot(perm_importances.head(20), orient="h")
    plt.title(
        "Top 20 Features by Permutation Importance (RandomForest on validation set)"
    )
    plt.show()


def feature_importance_rfe(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Performs Recursive Feature Elimination (RFE) using Logistic Regression.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
    """
    rfe_model = LogisticRegression(
        solver="liblinear", class_weight="balanced", random_state=42, max_iter=1000
    )

    rfe_selector = RFE(estimator=rfe_model, n_features_to_select=15, step=1)
    rfe_selector.fit(X_train, y_train)

    selected_features_rfe_mask = rfe_selector.support_
    selected_features_rfe = X_train.columns[selected_features_rfe_mask]

    print("\nSelected features by RFE (Logistic Regression):")
    print(selected_features_rfe)
