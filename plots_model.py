from typing import Union, List
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def churn_prediction(
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
        coef_type: str = 'coefficients'  # options: 'coefficients' or 'features'
) -> pd.DataFrame:
    """
    Train a classification model, evaluate performance, and visualize results.

    Parameters
    ----------
    model : sklearn estimator
        Classifier to train and evaluate.
    X_train, X_test : pd.DataFrame
        Training and testing feature matrices.
    y_train, y_test : pd.Series
        Training and testing target vectors.
    feature_names : list
        List of feature names corresponding to X_train columns.
    coef_type : str, optional
        'coefficients' for coef_ (linear models) or 'features' for feature_importances_ (tree models).

    Returns
    -------
    pd.DataFrame
        DataFrame containing features and their corresponding coefficients/importances.
    """
    # Train model
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Predict probabilities for ROC AUC
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.decision_function(X_test)

    # Extract coefficients or feature importances
    if coef_type == "coefficients" and hasattr(model, "coef_"):
        coef_values = model.coef_.ravel()
    elif coef_type == "features" and hasattr(model, "feature_importances_"):
        coef_values = model.feature_importances_
    else:
        coef_values = [0] * len(feature_names)

    coef_df = pd.DataFrame({
        "features": feature_names,
        "coefficients": coef_values
    }).sort_values(by="coefficients", ascending=False)

    # Print model metrics
    print(model)
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("Accuracy Score:", accuracy_score(y_test, predictions))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    sns.heatmap(conf_matrix, fmt="d", annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")

    # ROC Curve
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)
    print("Area under curve : ", auc_score, "\n")
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC: {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()

    # Feature Importances / Coefficients
    plt.subplot(2, 1, 2)
    sns.barplot(x="features", y="coefficients", data=coef_df)
    plt.title("Feature Importances / Coefficients")
    plt.xticks(rotation=90)
    plt.grid()

    plt.tight_layout()
    plt.show()
