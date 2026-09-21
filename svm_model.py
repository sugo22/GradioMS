##this is a Python module for SVM classification model implementation in the application for mass spectrometry data analysis and 
##classification modeling built completely in Python with the machine learning algorithms implemented in Python as well

import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split,GridSearchCV,StratifiedKFold)
from sklearn.preprocessing import (StandardScaler,LabelEncoder)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix,accuracy_score,cohen_kappa_score,ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

def run_svm(
    X,
    y,
    train_percent,
    svm_method,
    preprocess="none",
    cv_method="StratifiedKFold",
    resampling=5,
    cost=1,
    gamma=None
):

    ####################################################
    # Load data
    ####################################################
    X = X.copy()
    y = y.copy()

    if X is None or y is None:
        raise ValueError("Missing X or y")

    X = pd.DataFrame(X)
    y = pd.Series(y)
    ####################################################
    # Convert categorical predictors
    ####################################################

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    ####################################################
    # Remove missing values
    ####################################################

    keep = ~(X.isna().any(axis=1) | y.isna())

    X = X.loc[keep]
    y = y.loc[keep]

    ####################################################
    # Train/Test split
    ####################################################

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_percent,
        stratify=y,
        random_state=123
    )

    ####################################################
    # Pipeline
    ####################################################

    steps = []

    if preprocess != "none":
        steps.append(("scaler", StandardScaler()))

    ####################################################
    # SVM kernel
    ####################################################

    if svm_method == "svmLinear":

        svc = SVC(kernel="linear")

        param_grid = {
            "svm__C": [0.01, 0.1, 1, 10]
        }

    elif svm_method == "svmRadial":

        svc = SVC(kernel="rbf")

        param_grid = {
            "svm__C": [0.1, 1, 5, 10],
            "svm__gamma": ["scale", 0.001, 0.01, 0.1]
        }

    elif svm_method == "svmPoly":

        svc = SVC(kernel="poly")

        param_grid = {
            "svm__C": [0.1, 1, 5],
            "svm__degree": [2, 3],
            "svm__gamma": ["scale", 0.001, 0.01]
        }

    else:
        raise ValueError("Invalid SVM method")

    steps.append(("svm", svc))

    pipe = Pipeline(steps)

    ####################################################
    # Cross-validation
    ####################################################

    cv = StratifiedKFold(
        n_splits=resampling,
        shuffle=True,
        random_state=123
    )

    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )

    ####################################################
    # Train
    ####################################################

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    ####################################################
    # Predict
    ####################################################

    pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    kappa = cohen_kappa_score(y_test, pred)

    cm = confusion_matrix(y_test, pred)

    ####################################################
    # Confusion matrix dataframe
    ####################################################

    cm_df = pd.DataFrame(
        cm,
        index=best_model.classes_,
        columns=best_model.classes_
    )

    ####################################################
    # Accuracy plot
    ####################################################

    scores = grid.cv_results_["mean_test_score"]

    plt.figure(figsize=(8,6))
    plt.plot(scores, marker="o")
    plt.title("SVM Grid Search Accuracy")
    plt.xlabel("Parameter Combination")
    plt.ylabel("Mean CV Accuracy")
    plt.tight_layout()
    plt.savefig("model_accuracy.png")
    fig_acc, ax = plt.subplots(figsize=(8, 6))

    ax.plot(scores, marker="o")
    ax.set_title("SVM Grid Search Accuracy")
    ax.set_xlabel("Parameter Combination")
    ax.set_ylabel("Mean CV Accuracy")

    plt.tight_layout()
    plt.close()

    ####################################################
    # Variable Importance
    ####################################################

    plt.figure(figsize=(10,6))

    try:

        result = permutation_importance(
            best_model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=123,
            n_jobs=-1
        )

        importance = pd.Series(
            result.importances_mean,
            index=X.columns
        )

        importance.nlargest(20).plot.bar()

        plt.title("Permutation Importance")

    except Exception:

        plt.text(
            0.3,
            0.5,
            "Variable importance unavailable"
        )

    plt.tight_layout()
    plt.savefig("variable_importance.png")
    fig_imp, ax = plt.subplots(figsize=(10, 6))

    importance.nlargest(20).plot.bar(ax=ax)

    ax.set_title("Permutation Importance")

    plt.tight_layout()
    plt.close()

    ####################################################
    # JSON output
    ####################################################

    result = {
    "status": "success",
    "accuracy": float(accuracy),
    "kappa": float(kappa),
    "best_parameters": grid.best_params_,
    "confusion_matrix": cm_df,
    "accuracy_plot": fig_acc,
    "importance_plot": fig_imp,
    }

    return result
