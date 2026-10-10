## this is a Python module for classification using random forest in the application for mass spectrometry data analysis

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score
)

from sklearn.inspection import permutation_importance


def run_rf(
    X,
    y,
    train_percent,
    preprocess="none",
    cv_method="StratifiedKFold",
    resampling=5
):

    ####################################################
    # Load data
    ####################################################

    if X is None or y is None:
        raise ValueError("Missing X or y")

    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()


    ####################################################
    # Encode categorical predictors
    ####################################################

    for col in X.columns:

        if X[col].dtype == object:

            X[col] = LabelEncoder().fit_transform(
                X[col].astype(str)
            )


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

        steps.append(
            ("scaler", StandardScaler())
        )


    rf = RandomForestClassifier(
        random_state=123,
        n_jobs=-1
    )


    steps.append(
        ("rf", rf)
    )


    pipe = Pipeline(steps)


    ####################################################
    # Random Forest tuning parameters
    ####################################################

    param_grid = {

        "rf__n_estimators": [
            100,
            300,
            500
        ],

        "rf__max_features": [
            "sqrt",
            "log2"
        ],

        "rf__max_depth": [
            None,
            20,
            40
        ],

        "rf__min_samples_split": [
            2,
            5
        ],

        "rf__min_samples_leaf": [
            1,
            2
        ]
    }


    ####################################################
    # Cross validation
    ####################################################

    cv = StratifiedKFold(
        n_splits=resampling,
        shuffle=True,
        random_state=123
    )


    ####################################################
    # Randomized search
    ####################################################

    search = RandomizedSearchCV(

        estimator=pipe,

        param_distributions=param_grid,

        n_iter=20,

        scoring="accuracy",

        cv=cv,

        random_state=123,

        n_jobs=-1

    )


    ####################################################
    # Train model
    ####################################################

    search.fit(
        X_train,
        y_train
    )


    best_model = search.best_estimator_


    ####################################################
    # Prediction
    ####################################################

    pred = best_model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        pred
    )


    kappa = cohen_kappa_score(
        y_test,
        pred
    )


    ####################################################
    # Confusion matrix
    ####################################################

    cm = confusion_matrix(
        y_test,
        pred
    )


    cm_df = pd.DataFrame(

        cm,

        index=best_model.classes_,

        columns=best_model.classes_

    )


    ####################################################
    # Accuracy plot
    ####################################################

    scores = search.cv_results_[
        "mean_test_score"
    ]


    fig_acc, ax = plt.subplots(
        figsize=(8,6)
    )


    ax.plot(
        scores,
        marker="o"
    )


    ax.set_title(
        "Random Forest Randomized Search Accuracy"
    )

    ax.set_xlabel(
        "Parameter Combination"
    )

    ax.set_ylabel(
        "Mean CV Accuracy"
    )


    fig_acc.tight_layout()


    ####################################################
    # Variable importance
    ####################################################

    fig_imp, ax = plt.subplots(
        figsize=(10,6)
    )


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


        importance.nlargest(
            20
        ).plot.bar(
            ax=ax
        )


        ax.set_title(
            "Permutation Importance"
        )


    except Exception:


        ax.text(

            0.25,

            0.5,

            "Variable importance unavailable"

        )


    fig_imp.tight_layout()


    ####################################################
    # Return
    ####################################################

    return {

        "status": "success",

        "accuracy": float(accuracy),

        "kappa": float(kappa),

        "best_parameters": search.best_params_,

        "confusion_matrix": cm_df,

        "accuracy_plot": fig_acc,

        "importance_plot": fig_imp

    }