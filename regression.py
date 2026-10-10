## this is a Python module for regression analysis in the application for mass spectrometry data analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)

from sklearn.inspection import permutation_importance


def run_pls(
    X,
    y,
    train_percent=0.8,
    preprocess="none",
    cv_method="StratifiedKFold",
    resampling=5,
    max_comp=10
):

    # ----------------------------
    # Validate inputs
    # ----------------------------

    if X is None or y is None:
        raise ValueError("Missing X or y")


    X = pd.DataFrame(X).copy()
    y = pd.Series(y).copy()


    # ----------------------------
    # Remove zero variance features
    # ----------------------------

    feature_names = X.columns.tolist()

    selector = VarianceThreshold(
        threshold=0
    )

    X = selector.fit_transform(X)

    feature_names = np.array(feature_names)[
        selector.get_support()
    ]


    if X.shape[1] == 0:
        raise ValueError(
            "No features remain after variance filtering"
        )


    # ----------------------------
    # Encode target for PLS
    # ----------------------------

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)


    # ----------------------------
    # Train/test split
    # ----------------------------

    X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
        X,
        y,
        y_encoded,
        train_size=train_percent,
        random_state=123,
        stratify=y
    )


    # ----------------------------
    # Preprocessing
    # ----------------------------

    scaler = None

    if preprocess and preprocess != "none":

        if "center" in preprocess or "scale" in preprocess:

            scaler = StandardScaler()

            X_train = scaler.fit_transform(
                X_train
            )

            X_test = scaler.transform(
                X_test
            )


    # ----------------------------
    # Cross-validation setup
    # ----------------------------

    if cv_method == "RepeatedCV":

        cv = RepeatedStratifiedKFold(
            n_splits=resampling,
            n_repeats=3,
            random_state=123
        )

    elif cv_method == "LOOCV":

        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()

    else:

        cv = StratifiedKFold(
            n_splits=resampling,
            shuffle=True,
            random_state=123
        )


    # ----------------------------
    # Tune PLS components
    # ----------------------------

    best_accuracy = -np.inf

    best_model = None
    best_pls = None
    best_ncomp = None


    max_possible = min(
        max_comp,
        X_train.shape[1],
        X_train.shape[0]-1
    )


    for ncomp in range(
        1,
        max_possible + 1
    ):

        pls = PLSRegression(
            n_components=ncomp
        )


        # PLS transformation

        X_train_scores = pls.fit_transform(
            X_train,
            y_train_enc
        )[0]


        X_test_scores = pls.transform(
            X_test
        )


        classifier = LogisticRegression(
            max_iter=5000
        )


        classifier.fit(
            X_train_scores,
            y_train
        )


        cv_scores = cross_val_score(
            classifier,
            X_train_scores,
            y_train,
            cv=cv,
            scoring="accuracy"
        )


        mean_cv = np.mean(cv_scores)


        if mean_cv > best_accuracy:

            best_accuracy = mean_cv

            best_model = classifier
            best_pls = pls
            best_ncomp = ncomp



    # ----------------------------
    # Final prediction
    # ----------------------------

    X_train_scores = best_pls.transform(
        X_train
    )

    X_test_scores = best_pls.transform(
        X_test
    )


    pred = best_model.predict(
        X_test_scores
    )


    accuracy = accuracy_score(
        y_test,
        pred
    )


    kappa = cohen_kappa_score(
        y_test,
        pred
    )


    cm = confusion_matrix(
        y_test,
        pred,
        labels=encoder.classes_
    )


    cm_df = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    )


    # ----------------------------
    # Accuracy plot
    # ----------------------------

    fig_acc, ax = plt.subplots(
        figsize=(7,5)
    )

    ax.bar(
        ["Accuracy","Kappa"],
        [
            accuracy,
            kappa
        ]
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_title(
        "PLS-DA Performance"
    )

    fig_acc.tight_layout()


    # ----------------------------
    # Variable importance
    # ----------------------------

    perm = permutation_importance(
        best_model,
        X_test_scores,
        y_test,
        n_repeats=20,
        random_state=123
    )


    importance_df = pd.DataFrame(
        {
            "Component":
                np.arange(
                    len(perm.importances_mean)
                ),
            "Importance":
                perm.importances_mean
        }
    )


    fig_imp, ax = plt.subplots(
        figsize=(8,5)
    )

    importance_df.sort_values(
        "Importance",
        ascending=False
    ).head(20).plot.bar(
        x="Component",
        y="Importance",
        ax=ax
    )


    ax.set_title(
        "PLS Component Importance"
    )

    fig_imp.tight_layout()



    # ----------------------------
    # PLS scores plot
    # ----------------------------

    fig_scores, ax = plt.subplots(
        figsize=(7,5)
    )


    scores = best_pls.transform(
        X_train
    )


    scatter = ax.scatter(
        scores[:,0],
        scores[:,1],
        c=y_train_enc,
        alpha=0.7
    )


    ax.set_xlabel(
        "PLS Component 1"
    )

    ax.set_ylabel(
        "PLS Component 2"
    )

    ax.set_title(
        "PLS-DA Scores"
    )


    fig_scores.tight_layout()



    # ----------------------------
    # Return Gradio results
    # ----------------------------

    result = {

        "status":
            "success",

        "accuracy":
            float(accuracy),

        "kappa":
            float(kappa),

        "best_parameters":
            {
                "n_components":
                    int(best_ncomp)
            },

        "confusion_matrix":
            cm_df,

        "accuracy_plot":
            fig_acc,

        "importance_plot":
            fig_imp,

        "scores_plot":
            fig_scores

    }

    return result