## this is the backend code for the Gradio application, which includes functions for loading data, splitting metadata, 
## performing baseline correction, signal processing, and running machine learning models. The machine learning algorithms are 
## implemented in Python using libraries such as scikit-learn. The backend functions are designed to be called from the Gradio interface 
## to perform the necessary data processing and modeling tasks based on user input.
import pandas as pd
import gradio as gr
import numpy as np
from pybaselines import Baseline
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from randomforest import run_rf
from svm_model import run_svm
from regression import run_pls

#### loading file and returning the dataframe, preview, and list of metadata columns and target columns
def load_file(file, max_metadata_cols=10):

    if file is None:
        return (
            None,
            None,
            gr.CheckboxGroup(choices=[], value=[]),
            gr.Dropdown(choices=[], value=None)
        )

    filepath = file.name if hasattr(file, "name") else file

    df = pd.read_csv(filepath)

    preview = df.head()

    displayed_cols = list(df.columns[:max_metadata_cols])

    target_cols = displayed_cols.copy()

    return (
        df,
        preview,
        gr.CheckboxGroup(
            choices=displayed_cols,
            value=[]
        ),
        gr.Dropdown(
            choices=target_cols,
            value=None
        )
    )


# ---------------------------------------------------
# Split Metadata
# ---------------------------------------------------

def split_metadata(df, metadata_cols, target_col):

    if df is None or not metadata_cols:
        return None, None, None
    if target_col is None:
        raise ValueError("Target variable not selected")
    metadata_cols = metadata_cols or []

    if target_col in metadata_cols:
        raise ValueError(
            "Target variable cannot also be metadata"
        )
    reserved_cols = metadata_cols + [target_col]

    metadata_df = df[metadata_cols]

    y = df[target_col]

    spectra_df = df.drop(columns=reserved_cols)

    X = spectra_df.copy()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(X.T, linewidth=0.8)

    ax.set_title("Original Spectra")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Intensity")

    plt.close(fig)

    return metadata_df, X, y, fig

def baseline_correction(X, bl_method):

    if X is None:
        return None, None

    baseline_fitter = Baseline()

    method_map = {
        "asls": "asls",
        "airpls": "airpls",
        "modpoly": "modpoly"
    }

    method = method_map.get(bl_method, bl_method)

    if not hasattr(baseline_fitter, method):
        raise ValueError(f"Unknown baseline method: {bl_method}")

    baseline_function = getattr(baseline_fitter, method)

    corrected = []

    for _, spectrum in X.iterrows():

        spectrum = spectrum.values.astype(float)

        baseline, _ = baseline_function(spectrum)

        corrected.append(spectrum - baseline)

    X_corr = pd.DataFrame(
        corrected,
        columns=X.columns
    )

    fig, ax = plt.subplots(figsize=(7,4))

    ax.plot(X_corr.T)

    ax.set_title("Baseline Corrected Spectra")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Intensity")

    plt.close(fig)

    return fig, X_corr

def preprocess_data(X_corr,method,order,p,w,m):

    if X_corr is None:
        return None

    if method == "None":
        return X_corr

    if method == "Derivatives":
        return np.diff(
            X_corr,
            n=int(order),
            axis=1
        )

    if method == "Savitzky-Golay":

        w = int(w)
        p = int(p)

        if w % 2 == 0:
            w += 1

        if w <= p:
            w = p + 3

        return savgol_filter(
            X_corr,
            w,
            p,
            deriv=int(m),
            axis=1
        )

    return X_corr

def two_step(X_corr,m1,m2,order,p,w,m):

    if X_corr is None:
        return None, None

    X1 = preprocess_data(X_corr,m1,order,p,w,m)

    X2 = preprocess_data(X1,m2,order,p,w,m)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(X2.T, linewidth=0.8)

    ax.set_title("Processed Spectra")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Intensity")

    plt.close(fig)

    return fig, X2

MODEL_MAP = {
    "SVM": run_svm,
}

def run_pymodel(
    X,
    y,
    model_type,
    train_percent,
    preprocess,
    cv_method,
    resampling,
    svm_method=None,
    max_comp=10
):

    if model_type == "SVM":

        result = run_svm(
            X,
            y,
            train_percent,
            svm_method,
            preprocess,
            cv_method,
            resampling
        )


    elif model_type == "Random Forest":

        result = run_rf(
            X,
            y,
            train_percent,
            preprocess,
            cv_method,
            resampling
        )
    elif model_type == "PLS":

        result = run_pls(
            X,
            y,
            train_percent,
            preprocess,
            cv_method,
            resampling,
            max_comp
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    return (
        result["accuracy"],
        result["kappa"],
        result["best_parameters"],
        result["confusion_matrix"],
        result["accuracy_plot"],
        result["importance_plot"],
        result.get("scores_plot", None)
    )