## this is the backend code for the Gradio application, which includes functions for loading data, splitting metadata, 
## performing baseline correction, signal processing, and running machine learning models. The machine learning algorithms are 
## implemented in Python using libraries such as scikit-learn. The backend functions are designed to be called from the Gradio interface 
## to perform the necessary data processing and modeling tasks based on user input.
import pandas as pd
import numpy as np
from pybaselines import Baseline
import matplotlib.pyplot as plt

#### loading file and returning the dataframe, preview, and list of metadata columns and target columns
def load_file(file, max_metadata_cols=10):

    if file is None:
        return None, None, []

    df = pd.read_csv(file)

    preview = df.head()

    displayed_cols = list(df.columns[:max_metadata_cols])

    target_cols = displayed_cols.copy()

    return df, preview, displayed_cols, target_cols


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

    X = spectra_df.values

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

    for spectrum in X:
        baseline, _ = baseline_function(spectrum)
        corrected.append(spectrum - baseline)

    X_corr = np.asarray(corrected)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(X_corr.T, linewidth=0.8)
    ax.set_title("Baseline Corrected Spectra")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Intensity")
    plt.close(fig)

    return fig, X_corr