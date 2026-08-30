import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pybaselines import Baseline

def baseline_correction(file, bl_method):
    """
    Python script for baseline correction

    Parameters
    ----------
    file : str
        CSV file containing spectra
    bl_method : str
        Baseline method:
            als
            arpls
            airpls
            modpoly
    """

    ####################################################
    # Read data
    ####################################################

    df = pd.read_csv(file)

    spectra = df.to_numpy(dtype=float)

    baseline_fitter = Baseline()

    corrected = []
    baselines = []

    ####################################################
    # Correct every spectrum
    ####################################################

    for spectrum in spectra:

        if bl_method == "als":
            baseline, _ = baseline_fitter.asls(spectrum)

        elif bl_method == "arpls":
            baseline, _ = baseline_fitter.arpls(spectrum)

        elif bl_method == "airpls":
            baseline, _ = baseline_fitter.airpls(spectrum)

        elif bl_method == "modpoly":
            baseline, _ = baseline_fitter.modpoly(spectrum)

        else:
            raise ValueError(f"Unknown baseline method: {bl_method}")

        corrected.append(spectrum - baseline)
        baselines.append(baseline)

    corrected = np.asarray(corrected)
    baselines = np.asarray(baselines)

    ####################################################
    # Plot first spectrum
    ####################################################

    plt.figure(figsize=(10,6))

    plt.plot(spectra[0], label="Original")
    plt.plot(baselines[0], label="Estimated Baseline")
    plt.plot(corrected[0], label="Corrected")

    plt.legend()

    plt.tight_layout()

    plt.savefig("baseline_plot.png")

    plt.close()

    ####################################################
    # Save corrected spectra
    ####################################################
    corrected_df = pd.DataFrame(
        corrected,
        columns=df.columns
    )
    corrected_df.to_csv(
        "corrected_mass_spec.csv",
        index=False
    )
    
    return corrected_df