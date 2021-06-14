"""Plotting script for event detection."""

from os.path import join as opj

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import ev

SUBJECT = "sub-002ParkMabCm"
NROIS = "200"
NRAND = 100
TR = 0.83
MAINDIR = "/bcbl/home/public/PARK_VFERRER/PFM_data"
TEMPDIR = "/bcbl/home/public/PARK_VFERRER/PFM_data/temp_" + SUBJECT + "_" + NROIS
ATS = np.loadtxt(opj(MAINDIR, "pb06.sub-002ParkMabCm.denoised_no_censor_ATS_abs_95.1D"))
ATLAS = opj(TEMPDIR, "atlas.nii.gz")
FIGSIZE = (45, 30)
HISTORY = "Deconvolution based on event-detection."

font = {"family": "normal", "weight": "normal", "size": 22}
matplotlib.rc("font", **font)


def plot_comparison(
    rss_orig_sur,
    rssr_orig_sur,
    idxpeak_orig_sur,
    rss_fitt,
    rssr_fitt,
    idxpeak_fitt,
    rss_beta,
    rssr_beta,
    idxpeak_beta,
    rss_auc,
    rssr_auc,
    idxpeak_auc,
    ats,
):
    """
    Plot comparison of different RSS with vertical subplots.
    """
    greymap = cm.get_cmap("Greys")
    colors = greymap(np.linspace(0, 0.65, rssr_orig_sur.shape[1]))

    _, axs = plt.subplots(5, 1, figsize=FIGSIZE)
    for i in range(rssr_orig_sur.shape[1]):
        axs[0].plot(rssr_orig_sur[:, i], color=colors[i], linewidth=0.5)
    axs[0].plot(
        idxpeak_orig_sur,
        rss_orig_sur[idxpeak_orig_sur],
        "r*",
        label="orig_sur-peaks",
        markersize=20,
    )
    axs[0].plot(
        rss_orig_sur,
        color="k",
        linewidth=3,
        label="orig_sur",
    )
    axs[0].set_title("Original signal")

    for i in range(rssr_orig_sur.shape[1]):
        axs[1].plot(rssr_fitt[:, i], color=colors[i], linewidth=0.5)
    axs[1].plot(
        idxpeak_fitt, rss_fitt[idxpeak_fitt], "r*", label="fitt-peaks", markersize=20
    )
    axs[1].plot(rss_fitt, color="k", linewidth=3, label="fitt")
    axs[1].set_title("Fitted signal")

    for i in range(rssr_orig_sur.shape[1]):
        axs[2].plot(rssr_beta[:, i], color=colors[i], linewidth=0.5)
    axs[2].plot(
        idxpeak_beta, rss_beta[idxpeak_beta], "r*", label="beta-peaks", markersize=20
    )
    axs[2].plot(rss_beta, color="k", linewidth=3, label="beta")
    axs[2].set_title("Betas")

    for i in range(rssr_orig_sur.shape[1]):
        axs[3].plot(rssr_auc[:, i], color=colors[i], linewidth=0.5)
    axs[3].plot(
        idxpeak_auc, rss_auc[idxpeak_auc], "r*", label="AUC-peaks", markersize=20
    )
    axs[3].plot(rss_auc, color="k", linewidth=3, label="AUC")
    axs[3].set_title("AUCs")

    axs[4].plot(ats, label="ATS", color="black")
    axs[4].set_title("Activation time-series")

    plt.legend()
    plt.savefig(opj(MAINDIR, "event_detection.png"), dpi=300)


def plot_all(
    rss_orig_sur, idxpeak_orig_sur, rss_beta, idxpeak_beta, rss_fitt, idxpeak_fitt
):
    """
    Plot all RSS lines on same figure.
    """
    plt.figure(figsize=FIGSIZE)

    # Original signal
    rss_orig_norm = (rss_orig_sur - rss_orig_sur.min()) / (
        rss_orig_sur.max() - rss_orig_sur.min()
    )
    plt.plot(
        idxpeak_orig_sur,
        rss_orig_norm[idxpeak_orig_sur],
        "r*",
        "linewidth",
        3,
        label="orig_sur-peaks",
    )
    plt.plot(
        range(rss_orig_norm.shape[0]),
        rss_orig_norm,
        "k",
        "linewidth",
        3,
        label="orig_sur",
    )

    # Betas
    rss_beta_norm = (rss_beta - rss_beta.min()) / (rss_beta.max() - rss_beta.min())
    plt.plot(
        idxpeak_beta,
        rss_beta_norm[idxpeak_beta],
        "g*",
        "linewidth",
        3,
        label="deconvolved_peaks",
    )
    plt.plot(
        range(rss_beta_norm.shape[0]),
        rss_beta_norm,
        "b",
        "linewidth",
        3,
        label="deconvolved",
    )

    # Fitted signal
    rss_fitt_norm = (rss_fitt - rss_fitt.min()) / (rss_fitt.max() - rss_fitt.min())
    plt.plot(
        idxpeak_fitt,
        rss_fitt_norm[idxpeak_fitt],
        "m*",
        "linewidth",
        3,
        label="fitted_peaks",
    )
    plt.plot(
        range(rss_fitt_norm.shape[0]),
        rss_fitt_norm,
        "y",
        "linewidth",
        3,
        label="fitted",
    )
    plt.legend()
    plt.savefig(opj(MAINDIR, "event_detection_all.png"), dpi=300)


def plot_ets_matrix(ets, outdir, sufix="", vmin=-0.5, vmax=0.5):
    """
    Plots edge-time matrix
    """
    plt.figure(figsize=FIGSIZE)
    plt.imshow(ets.T, vmin=vmin, vmax=vmax, cmap="bwr", aspect="auto")
    plt.title("Edge-time series")
    plt.xlabel("Time (TR)")
    plt.ylabel("Edge-edge connections")
    plt.colorbar()
    plt.savefig(opj(outdir, f"ets{sufix}.png"), dpi=300)


def main():
    """
    Main function to perform event detection and plot results.
    """
    # Perform event detection on BETAS
    print("Performing event-detection on betas...")
    data_file = opj(MAINDIR, "pb06.sub-002ParkMabCm.denoised_no_censor_beta.nii.gz")
    (
        ets_beta,
        rss_beta,
        rssr_beta,
        idxpeak_beta,
        etspeaks_beta,
        mu_beta,
        _,
        _,
        _,
    ) = ev.event_detection(data_file, ATLAS, opj(TEMPDIR, "surrogate_"), "_beta")

    # Perform event detection on ORIGINAL data
    print("Performing event-detection on original data...")
    data_file = opj(MAINDIR, "pb06.sub-002ParkMabCm.denoised_no_censor.nii.gz")
    (
        ets_orig_sur,
        rss_orig_sur,
        rssr_orig_sur,
        idxpeak_orig_sur,
        etspeaks_orig_sur,
        mu_orig_sur,
        _,
        _,
        _,
    ) = ev.event_detection(data_file, ATLAS, opj(TEMPDIR, "surrogate_"))

    # Perform event detection on FITTED signal
    print("Performing event-detection on fitted signal...")
    data_file = opj(MAINDIR, "pb06.sub-002ParkMabCm.denoised_no_censor_fitt.nii.gz")
    (
        ets_fitt,
        rss_fitt,
        rssr_fitt,
        idxpeak_fitt,
        etspeaks_fitt,
        mu_fitt,
        _,
        _,
        _,
    ) = ev.event_detection(data_file, ATLAS, opj(TEMPDIR, "surrogate_"), "_fitt")

    # Perform event detection on AUC
    print("Performing event-detection on AUC...")
    data_file = opj(MAINDIR, "sub-002ParkMabCm_AUC_200.nii.gz")
    (
        ets_auc,
        rss_auc,
        rssr_auc,
        idxpeak_auc,
        etspeaks_AUC,
        mu_AUC,
        ets_auc_denoised,
        idx_u,
        idx_v,
    ) = ev.event_detection(data_file, ATLAS, opj(TEMPDIR, "surrogate_AUC_"))

    print("Making plots...")
    # Plot comparison of rss time series, null, and significant peaks for
    # original, betas, fitted, AUC and ATS
    plot_comparison(
        rss_orig_sur,
        rssr_orig_sur,
        idxpeak_orig_sur,
        rss_fitt,
        rssr_fitt,
        idxpeak_fitt,
        rss_beta,
        rssr_beta,
        idxpeak_beta,
        rss_auc,
        rssr_auc,
        idxpeak_auc,
        ATS,
    )

    # Plot all rss time series, null, and significant peaks in one plot
    plot_all(
        rss_orig_sur, idxpeak_orig_sur, rss_beta, idxpeak_beta, rss_fitt, idxpeak_fitt
    )

    print("Plotting original, AUC, and AUC-denoised ETS matrices...")
    # Plot ETS matrix of original signal
    plot_ets_matrix(ets_orig_sur, MAINDIR, "_original")

    # Plot ETS and denoised ETS matrices of AUC
    plot_ets_matrix(ets_auc, MAINDIR, "_AUC_original")
    plot_ets_matrix(ets_auc_denoised, MAINDIR, "_AUC_denoised")

    # Perform debiasing based on thresholded edge-time matrix
    data_file = opj(MAINDIR, "pb06.sub-002ParkMabCm.denoised_no_censor.nii.gz")
    beta, _ = ev.debiasing(
        data_file, ATLAS, ets_auc_denoised, idx_u, idx_v, TR, MAINDIR, HISTORY
    )

    print("Plotting edge-time matrix of ETS-based deconvolution.")
    denoised_beta_ets, _, _ = ev.calculate_ets(beta, beta.shape[1])
    plot_ets_matrix(denoised_beta_ets, MAINDIR, "_beta_denoised")

    print("THE END")


if __name__ == "__main__":
    main()
