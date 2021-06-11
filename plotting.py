from os.path import join as opj

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import ev

subject = "sub-002ParkMabCm"
nROI = "200"
nRAND = 100
DIR = "/bcbl/home/public/PARK_VFERRER/PFM_data"
TEMP = "/bcbl/home/public/PARK_VFERRER/PFM_data/temp_" + subject + "_" + nROI
ats = np.loadtxt(opj(DIR, "pb06.sub-002ParkMabCm.denoised_no_censor_ATS_abs.1D"))
ATLAS_file = opj(TEMP, "atlas.nii.gz")
FIGSIZE = (25, 30)


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
    rss_AUC,
    rssr_AUC,
    idxpeak_AUC,
    ats,
):
    """
    Plot comparison of different RSS with vertical subplots.
    """
    greymap = cm.get_cmap("Greys")
    colors = greymap(np.linspace(0, 0.65, rssr_orig_sur.shape[1]))

    plt.figure(figsize=FIGSIZE)
    fig, axs = plt.subplots(5, 1)
    for i in range(rssr_orig_sur.shape[1]):
        axs[0].plot(range(rssr_orig_sur.shape[0]), rssr_orig_sur[:, i], color=colors[i])
    axs[0].plot(
        idxpeak_orig_sur, rss_orig_sur[idxpeak_orig_sur], "r*", label="orig_sur-peaks"
    )
    axs[0].plot(
        range(rss_orig_sur.shape[0]),
        rss_orig_sur,
        "k",
        "linewidth",
        2,
        label="orig_sur",
    )
    axs[0].set_title("Original signal")

    for i in range(rssr_orig_sur.shape[1]):
        axs[1].plot(range(rssr_fitt.shape[0]), rssr_fitt[:, i], color=colors[i])
    axs[1].plot(idxpeak_fitt, rss_fitt[idxpeak_fitt], "r*", label="fitt-peaks")
    axs[1].plot(range(rss_fitt.shape[0]), rss_fitt, "k", "linewidth", 2, label="fitt")
    axs[1].set_title("Fitted signal")

    for i in range(rssr_orig_sur.shape[1]):
        axs[2].plot(range(rssr_beta.shape[0]), rssr_beta[:, i], color=colors[i])
    axs[2].plot(idxpeak_beta, rss_beta[idxpeak_beta], "r*", label="beta-peaks")
    axs[2].plot(range(rss_beta.shape[0]), rss_beta, "k", "linewidth", 2, label="beta")
    axs[2].set_title("Betas")

    for i in range(rssr_orig_sur.shape[1]):
        axs[3].plot(range(rssr_AUC.shape[0]), rssr_AUC[:, i], color=colors[i])
    axs[3].plot(idxpeak_AUC, rss_AUC[idxpeak_AUC], "r*", label="AUC-peaks")
    axs[3].plot(range(rss_AUC.shape[0]), rss_AUC, "k", "linewidth", 2, label="AUC")
    axs[3].set_title("AUCs")

    axs[4].plot(ats, label="ATS", color="black")
    axs[4].set_title("Activation time-series")

    plt.legend()
    plt.savefig(opj(DIR, "event_detection.png"), dpi=300)


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
    plt.savefig(opj(DIR, "event_detection_all.png"), dpi=300)


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
    # Perform event detection on BETAS
    print("Performing event-detection on betas...")
    DATA_file = opj(DIR, "pb06.sub-002ParkMabCm.denoised_no_censor_beta.nii.gz")
    (
        ets_beta,
        rss_beta,
        rssr_beta,
        idxpeak_beta,
        etspeaks_beta,
        mu_beta,
        _,
    ) = ev.event_detection(DATA_file, ATLAS_file, opj(TEMP, "surrogate_"), "_beta")

    # Perform event detection on ORIGINAL data
    print("Performing event-detection on original data...")
    DATA_file = opj(DIR, "pb06.sub-002ParkMabCm.denoised_no_censor.nii.gz")
    (
        ets_orig_sur,
        rss_orig_sur,
        rssr_orig_sur,
        idxpeak_orig_sur,
        etspeaks_orig_sur,
        mu_orig_sur,
        _,
    ) = ev.event_detection(DATA_file, ATLAS_file, opj(TEMP, "surrogate_"))

    # Perform event detection on FITTED signal
    print("Performing event-detection on fitted signal...")
    DATA_file = opj(DIR, "pb06.sub-002ParkMabCm.denoised_no_censor_fitt.nii.gz")
    (
        ets_fitt,
        rss_fitt,
        rssr_fitt,
        idxpeak_fitt,
        etspeaks_fitt,
        mu_fitt,
        _,
    ) = ev.event_detection(DATA_file, ATLAS_file, opj(TEMP, "surrogate_"))

    # Perform event detection on AUC
    print("Performing event-detection on AUC...")
    DATA_file = opj(DIR, "sub-002ParkMabCm_AUC_200.nii.gz")
    (
        ets_AUC,
        rss_AUC,
        rssr_AUC,
        idxpeak_AUC,
        etspeaks_AUC,
        mu_AUC,
        ets_AUC_denoised,
    ) = ev.event_detection(DATA_file, ATLAS_file, opj(TEMP, "surrogate_AUC_"))

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
        rss_AUC,
        rssr_AUC,
        idxpeak_AUC,
        ats,
    )

    # Plot all rss time series, null, and significant peaks in one plot
    plot_all(
        rss_orig_sur, idxpeak_orig_sur, rss_beta, idxpeak_beta, rss_fitt, idxpeak_fitt
    )

    # Plots ETS and denoised ETS
    plot_ets_matrix(ets_AUC, DIR, "_AUC_original")
    plot_ets_matrix(ets_AUC_denoised, DIR, "_AUC_denoised")

    # Save denoised ETS to npy for later use
    np.save(opj(DIR, "ets_AUC_denoised.npy"), ets_AUC_denoised)

    print("Denoised edge-time matrix computed and saved.")
    print("THE END")


if __name__ == "__main__":
    main()
