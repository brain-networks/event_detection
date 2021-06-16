"""Functions to perform event detection."""
import subprocess
from os.path import basename, join

import numpy as np
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import zscore

from Debiasing.debiasing_functions import debiasing_block, debiasing_spike
from Debiasing.hrf_matrix import HRFMatrix


def calculate_ets(y, n):
    """
    Calculate edge-time series.
    """
    # upper triangle indices (node pairs = edges)
    u, v = np.argwhere(np.triu(np.ones(n), 1)).T

    # edge time series
    ets = y[:, u] * y[:, v]

    return ets, u, v


def rss_surr(z_ts, u, v, surrprefix, sursufix, masker, irand):
    """
    Calculate RSS on surrogate data.
    """
    [t, n] = z_ts.shape

    if surrprefix != "":
        zr = masker.fit_transform(f"{surrprefix}{irand}{sursufix}.nii.gz")
        if "AUC" not in surrprefix:
            zr = np.nan_to_num(zscore(zr, ddof=1))
        zr = np.nan_to_num(zr)
    else:
        # perform numrand randomizations
        zr = np.copy(z_ts)
        for i in range(n):
            zr[:, i] = np.roll(zr[:, i], np.random.randint(t))

    # edge time series with circshift data
    etsr = zr[:, u] * zr[:, v]

    # calcuate rss
    rssr = np.sqrt(np.sum(np.square(etsr), axis=1))

    return rssr


def event_detection(DATA_file, atlas, surrprefix="", sursufix="", segments=True):
    """
    Perform event detection on given data.
    """
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize=False,
        memory="nilearn_cache",
        strategy="mean",
    )

    data = masker.fit_transform(DATA_file)
    # load and zscore time series
    # AUC does not get z-scored
    if "AUC" in surrprefix:
        z_ts = data
    else:
        z_ts = zscore(data, ddof=1)
    # Get number of time points/nodes
    [t, n] = z_ts.shape

    # calculate ets
    ets, u, v = calculate_ets(z_ts, n)

    # calculate rss
    rss = np.sqrt(np.sum(np.square(ets), axis=1))

    # repeat with randomized time series
    numrand = 100
    # initialize array for null rss
    rssr = np.zeros([t, numrand])

    results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(rss_surr)(z_ts, u, v, surrprefix, sursufix, masker, irand)
        for irand in range(numrand)
    )
    rssr = np.array(results).T
    rssr[0, :] = 0

    p = np.zeros([t, 1])
    rssr_flat = rssr.flatten()
    for i in range(t):
        p[i] = np.mean(rssr_flat >= rss[i])
    # apply statistical cutoff
    pcrit = 0.001

    if "AUC" in surrprefix:
        breakpoint()

    # find frames that pass statistical testz_ts
    idx = np.argwhere(p < pcrit)[:, 0]
    if segments:
        # identify contiguous segments of frames that pass statistical test
        dff = idx.T - range(len(idx))
        unq = np.unique(dff)
        nevents = len(unq)

        # find the peak rss within each segment
        idxpeak = np.zeros([nevents, 1])
        for ievent in range(nevents):
            idxevent = idx[dff.T == unq[ievent].T]
            rssevent = rss[idxevent]
            idxmax = np.argmax(rssevent)
            idxpeak[ievent] = idxevent[idxmax]
        idxpeak = idxpeak[:, 0].astype(int)
    # get activity at peak
    else:
        idxpeak = idx
    tspeaks = z_ts[idxpeak, :]

    # get co-fluctuation at peak
    etspeaks = tspeaks[:, u] * tspeaks[:, v]
    # calculate mean co-fluctuation (edge time series) across all peaks
    mu = np.nanmean(etspeaks, 0)

    if "AUC" in surrprefix:
        print("Reading AUC of surrogates to perform the thresholding step...")
        ets_surr = surrogates_to_array(surrprefix, sursufix, masker, numrand)
        ets_thr = threshold_ets_matrix(ets, ets_surr, idxpeak, percentile=99)
    else:
        ets_thr = None

    return ets, rss, rssr, idxpeak, etspeaks, mu, ets_thr, u, v


def threshold_ets_matrix(ets_matrix, surr_ets_matrix, selected_idxs, percentile):
    """
    Threshold the edge time-series matrix based on the selected time-points and
    the surrogate matrices.
    """

    # Initialize matrix with zeros
    thresholded_matrix = np.zeros(ets_matrix.shape)

    # Get selected columns from ETS matrix
    thresholded_matrix[selected_idxs, :] = ets_matrix[selected_idxs, :]

    # Calculate the percentile threshold from surrogate matrix
    thr = np.percentile(surr_ets_matrix, percentile)

    # Threshold ETS matrix based on surrogate percentile
    thresholded_matrix[thresholded_matrix < thr] = 0

    return thresholded_matrix


def surrogates_to_array(surrprefix, sursufix, masker, numrand=100):
    """
    Read AUCs of surrogates and concatenate into array.
    """
    for rand_i in range(numrand):
        auc = masker.fit_transform(f"{surrprefix}{rand_i}{sursufix}.nii.gz")
        [t, n] = auc.shape
        ets_temp, _, _ = calculate_ets(auc, n)

        if rand_i == 0:
            ets = np.zeros((numrand, len(ets_temp.flatten())))

        ets[rand_i, :] = ets_temp.flatten()

    return ets


def debiasing(data_file, mask, mtx, idx_u, idx_v, tr, out_dir, history_str):
    """
    Perform debiasing based on denoised edge-time matrix.
    """
    print("Performing debiasing based on denoised edge-time matrix...")
    masker = NiftiLabelsMasker(
        labels_img=mask,
        standardize=False,
        memory="nilearn_cache",
        strategy="mean",
    )

    # Read data
    data = masker.fit_transform(data_file)

    # Generate mask of significant edge-time connections
    ets_mask = np.zeros(data.shape)
    idxs = np.where(mtx != 0)
    time_idxs = idxs[0]
    edge_idxs = idxs[1]

    print("Generating mask of significant edge-time connections...")
    for idx, time_idx in enumerate(time_idxs):
        ets_mask[time_idx, idx_u[edge_idxs[idx]]] = 1
        ets_mask[time_idx, idx_v[edge_idxs[idx]]] = 1

    # Create HRF matrix
    hrf = HRFMatrix(
        TR=tr,
        TE=[0],
        nscans=data.shape[0],
        r2only=True,
        has_integrator=False,
        is_afni=True,
    )
    hrf.generate_hrf()

    # Perform debiasing
    deb_output = debiasing_spike(hrf, data, ets_mask)
    beta = deb_output["beta"]
    fitt = deb_output["betafitts"]

    # Transform results back to 4D
    beta_4D = masker.inverse_transform(beta)
    beta_file = join(out_dir, f"{basename(data_file[:-7])}_beta_ETS.nii.gz")
    beta_4D.to_filename(beta_file)
    subprocess.run(f"3dNotes {join(out_dir, beta_file)} -h {history_str}", shell=True)

    fitt_4D = masker.inverse_transform(fitt)
    fitt_file = join(out_dir, f"{basename(data_file[:-7])}_fitt_ETS.nii.gz")
    fitt_4D.to_filename(fitt_file)
    subprocess.run(f"3dNotes {join(out_dir, fitt_file)} -h {history_str}", shell=True)

    print("Debiasing finished and files saved.")

    return beta, fitt
