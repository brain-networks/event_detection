from os.path import join as opj

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import cm
from nilearn.input_data import NiftiLabelsMasker
from scipy.io import loadmat
from scipy.stats import zscore


def rss_surr(z_ts, u, v, surrprefix, sursufix, masker, irand):
    [t, n] = z_ts.shape
    if surrprefix != "":
        zr = zscore(
            masker.fit_transform(f"{surrprefix}{irand}{sursufix}.nii.gz"), ddof=1
        )
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
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize=True,
        memory="nilearn_cache",
        strategy="mean",
    )

    data = masker.fit_transform(DATA_file)
    # load and zscore time series
    z_ts = zscore(data, ddof=1)
    # Get number of time points/nodes
    [t, n] = z_ts.shape
    # upper triangle indices (node pairs = edges)
    u, v = np.argwhere(np.triu(np.ones(n), 1)).T

    # edge time series
    ets = z_ts[:, u] * z_ts[:, v]

    # calculate rss
    # rss = sum(ets.^2,2).^0.5
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
    # for irand in range(numrand):
    #     if surrprefix!='':
    #         zr=zscore(masker.fit_transform(f'{surrprefix}{irand}.nii.gz'),ddof=1)
    #     else:
    # # perform numrand randomizations
    #         zr = np.copy(z_ts)
    #         for i in range(n):
    #             zr[:,i] = np.roll(zr[:,i],np.random.randint(t))

    #     # edge time series with circshift data
    #     etsr = zr[:,u]*zr[:,v]

    #     # calcuate rss
    #     rssr[:,irand] = np.sqrt(np.sum(np.square(etsr), axis=1))

    p = np.zeros([t, 1])
    for i in range(t):
        p[i] = np.mean(np.reshape(rssr, rssr.shape[0] * rssr.shape[1]) >= rss[i])
    # apply statistical cutoff
    pcrit = 0.001

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
    return ets, rss, rssr, idxpeak, etspeaks, mu


def threshold_ets_matrix(ets_matrix, surr_ets_matrix, selected_idxs, percentile):
    """
    Threshold the edge time-series matrix based on the selected time-points and the surrogate matrices.
    """

    # Initialize matrix with zeros
    thresholded_matrix = np.zeros(ets_matrix.shape)

    # Get selected columns from ETS matrix
    thresholded_matrix[:, selected_idxs] = ets_matrix[:, selected_idxs]

    # Calculate the percentile threshold from surrogate matrix
    thr = np.percentile(surr_ets_matrix, percentile)

    # Threshold ETS matrix based on surrogate percentile
    thresholded_matrix[thresholded_matrix < thr] = 0

    return thresholded_matrix
