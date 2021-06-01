import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import join as opj
from joblib import Parallel, delayed
from nilearn.input_data import NiftiLabelsMasker

def rss_surr(z_ts,u,v,surrprefix,masker,irand):
    [t,n] =z_ts.shape
    if surrprefix!='':
        zr=zscore(masker.fit_transform(f'{surrprefix}{irand}.nii.gz'),ddof=1)
    else:    
    # perform numrand randomizations
        zr = np.copy(z_ts)
        for i in range(n):
            zr[:,i] = np.roll(zr[:,i],np.random.randint(t))
    
    # edge time series with circshift data
    etsr = zr[:,u]*zr[:,v]
    
    # calcuate rss
    rssr = np.sqrt(np.sum(np.square(etsr), axis=1))
    return rssr


def event_detection(DATA_file,atlas,surrprefix='',segments=True):
    masker = NiftiLabelsMasker(
            labels_img=atlas,
            standardize=True,
            memory="nilearn_cache",
            strategy="mean",
        )

    data = masker.fit_transform(opj(DIR, DATA_file))
    # load and zscore time series
    z_ts=zscore(data,ddof=1)
    # Get number of time points/nodes
    [t,n] =z_ts.shape
    # upper triangle indices (node pairs = edges)
    u,v= np.argwhere(np.triu(np.ones(n),1)).T

    # edge time series
    ets = z_ts[:,u]*z_ts[:,v]

    # calculate rss
    #rss = sum(ets.^2,2).^0.5
    rss = np.sqrt(np.sum(np.square(ets), axis=1))

    # repeat with randomized time series
    numrand = 100
    # initialize array for null rss
    rssr = np.zeros([t,numrand])
    results = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(rss_surr)(z_ts,u,v,surrprefix,masker,irand) for irand in range(numrand))
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

    p = np.zeros([t,1])
    for i in range(t):
        p[i] = np.mean(np.reshape(rssr,rssr.shape[0]*rssr.shape[1]) >= rss[i])
    # apply statistical cutoff
    pcrit = 0.001

    # find frames that pass statistical testz_ts
    idx = np.argwhere(p < pcrit)[:,0]
    if segments:
        # identify contiguous segments of frames that pass statistical test
        dff = idx.T - range(len(idx))
        unq = np.unique(dff)
        nevents = len(unq)

        # find the peak rss within each segment
        idxpeak = np.zeros([nevents,1])
        for ievent in range(nevents):
            idxevent = idx[dff.T == unq[ievent].T]
            rssevent = rss[idxevent]
            idxmax = np.argmax(rssevent)
            idxpeak[ievent] = idxevent[idxmax]
        idxpeak=idxpeak[:,0].astype(int)
    # get activity at peak
    else:
        idxpeak=idx
    tspeaks = z_ts[idxpeak,:]

    # get co-fluctuation at peak
    etspeaks = tspeaks[:,u]*tspeaks[:,v]
    # calculate mean co-fluctuation (edge time series) across all peaks
    mu = np.nanmean(etspeaks,0)
    return rss, idxpeak, etspeaks, mu

subject="sub-002ParkMabCm"
nROI="200"
DIR="/bcbl/home/public/PARK_VFERRER/PFM_data"
TEMP = "/bcbl/home/public/PARK_VFERRER/PFM_data/temp_"+subject+"_"+nROI
DATA_file=opj(DIR, "pb06.sub-002ParkMabCm.denoised_no_censor_beta.nii.gz")

rss_beta, idxpeak_beta, etspeaks_beta, mu_beta = event_detection(DATA_file,opj(TEMP,'atlas.nii.gz'))
DATA_file=opj(DIR, "pb06.sub-002ParkMabCm.denoised_no_censor.nii.gz")
rss_orig, idxpeak_orig, etspeaks_orig, mu_orig = event_detection(DATA_file,opj(TEMP,'atlas.nii.gz'))
rss_orig_sur, idxpeak_orig_sur, etspeaks_orig_sur, mu_orig_sur = event_detection(DATA_file,opj(TEMP,'atlas.nii.gz'),opj(TEMP,'surrogate_'))
DATA_file=opj(DIR, "pb06.sub-002ParkMabCm.denoised_no_censor_fitt.nii.gz")
rss_fitt, idxpeak_fitt, etspeaks_fitt, mu_fitt = event_detection(DATA_file,opj(TEMP,'atlas.nii.gz'))
# plot rss time series, null, and significant peaks

# plt.legend([ph[0], qh],['null','significant','orig'])

plt.plot(idxpeak_orig_sur,rss_orig_sur[idxpeak_orig_sur],'b*',label='orig_sur-peaks')
plt.plot(range(rss_orig_sur.shape[0]),rss_orig_sur,'y','linewidth',2,label='orig_sur')
plt.plot(idxpeak_orig,rss_orig[idxpeak_orig],'r*',label='orig-peaks')
plt.plot(range(rss_orig.shape[0]),rss_orig,'k','linewidth',2,label='orig')
plt.plot(idxpeak_beta,rss_beta[idxpeak_beta],'g*', label='deconvolved_peaks')
plt.plot(range(rss_beta.shape[0]),rss_beta,'b','linewidth',2,label='deconvolved')
plt.plot(idxpeak_fitt,rss_fitt[idxpeak_fitt],'m*', label='fitted_peaks')
plt.plot(range(rss_fitt.shape[0]),rss_fitt,'y','linewidth',2,label='fitted')
plt.legend()