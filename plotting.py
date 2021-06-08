import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import join as opj

import ev

subject="sub-002ParkMabCm"
nROI="200"
DIR="/bcbl/home/public/PARK_VFERRER/PFM_data"
TEMP = "/bcbl/home/public/PARK_VFERRER/PFM_data/temp_"+subject+"_"+nROI
DATA_file=opj(DIR, "pb06.sub-002ParkMabCm.denoised_no_censor_beta.nii.gz")

ets_beta, rss_beta ,rssr_beta, idxpeak_beta, etspeaks_beta, mu_beta = ev.event_detection(DATA_file,opj(TEMP,'atlas.nii.gz'),opj(TEMP,'surrogate_'),'_beta')
DATA_file=opj(DIR, "pb06.sub-002ParkMabCm.denoised_no_censor.nii.gz")
# ets_, rss_orig, idxpeak_orig, etspeaks_orig, mu_orig = ev.event_detection(DATA_file,opj(TEMP,'atlas.nii.gz'))
ets_orig_sur, rss_orig_sur, rssr_orig_sur, idxpeak_orig_sur, etspeaks_orig_sur, mu_orig_sur = ev.event_detection(DATA_file,opj(TEMP,'atlas.nii.gz'),opj(TEMP,'surrogate_'))
DATA_file=opj(DIR, "pb06.sub-002ParkMabCm.denoised_no_censor_fitt.nii.gz")
ets_fitt, rss_fitt, rssr_fitt, idxpeak_fitt, etspeaks_fitt, mu_fitt = ev.event_detection(DATA_file,opj(TEMP,'atlas.nii.gz'),opj(TEMP,'surrogate_'))
DATA_file=opj(DIR, "sub-002ParkMabCm_AUC_200.nii.gz")
ets_AUC, rss_AUC, rssr_AUC, idxpeak_AUC, etspeaks_AUC, mu_AUC = ev.event_detection(DATA_file,opj(TEMP,'atlas.nii.gz'),opj(TEMP,'surrogate_AUC_'))

# plot rss time series, null, and significant peaks

# plt.legend([ph[0], qh],['null','significant','orig'])
# plot rss time series, null, and significant peaks
greymap = cm.get_cmap('Greys')
colors=greymap(np.linspace(0,0.65,rssr_orig_sur.shape[1]))
fig,axs=plt.subplots(4,1)
for i in range(rssr_orig_sur.shape[1]):
    axs[0].plot(range(rssr_orig_sur.shape[0]),rssr_orig_sur[:,i],color=colors[i])
axs[0].plot(idxpeak_orig_sur,rss_orig_sur[idxpeak_orig_sur],'r*',label='orig_sur-peaks')
axs[0].plot(range(rss_orig_sur.shape[0]),rss_orig_sur,'k','linewidth',2,label='orig_sur')
axs[0].set_title('Original signal')

for i in range(rssr_orig_sur.shape[1]):
    axs[1].plot(range(rssr_fitt.shape[0]),rssr_fitt[:,i],color=colors[i])
axs[1].plot(idxpeak_fitt,rss_fitt[idxpeak_fitt],'r*',label='fitt-peaks')
axs[1].plot(range(rss_fitt.shape[0]),rss_fitt,'k','linewidth',2,label='fitt')
axs[1].set_title('Fitted signal')

for i in range(rssr_orig_sur.shape[1]):
    axs[2].plot(range(rssr_beta.shape[0]),rssr_beta[:,i],color=colors[i])
axs[2].plot(idxpeak_beta,rss_beta[idxpeak_beta],'r*',label='beta-peaks')
axs[2].plot(range(rss_beta.shape[0]),rss_beta,'k','linewidth',2,label='beta')
axs[2].set_title('Betas')

for i in range(rssr_orig_sur.shape[1]):
    axs[3].plot(range(rssr_AUC.shape[0]),rssr_AUC[:,i],color=colors[i])
axs[3].plot(idxpeak_AUC,rss_AUC[idxpeak_AUC],'r*',label='AUC-peaks')
axs[3].plot(range(rss_AUC.shape[0]),rss_AUC,'k','linewidth',2,label='AUC')
axs[3].set_title('AUCs')

fig=plt.figure()
rss_orig_norm=(rss_orig_sur-rss_orig_sur.min())/(rss_orig_sur.max()-rss_orig_sur.min())
plt.plot(idxpeak_orig_sur,rss_orig_norm[idxpeak_orig_sur],'r*','linewidth',3,label='orig_sur-peaks')
plt.plot(range(rss_orig_norm.shape[0]),rss_orig_norm,'k','linewidth',3,label='orig_sur')
# plt.plot(idxpeak_orig,rss_orig[idxpeak_orig],'r*',label='orig-peaks')
# plt.plot(range(rss_orig.shape[0]),rss_orig,'k','linewidth',3,label='orig')
rss_beta_norm=(rss_beta-rss_beta.min())/(rss_beta.max()-rss_beta.min())
plt.plot(idxpeak_beta,rss_beta_norm[idxpeak_beta],'g*','linewidth',3, label='deconvolved_peaks')
plt.plot(range(rss_beta_norm.shape[0]),rss_beta_norm,'b','linewidth',3,label='deconvolved')
rss_fitt_norm=(rss_fitt-rss_fitt.min())/(rss_fitt.max()-rss_fitt.min())
plt.plot(idxpeak_fitt,rss_fitt_norm[idxpeak_fitt],'m*','linewidth',3, label='fitted_peaks')
plt.plot(range(rss_fitt_norm.shape[0]),rss_fitt_norm,'y','linewidth',3,label='fitted')
plt.legend()