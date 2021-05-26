import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib import cm
from os.path import join as opj
from nilearn.input_data import NiftiLabelsMasker

subject="sub-002ParkMabCm"
nROI="200"
DIR="/bcbl/home/public/PARK_VFERRER/PFM_data"
TEMP = "/bcbl/home/public/PARK_VFERRER/PFM_data/temp_"+subject+"_"+nROI
DATA_file=opj(DIR, "pb06.sub-002ParkMabCm.denoised_no_censor.nii.gz")

masker = NiftiLabelsMasker(
        labels_img=opj(TEMP, "atlas.nii.gz"),
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
#rss = sum(ets.^2,2).^0.5;
rss = np.sqrt(np.sum(np.square(ets), axis=1))

# repeat with randomized time series
numrand = 100;
# initialize array for null rss
rssr = np.zeros([t,numrand]);

# perform numrand randomizations
for irand in range(numrand):
    # create circularly shifted time series
    zr = np.copy(z_ts);
    for i in range(n):
        zr[:,i] = np.roll(zr[:,i],np.random.randint(t))
    
    # edge time series with circshift data
    etsr = zr[:,u]*zr[:,v]
    
    # calcuate rss
    rssr[:,irand] = np.sqrt(np.sum(np.square(etsr), axis=1))

p = np.zeros([t,1]);
for i in range(t):
    p[i] = np.mean(np.reshape(rssr,rssr.shape[0]*rssr.shape[1]) >= rss[i])
# apply statistical cutoff
pcrit = 0.001

# find frames that pass statistical test
idx = np.argwhere(p < pcrit)[:,0]

# identify contiguous segments of frames that pass statistical test
dff = idx.T - range(len(idx))
unq = np.unique(dff)
nevents = len(unq)

# find the peak rss within each segment
idxpeak = np.zeros([nevents,1]);
for ievent in range(nevents):
    idxevent = idx[dff.T == unq[ievent].T]
    rssevent = rss[idxevent]
    idxmax = np.argmax(rssevent)
    idxpeak[ievent] = idxevent[idxmax]
idxpeak=idxpeak[:,0].astype(int)
# get activity at peak
tspeaks = z_ts[idxpeak,:]

# get co-fluctuation at peak
etspeaks = tspeaks[:,u]*tspeaks[:,v]

# plot rss time series, null, and significant peaks
greymap = cm.get_cmap('Greys')
colors=greymap(np.linspace(0,0.65,rssr.shape[1]))
plt.figure()
for i in range(rssr.shape[1]):
    ph=plt.plot(range(t),rssr[:,i],color=colors[i]);
qh=plt.plot(idxpeak,rss[idxpeak],'r*',range(t),rss,'k','linewidth',2);
plt.xlim([1,t])
plt.xlabel('frame'); 
plt.ylabel('rss');
# plt.legend([ph[0], qh],['null','significant','orig']);

