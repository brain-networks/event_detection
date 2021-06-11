# Cleveland Clinic - BCBL Epilepsy7T project

- [Cleveland Clinic - BCBL Epilepsy7T project](#cleveland-clinic---bcbl-epilepsy7t-project)
  - [Description of files (for each run)](#description-of-files-for-each-run)
  - [Scripts and how to use them](#scripts-and-how-to-use-them)
    - [*hist.py*](#histpy)
    - [*stc_and_debias.sh*](#stc_and_debiassh)
    - [Dependencies](#dependencies)
      - [*stc.py*](#stcpy)
      - [*debiasing.py*](#debiasingpy)
      - [*hrf_matrix.py*](#hrf_matrixpy)

## Description of files (for each run)

Volumes that are common to both methods:

- **<p style="color: #F39C12">S41vol.mask.nii.gz</p>** Whole brain binary mask of functional data. It is used to reduce the dimensionality of calculations.
- **<p style="color: #F39C12">S41vol.mask.DWM.nii.gz</p>** Binary mask of the deep white matter in the functional space. White matter tissue segmentation is computed with 3dSeg, binarized (probability > 0.6 of the Segsy/Classes+orig) and eroded by 3 voxels. This mask is used to calculate the threshold value that is applied on the AUC dataset.
- **<p style="color: #F39C12">S41vol.orig.smooth.det.spc.nii.gz</p>** Functional data after smoothing (smooth), detrending (det) and converting into signal percentage change (spc). This is the input to the PFM algorithms. It is automatically used by the debiasing script if the filename is not modified. 
  
  **In study478, the analysis were done in <span style="color: #F39C12">S41vol.denoised.smooth.det.spc.nii.gz</span> volume, where the original dataset was denoised by regressing out the time series of 1 independent component (FLS MELODIC also included in the folder) that was clearly identified as an artefact (see bad_components.txt).**

Volumes specific to 3dPFM (can be checked with 3dPFM -help):

```bash
3dPFM -input S41vol.orig.smooth.det.spc.nii.gz -mask S41vol.mask.nii.gz -criteria bic -algorithm lasso -hrf SPMG1 -maxiterfactor 0.3 -jobs 10 -outZAll lasso_BIC_SPMG1
```

- **<p style="color: #F39C12">S41vol.lasso.BIC.SPMG1.beta.nii.gz</p>** Debiased estimates of the neuronal-related time series (i.e. events). Same length as input data.
- **<p style="color: #F39C12">S41vol.lasso.BIC.SPMG1.betafitts.nii.gz</p>** Convolved debiased estimates neuronal-related time series (i.e. events convolved with HRF model). Same length as input data.
- **<p style="color: #F39C12">S41vol.lasso.BIC.SPMG1.lambda.nii.gz</p>** Map of the regularization parameter used in the deconvolution of each voxel. Selected according to Bayesian Information Criterion.
- **<p style="color: #F39C12">S41vol.lasso.BIC.SPMG1.Z_Tstats_beta.nii.gz</p>** Dataset with z-scores of the T-statistics of beta. Same length as input data.
- **<p style="color: #F39C12">S41vol.lasso.BIC.SPMG1.Z_Fstats_beta.nii.gz</p>** Map of Dataset of z-scores of the F-statistics of the deconvolved component. Same length as input data.


New PFM approach with Stability Selection:

- **<p style="color: #F39C12">S41vol.pySPFM.LARS.AUC.nii.gz</p>** Dataset containing the area under the curve (AUC) of the stability path. It is a 4D dataset explaining the probability of a neuronal-related event happening in each voxel and timepoint. It is used to obtain the debiased estimates of *beta* and *betafitts* once it is thresholded and spatio-temporal clustering (optional) is performed on it. Same length as input data.
- **<p style="color: #F39C12">S41vol.pySPFM.LARS.beta_99.nii.gz</p>** Debiased estimates of *beta* after thresholding the AUC dataset with the 99th percentile. No spatio-temporal clustering was performed to get this results. Same length as input data.
- **<p style="color: #F39C12">S41vol.pySPFM.LARS.betafitts_99.nii.gz</p>** Debiased estimates of *betafitts* after thresholding the AUC dataset with the 99th percentile. No spatio-temporal clustering was performed to get this results.
- **<p style="color: #F39C12">S41vol.pySPFM.LARS.STC.beta_99.nii.gz</p>** Debiased estimates of *beta* after thresholding the AUC dataset with the 99th percentile and performing spatio-temporal clustering (STC). A minimum cluster size of 10 voxels was used. The sliding window's size depended on the acquisition TR. A sliding window of -2/+2 TRs was used when TR=0.5, while a sliding window of -3/+3 TRs was used when TR=0.3. Same length as input data.
- **<p style="color: #F39C12">S41vol.pySPFM.LARS.STC.betafitts_99.nii.gz</p>** Debiased estimates of *betafitts* after thresholding the AUC dataset with the 99th percentile and performing spatio-temporal clustering (STC). A minimum cluster size of 10 voxels was used. The sliding window's size depended on the acquisition TR. A sliding window of -2/+2 TRs was used when TR=0.5, while a sliding window of -3/+3 TRs was used when TR=0.3.
- **<p style="color: #F39C12">S41vol_thr.csv</p>** Comma separeted file with the calculated percentile values. The header shows the percentile value (e.g. 99) and the row below shows the AUC value corresponding to the percentile (e.g. 0.401).

---

## Scripts and how to use them

This section aims to guide you through the two commands needed to obtain the estimated beta and betafits with the new PFM algorithm using Stability Selection (https://doi.org/10.1111/j.1467-9868.2010.00740.x). The new PFM approach returns the AUC timeseries described in the section above. This AUC timeseries describes the probability of a neuronal-event ocurring for each voxel and moment in time. Thus, this is not the real estimate of the neuronal-events, which must be computed with a debiasing step. The following two scripts take care of that.

> *CAUTION! The examples shown below always assume you run the scripts from within the directory containing the data.*

### *hist.py*

First, given that the AUC is a timeseries with no zero values by definition with the current procedure that performs the deconvolution using Least Angle Regression (LARS), a threshold must be applied to select the true neuronal-related events (i.e. beta) and their corresponding generated HRF (i.e. betafitts). This first script aims to provide a way of selecting such threshold. The script calculates the provided percentiles of the AUC timeseries in a region of no interest (here, Deep White Matter voxels) and saves them to a CSV file called *thr.csv*. By default, the script calculates the 95th, 99th and 100th percentiles, but other percentiles can be calculated too. You can find the help of the script using the `-h` option.

Using the script is very easy: just provide the AUC and the masks, and you're good to go! Two masks are needed though. First, a whole-brain mask, and second, a mask of the region of no interest from which the thresholds are calculted.

```python
python /path/to/script/hist.py --input AUC --mask REGION_OF_NO_INTEREST_MASK
```
For example, if we wanted to get the 95th, 99th and 100th percentiles (default) from run S41vol with the deep white matter (DWM) as our region of no interest, we would do:

```python
python /path/to/script/hist.py --input S41vol.pySPFM.LARS.AUC.nii.gz --mask S41vol.mask.DWM.nii.gz
```

The example would generate a file named *thr.csv*. We could rename it to avoid overwriting the file; e.g. with the following command:

```bash
mv thr.csv S41vol_thr.csv
```

As mentioned, percentiles different from the default ones can also be provided. To do so, we would only need to add the `--percentile` option and the values as shown below:

```python
python /path/to/script/hist.py --input S41vol.pySPFM.LARS.AUC.nii.gz --mask S41vol.mask.DWM.nii.gz --percentile 96 97 98
```

This last example would generate a file named *thr.csv* with the 96th, 97th and 98th percentiles.

### *stc_and_debias.sh*

> *CAUTION! Internally, this script calls *stc.py* and *debiasing.py*. Make sure that the path to these dependencies is correct in *stc_and_debias.sh*.*

Once we have decided a threshold, it's time to apply it on our AUC timeseries to obtain the beta and betafitts volumes. This script takes care of not only the debiasing step, but also performs the thresholding and spatio-temporal clustering. Internally, it is a bit more complicated than the previous one, but its usage is very simple. It relies on the correct naming of the files to work properly with no interaction from our part. The filenames could be edited in the script, but we suggest sticking to the "filename system" we provided to avoid any issues with the script. This will be clearer once we see the example below.

In order to obtain the beta and betafitts datasets from the AUC timeseries, this script performs the following three steps:

1. Thresholding the AUC timeseries (in the stc.py dependency).
2. Spatio-temporal clustering of the thresholded AUC timeseries (in the stc.py dependency).
3. Debiasing step based on the non-zero coefficients of the thresholded and clustered AUC timeseries (in the debiasing.py dependency).

> *The dependencies are briefly described at the end of this document.*

Let us see what arguments are needed to run this script, which are input via the terminal or could also be manually added to the script. Given that thresholds are saved in *thr.csv* we encourage you to pass the arguments through the terminal to avoid any errors when saving modifications to the script.

The script must be run as follows:

```bash
sh /path/to/script/stc_and_debias.sh PROJECT_DIRECTORY RUN LABEL THRESHOLD WINDOW_SIZE CLUSTER_SIZE
```

where the arguments are:

* PROJECT_DIRECTORY: Folder where the AUC timeseries, the SPC data and the masks are saved. If you're already in this directory, you should pass the following argument: `$(pwd)`.
* RUN: This argument refers to the dataset run; i.e. the beginning of the file, e.g. `S41vol`.
* LABEL: The label refers to the ending used in naming the new files that are generated by the script. It is useful to differentiate among the different thresholds used for example. We use the percentile value in our results, e.g. `99`.
* THRESHOLD: The value used to threshold the AUC timeseries, e.g. `0.401`.
* WINDOW_SIZE: The size of the (temporal) sliding window used in the spatio-temporal clustering method. The window will always be symmetrical; i.e. if a value of `2` is given, the window size will go from -2 to +2 TRs.
* CLUSTER_SIZE: The minimum number of voxels to form a clusters, otherwise the events are nulled; e.g. `10`.

For example, the command could be as follows:

```bash
sh /path/to/script/stc_and_debias.sh $(pwd) S41vol 99 0.401 2 10
```

>*Please, consider that the lower the threshold, the more non-zero coefficients will be used and the longer the computation time will be.*

The script will generate the following files:

**<p style="color: #F39C12"> S41vol.pySPFM.LARS.AUC.THR.STC_99.nii.gz</p>**
**<p style="color: #F39C12"> S41vol.pySPFM.LARS.STC.beta_99.nii.gz</p>**
**<p style="color: #F39C12"> S41vol.pySPFM.LARS.STC.betafitts_99.nii.gz</p>**

**CONGRATULATIONS!** You have computed the debiased results of the stability-based (new) PFM algorithm!

A final note: the thresholding value can be freely selected without the use of *hist.py*, the purpose of which is to facilitate and make the selection of the threshold data-driven. Every dataset will have a different optimal threshold, so feel free to not use *hist.py* and directly provide *stc_and_debias.sh* a threshold of your choice.

### Dependencies

> *The dependencies should not be edited for the correct functioning of the previous scripts.*

#### *stc.py*

This Python script creates a bash script with three AFNI commands that perform the spatio-temporal clustering.

#### *debiasing.py*

This Python script solves the least squares problem with the non-zero coefficients to provide a debiased estimation of the betas and betaffits.

#### *hrf_matrix.py*

This Python class is a dependency for *debiasing.py*. It takes care of generating the design matrix with the haemodynamic response function that is necessary to find the debiased estimates via orthogonal least squares.
