#!/bin/bash
#$ -cwd
#$ -o deb_out.txt
#$ -e deb_err.txt
#$ -S /bin/bash
#$ -q short.q

module load afni/latest
module load python/python3

cd /export/home/eurunuela/public/MEPFM/pySPFM_public/Debiasing

# Just to avoid typing/looking for the command when debiasing:
python -u debiasing.py -i S51vol.orig.det.smooth3.spc.nii.gz -m S51vol.mask.nii.gz --dir /export/home/eurunuela/public/Epilepsy7T/study488/S51dt --auc S51vol.pySPFM.LARS.AUC.THR_99.nii.gz -afni -fusion -fusion_nlambdas 50

cd /export/home/eurunuela/public/Epilepsy7T/study488/S51dt

3dcalc -a S51vol.pySPFM.LARS.DR2.nii.gz -b S51vol.mask.nii.gz -expr 'b*(a+0.0000001)' -prefix S51vol.pySPFM.LARS.beta_99.FUSION.nii.gz -overwrite
rm S51vol.pySPFM.LARS.DR2.nii.gz

3dcalc -a S51vol.pySPFM.LARS.dr2HRF.nii.gz -b S51vol.mask.nii.gz -expr 'b*(a+0.0000001)' -prefix S51vol.pySPFM.LARS.betafitts_99.FUSION.nii.gz -overwrite
rm S51vol.pySPFM.LARS.dr2HRF.nii.gz