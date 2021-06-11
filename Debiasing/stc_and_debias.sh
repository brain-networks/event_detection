#!bin/bash

# Input arguments (replace $1 with actual value if running from terminal is not desired)
PRJDIR=$1 # /export/home/eurunuela/public/Epilepsy7T/study297 path to the data directory. Use $(pwd) if already in the directory.
RUN=$2 # e.g. S43vol; the script already contains the rest of the filenames.
LBL=$3 # label the output filename contains
THR=$4 # value to threshold the AUC timeseries with
WIN=$5 # e.g. 2; the sliding window will be from -2 to +2 TRs big
CSZ=$6 # minimum amount of voxels to form a cluster

# Moves to project directory containing data
cd $PRJDIR

#####################################
# Performs the thresholding and spatio-temporal-clustering step
echo "Thresholding at ${THR} and STC on ${RUN}..."
python /export/home/eurunuela/public/MEPFM/pySPFM/stc.py --input ${RUN}.orig.smooth.det.spc.pySPFM.LARS.AUC.nii.gz --output ${RUN}.pySPFM.LARS.AUC.THR.STC_${LBL}.nii.gz \
--w_pos ${WIN} --w_neg ${WIN} --clustsize ${CSZ} --thr ${THR}

#####################################
# Performs the debiasing step
echo "Debiasing ${RUN}..."
python -u /export/home/eurunuela/public/MEPFM/pySPFM/debiasing.py --datasets /${RUN}.orig.smooth.det.spc.nii.gz --mask /${RUN}.mask.nii.gz \
--dir ${PRJDIR} --auc /${RUN}.pySPFM.LARS.AUC.THR.STC_${LBL}.nii.gz --afni --key ${LBL}

#####################################
# Adds a very small value to all voxels to make them non-zero (only for visualization purposes)
# Also renames file to contain STC label
3dcalc -a ${PRJDIR}/${RUN}.pySPFM.LARS.beta.${LBL}.nii.gz -b ${PRJDIR}/${RUN}.mask.nii.gz -expr 'b*(a+0.0000001)' \
-prefix ${PRJDIR}/${RUN}.pySPFM.LARS.STC.beta_${LBL}.nii.gz -overwrite

rm ${PRJDIR}/${RUN}.pySPFM.LARS.beta.${LBL}.nii.gz

3dcalc -a ${PRJDIR}/${RUN}.pySPFM.LARS.betafitts.${LBL}.nii.gz -b ${PRJDIR}/${RUN}.mask.nii.gz -expr 'b*(a+0.0000001)' \
-prefix ${PRJDIR}/${RUN}.pySPFM.LARS.STC.betafitts_${LBL}.nii.gz -overwrite

rm ${PRJDIR}/${RUN}.pySPFM.LARS.betafitts.${LBL}.nii.gz