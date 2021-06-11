#!/usr/bin/python
import sys, argparse, os, socket, getpass, datetime
import numpy as np
import csv
import nibabel as nib
import subprocess
import shutil
import time
import scipy as sci
from scipy.signal import find_peaks
from hrf_matrix import HRFMatrix
from debiasing_cli import _get_parser
from debiasing_functions import debiasing_spike, debiasing_block


def read_volumes(paths, nTE):
    # Input checking
    if paths:
        if 'datasets' not in paths:
            raise ValueError(
                'No echo volumens were inserted. Please insert the path(s) to your echo volume(s)')
        if 'mask' not in paths:
            raise ValueError(
                'No mask volumens was inserted. Please insert the path to your mask volume')
    else:
        raise ValueError('No dictionary containing the paths to the files was inserted. Please '
                         'insert a dictionary with the paths to your files.')

    print('Reading files...')

    if type(paths['mask']) is list:
        nMasks = len(paths['mask'])
        mask_data = nib.load(os.path.join(paths['dir'], paths['mask'])).get_fdata()
        for maskidx in range(nMasks - 1):
            temp_mask = nib.load(
                paths['mask'][maskidx + 1]).get_fdata()
            mask_data = mask_data + temp_mask
    else:
        mask_data = nib.load(os.path.join(paths['dir'], paths['mask'])).get_fdata()

    # Reads header
    echo_hdr = nib.load(os.path.join(paths['dir'], paths['datasets'][0])).header

    nscans = 0
    nvoxels_orig = 0
    tr_value = 0

    for echo_idx in range(nTE):
        # Reads echo file
        echo_data = nib.load(os.path.join(paths['dir'], paths['datasets'][echo_idx])).get_fdata()

        # Gets dimensions of data
        if len(mask_data.shape) < len(echo_data.shape):
            dims = echo_data.shape
        else:
            dims = mask_data.shape
        # Checks nscans are the same among different time echos
        if echo_idx > 1 and nscans != dims[3]:
            raise ValueError(
                'Introduced files do not contain same number of scans. \n')
        else:
            nscans = dims[3]
        # Checks nvoxels are the same among different time echos
        if echo_idx > 1 and nvoxels_orig != np.prod(echo_hdr['dim'][1:4]):
            raise ValueError(
                'Introduced files do not contain same number of voxels. \n')
        else:
            nvoxels_orig = np.prod(echo_hdr['dim'][1:4])
        # Checks TR is the same among different time echos
        if echo_idx > 1 and tr_value != echo_hdr['pixdim'][4]:
            raise ValueError(
                'Introduced files do not contain same TR value. \n')
        else:
            tr_value = echo_hdr['pixdim'][4]

        # Initiates masked_data to make loop faster
        masked_data = np.zeros((dims[0], dims[1], dims[2], dims[3]))

        # Masks data
        if len(mask_data.shape) < 4:
            for i in range(nscans):
                masked_data[:, :, :, i] = np.squeeze(
                    echo_data[:, :, :, i]) * mask_data
        else:
            masked_data = echo_data * mask_data

        # Initiates data_restruct to make loop faster
        data_restruct_temp = np.reshape(np.moveaxis(
            masked_data, -1, 0), (nscans, nvoxels_orig))
        mask_idxs = np.unique(np.nonzero(data_restruct_temp)[1])
        data_restruct = data_restruct_temp[:, mask_idxs]

        # Concatenates different time echo signals to (nTE nscans x nvoxels)
        if echo_idx == 0:
            signal = np.zeros((len(mask_idxs), nscans, nTE))
        signal[:, :, echo_idx] = data_restruct.T.astype('float32')

        if echo_idx == 0:
            print(signal.shape[1], 'frames and',
                  signal.shape[0], 'voxels inside the mask')
        print(f'{int(np.round((echo_idx + 1) / nTE * 100))}%')

    print('Files successfully read')
    signal = np.squeeze(signal.T)

    output = {'signal': signal, 'mask': mask_idxs, 'tr': tr_value, 'dims': dims,
              'header': echo_hdr}

    return(output)


def reshape_and_mask2Dto4D(signal2d, dims, mask_idxs):
    signal4d = np.zeros((dims[0] * dims[1] * dims[2], signal2d.shape[0]))

    # Merges signal on mask indices with blank image
    for i in range(dims[3]):
        if len(mask_idxs.shape) > 3:
            idxs = np.where(mask_idxs[:, :, :, i] != 0)
        else:
            idxs = mask_idxs

        signal4d[idxs, i] = signal2d[i, :]

    # Reshapes matrix from 2D to 4D double
    signal4d = np.reshape(signal4d, (dims[0], dims[1], dims[2], signal2d.shape[0]))
    del signal2d, idxs, dims, mask_idxs
    return(signal4d)


def export_volumes(volumes, paths, header, nscans, history, te=None):

    if not 'output_dir' in paths:
        paths['output_dir'] = paths['dir']

    if 'LARS' in paths['auc']:
        lars_key = 'LARS.'
    else:
        lars_key = ''

    # R2* estimates
    if 'beta' in volumes:
        print('Exporting BETA estimates...')
        img = nib.nifti1.Nifti1Image(volumes['beta'], None, header=header)
        file_base = paths['auc'][:len(paths['auc']) - 7]
        auc_pos = file_base.find('AUC')
        file_base = file_base[:auc_pos-1]
        filename = os.path.join(paths['output_dir'], file_base + '.DR2.' + paths['key'] + 'nii.gz')
        nib.save(img, filename)
        print('BETA estimates correctly exported.')
        subprocess.run('3dcopy {} {} -overwrite'.format(filename, filename), shell=True)
        if history is not None:
            print('Updating file history...')
            subprocess.run('3dNotes -h "' + history + '" ' + filename, shell=True)
            print('File history updated.')

    # Fitted signal estimates
    if 'betafitts' in volumes:
        print('Exporting BETAFIITS estimates...')
        if te is not None:
            # Get TE values
            with open(paths['dir'] + paths['te'], 'r') as f:
                reader = csv.reader(f)
                te_list = list(reader)[0]

            te_values = [float(i)/1000 for i in te_list]
            nTE = len(te_values)
        else:
            nTE = 1

        if nTE > 1:
            for teidx in range(nTE):
                img = nib.nifti1.Nifti1Image(volumes['betafitts'][teidx*nscans+1:nscans*(teidx+1),:], None, header=header)
                file_base = paths['auc'][:len(paths['auc']) - 7]
                auc_pos = file_base.find('AUC')
                file_base = file_base[:auc_pos-1]
                filename = os.path.join(paths['output_dir'], file_base + '.dr2HRF_E0' + str(teidx+1) + '.' + paths['key'] + 'nii.gz')
                nib.save(img, filename)
                print('BETAFIITS estimates correctly exported.')
                subprocess.run('3dcopy {} {} -overwrite'.format(filename, filename), shell=True)
                if history is not None:
                    print('Updating file history...')
                    subprocess.run('3dNotes -h "' + history + '" ' + filename, shell=True)
                    print('File history updated.')
        else:
            img = nib.nifti1.Nifti1Image(volumes['betafitts'], None, header=header)
            file_base = paths['auc'][:len(paths['auc']) - 7]
            auc_pos = file_base.find('AUC')
            file_base = file_base[:auc_pos-1]
            filename = os.path.join(paths['output_dir'], file_base + '.dr2HRF.' + paths['key'] + 'nii.gz')
            nib.save(img, filename)
            print('BETAFIITS estimates correctly exported.')
            subprocess.run('3dcopy {} {} -overwrite'.format(filename, filename), shell=True)
            if history is not None:
                print('Updating file history...')
                subprocess.run('3dNotes -h "' + history + '" ' + filename, shell=True)
                print('File history updated.')

    # S0 estimates
    if 's_zero' in volumes:
        if volumes['s_zero'].size != 0:
            print('Exporting S0 estimates...')
            img = nib.nifti1.Nifti1Image(volumes['s_zero'], None, header=header)
            filename = os.path.join(paths['output_dir'], paths['auc'][:len(paths['auc']) - 7] + '.s0.' + paths['key'] + 'nii.gz')
            nib.save(img, filename)
            print('S0 estimates correctly exported.')
            subprocess.run('3dcopy {} {} -overwrite'.format(filename, filename), shell=True)
            if history is not None:
                print('Updating file history...')
                subprocess.run('3dNotes -h "' + history + '" ' + filename, shell=True)
                print('File history updated.')
        else:
            print('No S0 estimates to export.')

    # AUC timecourses
    if 'auc' in volumes:
        if volumes['auc'].size != 0:
            print('Exporting AUC timecourses...')
            img = nib.nifti1.Nifti1Image(volumes['auc'], None, header=header)
            filename = os.path.join(paths['output_dir'], paths['auc'][:len(paths['auc']) - 7] + '.AUC.THR.' + paths['key'] + 'nii.gz')
            nib.save(img, filename)
            print('AUC timecourses correctly exported in {}'.format(filename))
            subprocess.run('3dcopy {} {} -overwrite'.format(filename, filename), shell=True)
            if history is not None:
                print('Updating file history...')
                subprocess.run('3dNotes -h "' + history + '" ' + filename, shell=True)
                print('File history updated.')
        else:
            print('No AUC timecourses to export.')

def read_auc(paths, nscans, mask_idxs):

    auc_4d = nib.load(os.path.join(paths['dir'], paths['auc'])).get_fdata()
    auc_4d_hdr = nib.load(os.path.join(paths['dir'], paths['auc'])).header
    mask_data = nib.load(os.path.join(paths['dir'], paths['mask'])).get_fdata()

    nvoxels_orig = np.prod(auc_4d_hdr['dim'][1:4])

    # Gets dimensions of data
    if len(mask_data.shape) < len(auc_4d.shape):
        dims = auc_4d.shape
    else:
        dims = mask_data.shape

    # Initiates masked_data to make loop faster
    masked_data = np.zeros((dims[0], dims[1], dims[2], dims[3]))

    # Masks data
    if len(mask_data.shape) < 4:
        for i in range(nscans):
            masked_data[:, :, :, i] = np.squeeze(auc_4d[:, :, :, i]) * mask_data
    else:
        masked_data = auc_4d * mask_data

    # Initiates data_restruct to make loop faster
    auc_2d_temp = np.reshape(np.moveaxis(masked_data, -1, 0), (nscans, nvoxels_orig))
    # mask_idxs = np.unique(np.nonzero(auc_2d_temp)[1])
    auc_2d = auc_2d_temp[:, mask_idxs]

    return(auc_2d)


def debiasing(datasets, mask, auc, te=None, dir='', auc_dwm=None, mask_dwm=None, key='', afni=False, hrfmodel='SPMG1',
              hrfpath=None, thr=95, max_only=False, dist=2, ls=False, stc=False, cluster_size=10, block=False, fusion=False,
              nlambdas=20, fusion_lambda=3, fusion_gamma=0.5, groups=False, group_dist=3, history=''):

    if type(mask) is list:
        mask = mask[0]
    if type(auc) is list:
        auc = auc[0]
    if type(nlambdas) is list:
        nlambdas = nlambdas[0]
    if type(hrfmodel) is list:
        hrfmodel = hrfmodel[0]

    paths = {'datasets': datasets, 'mask': mask, 'dir': dir, 'auc': auc, 'key': key}

    if hrfpath is not None:
        if type(hrfpath) is list:
            hrfpath = os.path.join(paths['dir'], hrfpath[0])
        else:
            hrfpath = os.path.join(paths['dir'], hrfpath)

    if te is None:
        nTE = 1
    else:
        nTE = len(te)
    input_data = read_volumes(paths, nTE)
    orig_signal = input_data['signal']
    nscans = input_data['dims'][3]
    nvoxels = orig_signal.shape[1]

    # Generates HRF matrix
    hrf = HRFMatrix(TR=input_data['tr'], TE=te, nscans=nscans, r2only=True, has_integrator=False, is_afni=afni,
                           lop_hrf=hrfmodel, path=hrfpath, wfusion=fusion)
    hrf.generate_hrf()

    # Reads AUC
    auc = read_auc(paths, nscans, input_data['mask'])

    # Reads DWM AUC if both MASK and AUC of DWM are given
    if thr is not None:
        if mask_dwm is not None and auc_dwm is not None:
            print('Thresholding AUC...')
            paths['auc'] = auc_dwm
            paths['mask'] = mask_dwm
            auc_dwm = read_auc(paths, nscans, input_data['mask'])
            paths['mask'] = mask

            # Thresholding step
            thr_val = np.percentile(auc_dwm, thr)
            auc[auc <= thr_val] = 0
            print('Reshaping thresholded AUC timecourses from 2D to 4D...')
            auc_out = reshape_and_mask2Dto4D(auc, input_data['dims'], input_data['mask'])
            print('Thresholded AUC timecourses reshaping completed.')
            output_vols = {'auc': auc_out}
            export_volumes(output_vols, paths, input_data['header'], nscans, history)
        # Extracts DWM AUC from main dataset
        if mask_dwm is not None and auc_dwm is None:
            print('Thresholding AUC...')
            paths['mask'] = mask_dwm
            auc_dwm = read_auc(paths, nscans, input_data['mask'])
            paths['mask'] = mask

            # Thresholding step
            thr_val = np.percentile(auc_dwm, thr)
            auc[auc <= thr_val] = 0
            print('Reshaping thresholded AUC timecourses from 2D to 4D...')
            auc_out = reshape_and_mask2Dto4D(auc, input_data['dims'], input_data['mask'])
            print('Thresholded AUC timecourses reshaping completed.')
            output_vols = {'auc':auc_out}
            export_volumes(output_vols, paths, input_data['header'], nscans, history)

    if stc:
        print('Computing spatio-temporal clustering...')
        if 'LARS' in paths['auc']:
            lars_key = 'LARS.'
        else:
            lars_key = ''
        auc_filename = paths['output_dir'] + paths['datasets'][1:len(paths['datasets']) - 7] + '.pySPFM.' + lars_key + 'AUC.THR.' + paths['key'] + 'nii.gz'
        paths['key'] = 'STC.' + paths['key']
        stc_filename = paths['datasets'][1:len(paths['datasets']) - 7] + '.pySPFM.' + lars_key + 'AUC.THR.' + paths['key'] + 'nii.gz'
        stc_filename_full = paths['output_dir'] + stc_filename
        stc_command = 'python /export/home/eurunuela/public/MEPFM/spatio-temporal-clustering/stc.py --input {} --output {} --clustsize {} --thr {}'.format(
                    auc_filename, stc_filename_full, cluster_size, thr_val)
        subprocess.run(stc_command, shell=True)
        del auc, auc_out, auc_dwm
        paths['auc'] = '/' + stc_filename
        print('Reading AUC after spatio-temporal clustering...')
        auc = read_auc(paths, nscans, input_data['mask'])

    if block:
        # Initiates beta matrix
        beta = np.zeros((nscans, nvoxels))
        percentage_old = 0

        if ls and not max_only:
            print('Distance selected for peak finding: {}'.format(dist))

        print('Starting debiasing step...')
        print('0% debiased...')
        # Performs debiasing
        for vox_idx in range(nvoxels):
            # Keep only maximum values in AUC peaks
            if ls:
                # print('Finding peaks...')
                temp = np.zeros((auc.shape[0],))
                if max_only:
                    for timepoint in range(nscans):
                        if timepoint != 0:
                            if auc[timepoint, vox_idx] != 0:
                                if auc[timepoint - 1, vox_idx] != 0:
                                    if auc[timepoint, vox_idx] > auc[timepoint - 1, vox_idx]:
                                        max_idx = timepoint
                                else:
                                    max_idx = timepoint
                            else:
                                if auc[timepoint - 1, vox_idx] != 0:
                                    temp[max_idx] = auc[max_idx, vox_idx].copy()
                        else:
                            if auc[timepoint, vox_idx] != 0:
                                max_idx = timepoint
                else:
                    peak_idxs, _ = find_peaks(auc[:,vox_idx], prominence=thr_val, distance=dist)
                    temp[peak_idxs] = auc[peak_idxs, vox_idx].copy()

                auc[:, vox_idx] = temp.copy()

            beta[:, vox_idx], S = debiasing_block(auc[:, vox_idx], hrf.hrf_norm, orig_signal[:, vox_idx], ls)

            percentage = np.ceil((vox_idx+1)/nvoxels*100)
            if percentage > percentage_old:
                print('{}% debiased...'.format(int(percentage)))
                percentage_old = percentage

        # Calculates fitted signal
        betafitts = np.dot(hrf.hrf_norm, beta)

        if max_only:
            if 'LARS' in paths['auc']:
                lars_key = 'LARS.'
            else:
                lars_key = ''
            max_filename = paths['datasets'][1:len(paths['datasets']) - 7] + '.pySPFM.' + lars_key + 'AUC.THR.MAX.' + paths['key'] + 'nii.gz'
            paths['auc'] = '/' + max_filename
            auc_out = reshape_and_mask2Dto4D(auc, input_data['dims'], input_data['mask'])
            print('Thresholded AUC peaks timecourses reshaping completed.')
            output_vols = {'auc':auc_out}
            export_volumes(output_vols, paths, input_data['header'], nscans, history)
    else:

        if max_only:
            for vox_idx in range(nvoxels):
            # Keep only maximum values in AUC peaks
                temp = np.zeros((auc.shape[0],))
                if max_only:
                    for timepoint in range(nscans):
                        if timepoint != 0:
                            if auc[timepoint, vox_idx] != 0:
                                if auc[timepoint - 1, vox_idx] != 0:
                                    if auc[timepoint, vox_idx] > auc[timepoint - 1, vox_idx]:
                                        max_idx = timepoint
                                else:
                                    max_idx = timepoint
                            else:
                                if auc[timepoint - 1, vox_idx] != 0:
                                    temp[max_idx] = auc[max_idx, vox_idx].copy()
                        else:
                            if auc[timepoint, vox_idx] != 0:
                                max_idx = timepoint
                else:
                    peak_idxs, _ = find_peaks(auc[:,vox_idx], distance=dist)
                    temp[peak_idxs] = auc[peak_idxs, vox_idx].copy()

                auc[:, vox_idx] = temp.copy()

        print('Starting debiasing...')

        deb_output = debiasing_spike(hrf, orig_signal, auc, fusion, nlambdas, groups, group_dist)
        beta = deb_output['beta']
        betafitts = deb_output['betafitts']
        print('Debiasing finished.')

    # Transforms 2D to 4D
    print('Reshaping BETA timecourses from 2D to 4D...')
    beta_out = reshape_and_mask2Dto4D(beta, input_data['dims'], input_data['mask'])
    print('BETA timecourses reshaping completed.')

    print('Reshaping BETAFIITS timecourses from 2D to 4D...')
    betafitts_out = reshape_and_mask2Dto4D(betafitts, input_data['dims'], input_data['mask'])
    print('BETAFITTS timecourses reshaping completed.')

    # Dictionary to pass args to WriteVolumes
    output_vols = {'beta': beta_out, 'betafitts': betafitts_out}

    # Exports results
    export_volumes(output_vols, paths, input_data['header'], nscans, history)

    print('Debiasing finished. Your results were successfully exported.')


def _main(argv=None):
    """Entry point for aroma CLI."""
    # Command to save in file's history to be seen with the 3dinfo command
    options = _get_parser().parse_args(argv)

    args_str = str(options)[9:]
    history_str = '[{username}@{hostname}: {date}] python debiasing.py with {arguments}'.format(username=getpass.getuser(),\
                    hostname=socket.gethostname(), date=datetime.datetime.now().strftime('%c'), arguments=args_str)

    kwargs = vars(options)
    kwargs['history'] = history_str

    debiasing(**kwargs)


if __name__ == "__main__":
    _main()
