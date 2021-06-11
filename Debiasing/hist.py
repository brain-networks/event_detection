import csv
import numpy as np
import nibabel as nib
import sys, argparse, os, socket, getpass, datetime


def read_vol(vol_file, mask_file):
    vol_4d = nib.load(vol_file).get_fdata()
    vol_4d_hdr = nib.load(vol_file).header
    mask_data = nib.load(mask_file).get_fdata()

    nvoxels_orig = np.prod(vol_4d_hdr["dim"][1:4])

    # Gets dimensions of data
    if len(mask_data.shape) < len(vol_4d.shape):
        dims = vol_4d.shape
    else:
        dims = mask_data.shape

    nscans = dims[3]

    # Initiates masked_data to make loop faster
    masked_data = np.zeros((dims[0], dims[1], dims[2], dims[3]))

    # Masks data
    if len(mask_data.shape) < 4:
        for i in range(nscans):
            masked_data[:, :, :, i] = np.squeeze(vol_4d[:, :, :, i]) * mask_data
    else:
        masked_data = vol_4d * mask_data

    # Initiates data_restruct to make loop faster
    vol_2d_temp = np.reshape(np.moveaxis(masked_data, -1, 0), (nscans, nvoxels_orig))
    mask_idxs = np.unique(np.nonzero(vol_2d_temp)[1])
    vol_2d = vol_2d_temp[:, mask_idxs]

    return vol_2d


def main(argv):

    # Argparser
    parser = argparse.ArgumentParser(
        description="Calculates thresholds for AUC data based on percentile of given mask."
    )
    parser.add_argument("--input", type=str, nargs=1, help="Filenames of the AUC data.")
    parser.add_argument(
        "--mask", type=str, nargs=1, help="Filenames of the mask to threshold the AUC."
    )
    parser.add_argument(
        "--percentile",
        type=int,
        nargs="+",
        help=("Percentiles " "to calculate (default=95 99 100)."),
        default=[95, 99, 100],
    )
    args = parser.parse_args()

    # Read files and reshape them to 2D
    print("Reading AUC file and applying mask...")
    data = read_vol(args.input[0], args.mask[0])

    header = list()
    values = list()
    for p_idx in range(len(args.percentile)):
        pctl = np.percentile(data, args.percentile[p_idx])
        print("Percentile {} is: {}".format(args.percentile[p_idx], pctl))
        header.append(args.percentile[p_idx])
        values.append(pctl)

    with open("thr.csv", mode="w") as thr_file:
        thr_writer = csv.writer(
            thr_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )

        thr_writer.writerow(header)
        thr_writer.writerow(values)


if __name__ == "__main__":
    main(sys.argv[1:])
