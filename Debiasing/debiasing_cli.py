#!/usr/bin/env python
"""Parser for debiasing workflow."""
import argparse


def _get_parser():
    """
    Parse command line inputs for aroma.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser(
        description=(
            "Script to run the debiasing step after PFM. "
            "See the companion manual for further information."
        )
    )

    # Required options
    reqoptions = parser.add_argument_group("Required arguments")
    reqoptions.add_argument(
        "-i", "--input",
        dest="datasets",
        required=True,
        help="Input datasets.",
        nargs='+'
    )
    reqoptions.add_argument(
        "-m", "--mask",
        dest="mask",
        required=True,
        help="Input datasets.",
        nargs=1
    )
    reqoptions.add_argument(
        "-auc", "--auc",
        dest="auc",
        required=True,
        help="File name of the AUC results.",
        nargs=1
    )

    # Optional options
    optoptions = parser.add_argument_group("Optional arguments")
    optoptions.add_argument(
        "-te",
        "--te",
        dest="te",
        type=list,
        default=None,
        help=(
            "List of echo times (default=None). If no TE is given, "
            "the single echo version will be run."
        ),
        nargs=1
    )
    optoptions.add_argument(
        "-d",
        "--dir",
        dest="dir",
        type=str,
        default='.',
        help=(
            "Main directory with datasets (default=cwd)."
        )
    )
    optoptions.add_argument(
        "-auc_dwm",
        "--auc_dwm",
        dest="auc_dwm",
        type=str,
        default=None,
        help=(
            "File name of the Deep White Matter's AUC results. It is used for thresholdind purposes (default+None)."
        ),
        nargs=1
    )
    optoptions.add_argument(
        "-mask_dwm",
        "--mask_dwm",
        dest="mask_dwm",
        type=str,
        default=None,
        help=(
            "File name of the mask of voxels in the Deep White Matter. It is used for thresholdind purposes (default+None)."
        ),
        nargs=1
    )
    optoptions.add_argument(
        "-key",
        "--key",
        dest="key",
        type=str,
        default='',
        help=(
            "String that the output filenames will contain at their end. (default="")."
        ),
        nargs=1
    )
    optoptions.add_argument(
        "-afni",
        "--afni",
        dest="afni",
        action="store_true",
        help="Whether the HRF matrix is generated using 3dDeconvolve from AFNI (default=False).",
        default=False,
    )
    optoptions.add_argument(
        "-hrfmodel",
        "--hrfmodel",
        dest="hrfmodel",
        type=str,
        default="SPMG1",
        help="HRF model for 3dDeconvolve from AFNI (default='SPMG1').",
        nargs=1
    )
    optoptions.add_argument(
        "-hrfpath",
        "--hrfpath",
        dest="hrfpath",
        type=str,
        default=None,
        help="Path to custom HRF model (default=None).",
        nargs=1
    )
    optoptions.add_argument(
        "-thr",
        "--thr",
        dest="thr",
        type=int,
        default=95,
        help="Percentil value (integer) of the DWM AUC to apply the threshold with (default=95).",
        nargs=1
    )
    optoptions.add_argument(
        "-max",
        "--max",
        dest="max_only",
        action="store_true",
        help="Whether to only keep maximum values in AUC peaks (default=False).",
        default=False,
    )
    optoptions.add_argument(
        "-dist",
        "--dist",
        dest="dist",
        type=int,
        default=2,
        help="Minimum distance between two AUC peaks (default=2).",
        nargs=1
    )
    optoptions.add_argument(
        "-ls",
        "--ls",
        dest="ls",
        action="store_true",
        help="Whether to use least squares to apply debiasing on the block model (default=False). It is the fastest option. "
             "Otherwise, ridge regression with cross validation is used for further stability.",
        default=False,
    )
    optoptions.add_argument(
        "-stc",
        "--stc",
        dest="stc",
        action="store_true",
        help="Whether to compute Spatio-Temporal Clustering on the thresholded AUC before debiasing (default=False).",
        default=False,
    )
    optoptions.add_argument(
        "-cluster_size",
        "--cluster_size",
        dest="cluster_size",
        type=int,
        default=10,
        help="Cluster size in STC (default=10).",
        nargs=1
    )
    optoptions.add_argument(
        "-block",
        "--block",
        dest="block",
        action="store_true",
        help="Whether the AUC was calculated with the block model formulation (default=False).",
        default=False,
    )
    optoptions.add_argument(
        "-fusion",
        "--fusion",
        dest="fusion",
        action="store_true",
        help="Whether to use the weighted fusion formulation (default=False).",
        default=False,
    )
    optoptions.add_argument(
        "-fusion_nlambdas",
        "--fusion_nlambdas",
        dest="nlambdas",
        type=int,
        default=20,
        help="Numer of lambdas for fusion debiasing cross validation (default=20).",
        nargs=1
    )
    optoptions.add_argument(
        "-fusion_lambda",
        "--fusion_lambda",
        dest="fusion_lambda",
        type=float,
        default=3,
        help="Value of lambda for fusion debiasing (default=3).",
        nargs=1
    )
    optoptions.add_argument(
        "-fusion_gamma",
        "--fusion_gamma",
        dest="fusion_gamma",
        type=float,
        default=0.5,
        help="Value of gamma for fusion debiasing (default=0.5).",
        nargs=1
    )
    optoptions.add_argument(
        "-groups",
        "--groups",
        dest="groups",
        action="store_true",
        help="Whether to group contiguous spikes (default=False).",
        default=False,
    )
    optoptions.add_argument(
        "-group_dist",
        "--group_dist",
        dest="group_dist",
        type=int,
        default=3,
        help="Maximum distance between two consecutive groups of spikes in TRs (default=3).",
        nargs=1
    )
    return parser
