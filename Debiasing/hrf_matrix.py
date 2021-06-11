import numpy as np
import scipy.io
import scipy.stats
import subprocess


def hrf_linear(RT, p):
    # Returns a hemodynamic response function
    # FORMAT hrf <- myHRF(RT,[p])

    # RT - scan repeat time
    # P - input parameters of the response function (two gamma functions)
    #
    #                                                     defaults
    #                                                    (seconds)
    #   p[1] - delay of response (relative to onset)         6
    #   p[2] - delay of undershoot (relative to onset)      16
    #   p[3] - dispersion of response                        1
    #   p[4] - dispersion of undershoot                      1
    #   p[5] - ratio of response to undershoot               6
    #   p[6] - onset (seconds)                               0
    #   p[7] - length of kernel (seconds)                   32
    #
    # The function returns the hrf  - hemodynamic response function
    # __________________________________________________________________________
    # Based on the spm_hrf function in SPM8
    # Written in R by Cesar Caballero Gaudes

    # global parameter
    # --------------------------------------------------------------------------
    fMRI_T = 16

    # modelled hemodynamic response function - {mixture of Gammas}
    # --------------------------------------------------------------------------
    dt = RT / fMRI_T
    u = np.arange(0, p[6] / dt + 1, 1) - p[5] / dt
    a1 = p[0] / p[2]
    b1 = 1 / p[3]
    a2 = p[1] / p[3]
    b2 = 1 / p[3]

    hrf = (
        scipy.stats.gamma.pdf(u * dt, a1, scale=b1)
        - scipy.stats.gamma.pdf(u * dt, a2, scale=b2) / p[4]
    ) / dt
    time_axis = np.arange(0, int(p[6] / RT + 1), 1) * fMRI_T
    hrf = hrf[time_axis]
    min_hrf = 1e-9 * min(hrf[hrf > 10 * np.finfo(float).eps])

    if min_hrf < 10 * np.finfo(float).eps:
        min_hrf = 10 * np.finfo(float).eps

    hrf[hrf == 0] = min_hrf

    return hrf


def hrf_afni(tr, lop_hrf):
    dur_hrf = 8
    last_hrf_sample = 1
    #  Increases duration until last HRF sample is zero
    while last_hrf_sample != 0:
        dur_hrf = 2 * dur_hrf
        npoints_hrf = np.around(dur_hrf, int(tr))
        hrf_command = (
            "3dDeconvolve -x1D_stop -nodata %d %f -polort -1 -num_stimts 1 -stim_times 1 '1D:0' '%s' -quiet -x1D stdout: | 1deval -a stdin: -expr 'a'"
            % (dur_hrf, tr, lop_hrf)
        )
        hrf_tr_str = subprocess.check_output(
            hrf_command, shell=True, universal_newlines=True
        ).splitlines()
        hrf_tr = np.array([float(i) for i in hrf_tr_str])
        last_hrf_sample = hrf_tr[len(hrf_tr) - 1]
        if last_hrf_sample != 0:
            print(
                "Duration of HRF was not sufficient for specified model. Doubling duration and computing again."
            )

    #  Removes tail of zero samples
    while last_hrf_sample == 0:
        hrf_tr = hrf_tr[0 : len(hrf_tr) - 1]
        last_hrf_sample = hrf_tr[len(hrf_tr) - 1]

    return hrf_tr


class HRFMatrix:
    def __init__(
        self,
        TR=2,
        TE=None,
        nscans=200,
        r2only=1,
        is_afni=True,
        lop_hrf="SPMG1",
        path=None,
        has_integrator=False,
        wfusion=False,
        lambda_fusion=3,
        gamma_weights=0.5
    ):
        self.TR = TR
        self.TE = TE
        self.nscans = nscans
        self.r2only = r2only
        self.lop_hrf = lop_hrf
        self.is_afni = is_afni
        self.hrf_path = path
        self.has_integrator = has_integrator
        self.wfusion = wfusion
        self.lambda_fusion = lambda_fusion
        self.gamma_weights = gamma_weights

    def generate_hrf(self):

        if self.hrf_path is None and self.is_afni:
            hrf_SPM = hrf_afni(self.TR, self.lop_hrf)
        elif self.hrf_path is not None:
            hrf_SPM = np.loadtxt(self.hrf_path)
        else:
            p = [6, 16, 1, 1, 6, 0, 32]
            hrf_SPM = hrf_linear(self.TR, p)

        self.L_hrf = len(hrf_SPM)  # Length
        max_hrf = max(abs(hrf_SPM))  # Max value
        filler = np.zeros(self.nscans - hrf_SPM.shape[0], dtype=np.int)
        hrf_SPM = np.append(hrf_SPM, filler)  # Fill up array with zeros until nscans

        temp = hrf_SPM

        for i in range(self.nscans - 1):
            foo = np.append(np.zeros(i + 1), hrf_SPM[0 : (len(hrf_SPM) - i - 1)])
            temp = np.column_stack((temp, foo))

        if self.TE is not None and len(self.TE) > 1:
            tempTE = -self.TE[0] * temp
            for teidx in range(len(self.TE) - 1):
                tempTE = np.vstack((tempTE, -self.TE[teidx + 1] * temp))
        else:
            tempTE = temp

        if self.r2only:
            self.hrf = tempTE

        self.hrf_norm = self.hrf / max_hrf

        if self.has_integrator:
            if self.TE is not None and len(self.TE) > 1:
                for teidx in range(len(self.TE)):
                    temp = self.hrf[
                        teidx * self.nscans : (teidx + 1) * self.nscans - 1, :
                    ].copy()
                    self.hrf[
                        teidx * self.nscans : (teidx + 1) * self.nscans - 1, :
                    ] = np.matmul(temp, np.tril(np.ones(self.nscans)))
                    temp = self.hrf_norm[
                        teidx * self.nscans : (teidx + 1) * self.nscans - 1, :
                    ].copy()
                    self.hrf_norm[
                        teidx * self.nscans : (teidx + 1) * self.nscans - 1, :
                    ] = np.matmul(temp, np.tril(np.ones(self.nscans)))
            else:
                self.hrf = np.matmul(self.hrf, np.tril(np.ones(self.nscans)))
                self.hrf_norm = np.matmul(
                    self.hrf_norm, np.tril(np.ones(self.nscans))
                )

        if self.wfusion:
            hrf_cov = np.dot(self.hrf_norm.T, self.hrf_norm)
            weights = (np.abs(hrf_cov) ** self.gamma_weights) / (1 - np.abs(hrf_cov))
            diag_W = np.sum(weights, axis=1) - np.diag(weights)
            Wfusion = -np.sign(hrf_cov) ** weights
            for i in range (self.nscans):
                Wfusion[i,i] = diag_W[i]

            Wfusion = Wfusion / (self.nscans)
            # cholesky decomposition of matrix W (through eigenvalue decomposition)
            [Vfusion,Dfusion] = np.linalg.eig(Wfusion)
            Dfusion[Dfusion < 0] = 0  # force positive eigenvalues due to numerical precision.
            Qfusion = (Dfusion ** 0.5) * Vfusion.T
            X_fusion = np.sqrt(self.lambda_fusion) * Qfusion
            self.hrf_norm_fusion = np.vstack(self.hrf_norm, X_fusion)

        return self
