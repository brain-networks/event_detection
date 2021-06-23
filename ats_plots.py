from os.path import join as opj

import matplotlib.pyplot as plt
import numpy as np

prj_dir = "/export/home/eurunuela/public/PARK_VFERRER/PFM_data/sub-002ParkMabCm_200"
temp_dir = "/export/home/eurunuela/public/PARK_VFERRER/PFM_data/temp_sub-002ParkMabCm_200"
n_rand = 100
fig_size = (16, 9)
ats_mode = "power"

ats_95 = f"pb06.sub-002ParkMabCm.denoised_no_censor_ATS_{ats_mode}_95.1D"
ats_99 = f"pb06.sub-002ParkMabCm.denoised_no_censor_ATS_{ats_mode}_99.1D"

plt.figure(figsize=fig_size)

for surr_file in range(n_rand):
    surr_ats = np.loadtxt(opj(temp_dir, f"surrogate_{surr_file}_ATS_{ats_mode}_95.1D"))
    plt.plot(surr_ats, color="lightskyblue", linewidth=0.5)

for surr_file in range(n_rand):
    surr_ats = np.loadtxt(opj(temp_dir, f"surrogate_{surr_file}_ATS_{ats_mode}_99.1D"))
    plt.plot(surr_ats, color="lightcoral", linewidth=0.5)

ats_95_ts = np.loadtxt(opj(prj_dir, ats_95))
plt.plot(ats_95_ts, color="royalblue", linewidth=3, label="ATS with 95th threshold")

ats_99_ts = np.loadtxt(opj(prj_dir, ats_99))
plt.plot(ats_99_ts, color="red", linewidth=3, label="ATS with 99th threshold")

plt.tight_layout()
plt.legend()

plt.savefig(opj(prj_dir, "ats_comparison.png"), dpi=300, bbox_inches="tight")
