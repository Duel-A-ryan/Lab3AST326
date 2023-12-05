from astropy.wcs import WCS
import astropy
import numpy as np
import matplotlib.pyplot as plt
import helpful_functions as pf
import warnings

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=astropy.wcs.FITSFixedWarning)

"""=====MAIN====="""

times = np.loadtxt("Data/cleaned data/times")
mags = np.loadtxt("Data/cleaned data/mags")
mag_err = np.loadtxt("Data/cleaned data/mag_uncs")

times = times[~np.isnan(mags)]
mag_err = mag_err[~np.isnan(mags)]
mags = mags[~np.isnan(mags)]

s2n = mags / mag_err

times = times[s2n >= 3]
mags = mags[s2n >= 3]
mag_err = mag_err[s2n >= 3]

times = times[17 < mags]
times = times[mags < 23]
mag_err = mag_err[17 < mags]
mag_err = mag_err[mags < 23]
mags = mags[17 < mags]
mags = mags[mags < 23]

binned_times, binned_mags, binned_mag_err = pf.binning([-5e4, 3e6], 101, times, mags, mag_err)

for n in range(2, 5):

    p_og, res_og, _, _, _ = np.polyfit(times, mags, n, full=True)
    pred_og = np.polyval(p_og, times)
    chi_og = pf.reduced_chi_squared(mags, pred_og, mag_err, len(mags) - 1)
    print(f"The original chi value is {chi_og:.4}")

    plt.figure(figsize=(15, 5))
    plt.title(f"Reduced $\chi^2$ = {chi_og:.4}")
    plt.subplot(2, 1, 1)
    plt.errorbar(times, mags, mag_err, fmt="o")
    plt.plot(times, pred_og, label=f"n={n}")
    plt.ylim(24, 18)
    plt.legend()
    #plt.savefig(f"Plots/light_fit/og_{n}")

    p_new, cov_new = np.polyfit(binned_times, binned_mags, n, w=1/binned_mag_err**2, cov=True)
    p = np.poly1d(p_new)

    filter = np.abs(binned_mags - p(binned_times)) < 3 * binned_mag_err

    binned_times = binned_times[filter]
    binned_mags = binned_mags[filter]
    binned_mag_err = binned_mag_err[filter]

    p_new, cov_new = np.polyfit(binned_times, binned_mags, n, w=1 / binned_mag_err ** 2, cov=True)
    p_n = np.poly1d(p_new)

    chi_new = pf.reduced_chi_squared(binned_mags, p_n(binned_times), binned_mag_err, len(binned_mags) - 1)
    print(f"The new chi value is {chi_new:.4}\n")

    plt.title(f"Reduced $\chi^2$ = {chi_new:.4}")
    plt.subplot(2, 1, 2)
    plt.plot(binned_times, p_n(binned_times), label=f"n={n}")
    plt.errorbar(binned_times, binned_mags, binned_mag_err, fmt="o")
    plt.ylim(24, 18)
    plt.legend()
    plt.savefig(f"Plots/light_fit/bin_{n}")


