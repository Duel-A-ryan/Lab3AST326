from astropy.wcs import WCS
import astropy
import numpy as np
import matplotlib.pyplot as plt
import helpful_functions as pf
import warnings

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=astropy.wcs.FITSFixedWarning)

"""=====MAIN====="""

times, mags, mag_err = np.loadtxt("Data/cleaned data/testing", delimiter=',', unpack=True)

times = times/86400

_filter = ~np.isnan(mags)
_filter = _filter & (mags > 17) & (mags < 23)

times, mags, mag_err = times[_filter], mags[_filter], mag_err[_filter]

p_og, res_og = np.polyfit(times, mags, 3, cov=True)
p = np.poly1d(p_og)

_filter = np.abs(mags - p(times)) < 3.6*mag_err
times, mags, mag_err = times[_filter], mags[_filter], mag_err[_filter]

p_new, res_new = np.polyfit(times, mags, 3, cov=True)
p = np.poly1d(p_new)
chi_og = pf.reduced_chi_squared(mags, p(times), mag_err, len(mags) - 1)

plt.figure(figsize=(15, 5))
plt.title("B-band Magnitude Light Curve of AST325-326-SN")
plt.ylabel("B-band Magnitude")
plt.xlabel("Days (Reference Day September 24th)")

plt.errorbar(times, mags, mag_err, fmt="o")
plt.plot(times, p(times), label=f"n={3}")
plt.ylim(24, 18)
plt.legend()
plt.savefig(f"Plots/light_curve_fit")

res = mags - p(times)
plt.figure()
plt.scatter(times, res, label=r"Residuals ($$")
plt.hlines(0, xmin=-2, xmax=37, colors='black')

plt.title("Residual Plot of Light Curve Fitting")
plt.savefig("Plots/Residuals/light_curve_res")

print(f"Reduced $\chi^2$ = {chi_og:.4}")



