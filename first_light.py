import scipy.optimize as sc
from astropy.wcs import WCS
import astropy
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import helpful_functions as pf
import warnings
from astropy import units as u
from astropy.coordinates import SkyCoord
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=astropy.wcs.FITSFixedWarning)


# Shape --> {key: [Coordinates in sky (degrees), magnitude, uncertainty in magnitude]}
OBJECTS = {
    "star_1": [SkyCoord(ra='0:56:49.70', dec='-37:01:38.31', frame="icrs", unit=(u.hourangle, u.deg)), 16.32, 0.07],
    "star_2": [SkyCoord(ra='0:56:46.43', dec='-37:02:29.50', frame="icrs", unit=(u.hourangle, u.deg)), 17.16, 0.08],
    "star_3": [SkyCoord(ra='0:56:58.27', dec='-36:58:16.60', frame="icrs", unit=(u.hourangle, u.deg)), 15.62, 0.02],
    "SN": [SkyCoord(ra="0:57:03.1827", dec='-37:02:23.654', frame="icrs", unit=(u.hourangle, u.deg)), 0, 0]}

h = fits.open("Data/Fits/AST325-326-SN-20150924.5410.fits")[0].header
ref_time = datetime.fromisoformat(h["DATE-OBS"])

"""==================== MAIN ======================="""
# Read text file of all the file names
with open("Data/cleaned_filelist.txt") as f:
    filenames = [line for line in f.readlines()]

filenames = np.array(filenames)

# Sets arrays to carry flux and magnitude data for the reference stars and supernova
time = np.loadtxt("Data/cleaned data/times")
mag = np.loadtxt("Data/cleaned data/mags")
mag_err = np.loadtxt("Data/cleaned data/mag_uncs")

# Removes nan values that may have gotten through and makes sure the arrays match these changes
time = time[~np.isnan(mag)]
mag_err = mag_err[~np.isnan(mag)]
mag = mag[~np.isnan(mag)]

# Checks and removes magnitudes with signal-to-noise ratio less than 3
s2n = mag/mag_err
time = time[s2n >= 3]
mag_err = mag_err[s2n >= 3]
mag = mag[s2n >= 3]
peak_mag = np.min(mag[(22 > mag) & (mag > 17)])
peak_mag_err = np.min(mag_err[(22 > mag) & (mag > 17)])

# Converts the magnitudes to norm intensities
norm_intensities, norm_intensities_err = pf.mag_to_flux(mag, mag_err, peak_mag, peak_mag_err)

time = time[norm_intensities <= 0.4]
norm_intensities_err = norm_intensities_err[norm_intensities <= 0.4]
norm_intensities = norm_intensities[norm_intensities <= 0.4]


norm_intensities = norm_intensities[time < 1.5e6]
norm_intensities_err = norm_intensities_err[time < 1.5e6]
time = time[time < 1.5e6]
time = time/86400


binned_times, binned_flux, binned_flux_err = pf.binning([-4, 8], 8, time, norm_intensities, norm_intensities_err)

popt_og, pcov_og = sc.curve_fit(pf.light_curve, time, norm_intensities)
popt_bin, pcov_bin = sc.curve_fit(pf.light_curve, binned_times, binned_flux)


plt.figure(figsize=(10, 10))
plt.suptitle("Relative Intensity of Supernova")

plt.subplot(2, 1, 1)
plt.errorbar(time, norm_intensities, norm_intensities_err, fmt='o', label="Magnitude")
plt.plot(time, pf.light_curve(time, *popt_og))
plt.ylabel("Relative Intensity")
plt.legend()

plt.subplot(2, 1, 2)
plt.errorbar(binned_times, binned_flux, binned_flux_err, label="Binned", fmt="o")
plt.plot(binned_times, pf.light_curve(binned_times, *popt_bin))
plt.ylabel("Relative Intensity (Binned)")
plt.xlabel("Days")
plt.legend()

plt.savefig("Plots/Curve_fit")
plt.show()
