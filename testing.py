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
_filter = ~np.isnan(mag)
_filter = _filter & (mag > 17) & (mag < 23)

times, mags, mag_err = time[_filter], mag[_filter], mag_err[_filter]

flux, flux_err = pf.mag_to_flux(mags, mag_err)
# Checks and removes magnitudes with signal-to-noise ratio less than 3

peak_flux = np.max(flux)
index = np.argmax(flux)
peak_flux_err = flux_err[index]

norm_flux, norm_flux_err = pf.normalize_flux(flux, flux_err, peak_flux, peak_flux_err)

peak_mag = np.min(mags)
index = np.argmin(mags)
peak_mag_err = mag_err[index]

_filter = norm_flux <= 0.4

time = times[_filter]
norm_flux_err = norm_flux_err[_filter]
norm_flux = norm_flux[_filter]

_filter = time < 1.5e6

norm_flux = norm_flux[_filter]
norm_flux_err = norm_flux_err[_filter]
time = time[_filter]
time = time/86400

chi_vals = []
step_ind = []

steps = np.arange(6, 100, 1)

for step in steps:
    binned_times, binned_flux, binned_flux_err = pf.binning([-4, 8], step, time, norm_flux, norm_flux_err)

    popt_bin, pcov_bin = sc.curve_fit(pf.light_curve, binned_times, binned_flux)

    chi = pf.reduced_chi_squared(binned_flux, pf.light_curve(binned_times, *popt_bin), binned_flux_err, len(binned_flux) - 1)
    chi_vals.append(chi)
    step_ind.append(step)

print(min(chi_vals))
step_index = chi_vals.index(min(chi_vals))
print(step_index)
print(step_ind[step_index])
