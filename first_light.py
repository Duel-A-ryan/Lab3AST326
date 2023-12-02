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
from astropy.wcs.utils import skycoord_to_pixel

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=astropy.wcs.FITSFixedWarning)

from datetime import datetime

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

"""GRID SEARCH"""
# Removes nan values that may have gotten through and makes sure the arrays match these changes
time = time[~np.isnan(mag)]
mag_err = mag_err[~np.isnan(mag)]
mag = mag[~np.isnan(mag)]

s2n = mag/mag_err

time = time[s2n >= 3]
mag = mag[s2n >= 3]
peak_mag = np.min(mag[(22 > mag) & (mag > 17)])

norm_intensities = 10**(-(mag-peak_mag)/2.5)

time = time[norm_intensities <= 0.4]
norm_intensities = norm_intensities[norm_intensities <= 0.4]

norm_intensities = norm_intensities[time < 1.5e6]
time = time[time < 1.5e6]
time = time/86400

plt.figure(figsize=(15, 5))
plt.scatter(time, norm_intensities)
plt.show()

plt.figure(figsize=(15, 5))
popt, pcov = sc.curve_fit(pf.light_curve, time, norm_intensities)
plt.title(f"C={popt[0]:.4} and t_1={popt[1]:.4}")
plt.plot(time, pf.light_curve(time, popt[0], popt[1]))
plt.scatter(time, norm_intensities, label="Magnitude")
plt.legend()
plt.savefig("Plots/Curve_fit")
plt.show()
