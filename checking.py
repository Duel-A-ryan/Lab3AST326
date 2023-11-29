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
file_array = filenames
fluxes = np.zeros((4, len(file_array)))
flux_errs = np.zeros((4, len(file_array)))
mag = np.zeros(len(file_array))
mag_err = np.zeros(len(file_array))
time = np.zeros(len(file_array))

for ii, f in enumerate(file_array):
    # Opens file and saves data and header
    file = fits.open(f[:-1])[0]
    data, header = file.data, file.header

    image_date = datetime.fromisoformat(header["DATE-OBS"])
    time[ii] = (image_date - ref_time).total_seconds()
    w = WCS(header)

    # Finding pixel center for detected supernova
    sn_pos_pxl = skycoord_to_pixel(OBJECTS["SN"][0], w)
    sn_x, sn_y = sn_pos_pxl[0].astype(int), sn_pos_pxl[1].astype(int)

    sub_im_sn, aperture_sn, annulus_sn, _ = pf.fluxes(data, "SN", w)
    fluxes[3, ii], flux_errs[3, ii] = pf.get_flux(sub_im_sn, aperture_sn, annulus_sn)
    # Runs through the 3 reference stars

    mag_sn = np.zeros(3)
    mag_err_sn = np.zeros(3)

    for jj, key in enumerate(["star_1", "star_2", "star_3"]):
        # Finds the position the stars take in the array

        sub_im, aperture, annulus, bad = pf.fluxes(data, key, w)

        # Calculates flux
        if bad is False:
            fluxes[jj, ii], flux_errs[jj, ii] = pf.get_flux(sub_im, aperture, annulus)
            mag_sn[jj], mag_err_sn[jj] = pf.magnitude(fluxes[3, ii], flux_errs[3, ii], fluxes[jj, ii],
                                                      flux_errs[jj, ii], OBJECTS[key][1],
                                                      OBJECTS[key][2]
                                                      )
    mag[ii], mag_err[ii] = np.mean(mag_sn), np.mean(mag_err_sn)

"""GRID SEARCH"""

# Last minute removal of nan values in the way
time = time[~np.isnan(mag)]
mag = mag[~np.isnan(mag)]

peak_mag = np.min(mag[(22 > mag) & (mag > 17)])

norm_intensities = 10**(-(mag-peak_mag)/2.5)

plt.figure(figsize=(15, 5))
plt.ylim(-0.2, 1.2)
plt.scatter(time, norm_intensities)
plt.show()
"""
plt.figure(figsize=(15, 5))
popt, pcov = sc.curve_fit(pf.light_curve, time, mag)
plt.plot(time, pf.light_curve(time, popt[0], popt[1]))

plt.scatter(time, mag, label="Magnitude")
plt.ylim(30, 15)
plt.legend()
plt.savefig("Plots/Magnitude_Plots")
plt.show()
"""
