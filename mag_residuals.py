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

time, flux1, flux1_err, flux2, flux2_err, flux3, flux3_err = np.loadtxt("Data/cleaned data/fluxes", delimiter=',', unpack=True)

#flux1, flux1_err = np.loadtxt("Data/cleaned data/fluxes/fluxes_0", delimiter=',', unpack=True)
#flux2, flux2_err = np.loadtxt("Data/cleaned data/fluxes/fluxes_1", delimiter=',', unpack=True)
#flux3, flux3_err = np.loadtxt("Data/cleaned data/fluxes/fluxes_2", delimiter=',', unpack=True)

m1_ref, m1_ref_err = OBJECTS['star_1'][1:]
m2_ref, m2_ref_err = OBJECTS['star_2'][1:]
m3_ref, m3_ref_err = OBJECTS['star_3'][1:]

m12, m12_err = pf.magnitude(flux1, flux1_err, flux2, flux2_err, m2_ref, m2_ref_err)
m13, m13_err = pf.magnitude(flux1, flux1_err, flux3, flux3_err, m3_ref, m3_ref_err)

m21, m21_err = pf.magnitude(flux2, flux2_err, flux1, flux1_err, m1_ref, m1_ref_err)
m23, m23_err = pf.magnitude(flux2, flux2_err, flux3, flux3_err, m3_ref, m3_ref_err)

m31, m31_err = pf.magnitude(flux3, flux3_err, flux1, flux1_err, m1_ref, m1_ref_err)
m32, m32_err = pf.magnitude(flux3, flux3_err, flux2, flux2_err, m2_ref, m2_ref_err)

m1, m2, m3 = [], [], []
m1_err, m2_err, m3_err = [], [], []

for j in range(0, len(m12)):
    mag1, mag1_err = pf.weighted_mean([m12[j], m13[j]], [m12_err[j], m13_err[j]])
    mag2, mag2_err = pf.weighted_mean([m21[j], m23[j]], [m21_err[j], m23_err[j]])
    mag3, mag3_err = pf.weighted_mean([m31[j], m32[j]], [m31_err[j], m32_err[j]])

    m1.append(mag1)
    m1_err.append(mag1_err)

    m2.append(mag2)
    m2_err.append(mag2_err)

    m3.append(mag3)
    m3_err.append(mag3_err)

m1, m2, m3 = np.array(m1), np.array(m2), np.array(m3)
m1_err, m2_err, m3_err = np.array(m1_err), np.array(m2_err), np.array(m3_err)

m1_res, m1_res_err = pf.diff(m1, m1_err, m1_ref, m1_ref_err)
m2_res, m2_res_err = pf.diff(m2, m2_err, m2_ref, m2_ref_err)
m3_res, m3_res_err = pf.diff(m3, m3_err, m3_ref, m3_ref_err)

plt.errorbar(time, m1, m1_err, label="Star 1 Data", fmt='o', capsize=2)
plt.errorbar(time, m2, m2_err, label="Star 2 Data", fmt='o', capsize=2)
plt.errorbar(time, m3, m3_err, label="Star 3 Data", fmt='o', capsize=2)

plt.hlines([OBJECTS["star_1"][1], OBJECTS["star_2"][1], OBJECTS["star_3"][1]], xmin=-0.3e6, xmax=3.3e6, colors="black")
plt.legend()
plt.savefig("Plots/magnitude_check")

plt.figure()

plt.errorbar(time, m1_res, m1_res_err, label="Star 1 Data", fmt='.', capsize=2)
plt.errorbar(time, m2_res, m2_res_err, label="Star 2 Data", fmt='.', capsize=2)
plt.errorbar(time, m3_res, m3_res_err, label="Star 3 Data", fmt='.', capsize=2)
plt.hlines(0, xmin=-0.3e6, xmax=3.3e6, colors='black')

plt.legend()
plt.savefig("Plots/magnitude_check_residuals")
plt.show()

