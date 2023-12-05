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

flux1, flux1_err = np.loadtxt("Data/cleaned data/fluxes/fluxes_0", delimiter=',', unpack=True)
flux2, flux2_err = np.loadtxt("Data/cleaned data/fluxes/fluxes_1", delimiter=',', unpack=True)
flux3, flux3_err = np.loadtxt("Data/cleaned data/fluxes/fluxes_2", delimiter=',', unpack=True)

m1_ref, m1_err_ref = OBJECTS['star_1'][1:]
m2_ref, m2_err_ref = OBJECTS['star_2'][1:]
m3_ref, m3_err_ref = OBJECTS['star_3'][1:]

flux12, flux12_err = flux1/flux2, -1 # TODO: Function for uncertainty
flux13, flux13_err = flux1/flux3, -1

flux21, flux21_err = flux2/flux1, -1
flux23, flux23_err = flux2/flux3, -1

flux31, flux31_err = flux3/flux1, -1
flux32, flux32_err = flux3/flux2, -1




