import scipy.optimize
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
import scipy.optimize as sc

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=astropy.wcs.FITSFixedWarning)

from datetime import datetime

OBJECTS = {
    "star_1": [SkyCoord(ra='0:56:49.70', dec='-37:01:38.31', frame="icrs", unit=(u.hourangle, u.deg)), 16.32, 0.07],
    "star_2": [SkyCoord(ra='0:56:46.43', dec='-37:02:29.50', frame="icrs", unit=(u.hourangle, u.deg)), 17.16, 0.08],
    "star_3": [SkyCoord(ra='0:56:58.27', dec='-36:58:16.60', frame="icrs", unit=(u.hourangle, u.deg)), 15.62, 0.02],
    "SN": [SkyCoord(ra="0:57:03.1827", dec='-37:02:23.654', frame="icrs", unit=(u.hourangle, u.deg)), 0, 0]}

"""=====MAIN====="""
# Read text file of all the file names
with open("Data/AST325-326-SN-filelist.txt") as f:
    filenames = ["Data/Fits/" + line for line in f.readlines()]

# Organizes the files in order of date collected
file_times = []
for f in filenames:
    header = fits.open(f[:-1])[0].header
    dt = datetime.fromisoformat(header["DATE-OBS"])
    file_times.append(dt)

# sort files by the same way
file_times, filenames = zip(*sorted(zip(file_times, filenames)))

# Turns the lists into numpy arrays
filenames = np.array(filenames)
file_times = np.array(file_times)

file_array = filenames[100:110]

fluxes = np.zeros(len(file_array))
flux_errs = np.zeros(len(file_array))

for ii, file in enumerate(file_array):

    data, header = pf.fit_open(file[:-1])

    w = WCS(header)

    # Gets sub image data, aperture and annulus of supernova
    sub_im, aperture, annulus, _ = pf.fluxes(data, "SN", w)

    fluxes[ii], flux_errs[ii] = pf.get_flux(sub_im, aperture, annulus)

print(fluxes)
print(flux_errs)