"""===IMPORTS==="""
from datetime import datetime
from astropy.wcs import WCS
import astropy
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import helpful_functions as pf
import warnings
from astropy import units as u
from astropy.coordinates import SkyCoord

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=astropy.wcs.FITSFixedWarning)

OBJECTS = {
    "star_1": [SkyCoord(ra='0:56:49.70', dec='-37:01:38.31', frame="icrs", unit=(u.hourangle, u.deg)), 16.32, 0.07],
    "star_2": [SkyCoord(ra='0:56:46.43', dec='-37:02:29.50', frame="icrs", unit=(u.hourangle, u.deg)), 17.16, 0.08],
    "star_3": [SkyCoord(ra='0:56:58.27', dec='-36:58:16.60', frame="icrs", unit=(u.hourangle, u.deg)), 15.62, 0.02],
    "SN": [SkyCoord(ra="0:57:03.1827", dec='-37:02:23.654', frame="icrs", unit=(u.hourangle, u.deg)), 0, 0]}

fluxes, flux_errs = np.zeros(4), np.zeros(4)
mag_sn, mag_err_sn = np.zeros(3), np.zeros(3)

f = "Data/Fits/AST325-326-SN-20151027.2806.fits"
file = fits.open(f)[0]
data, header = file.data, file.header

# Saves the (w value?) to be used in determining coordinates
w = WCS(header)

#sub_im_sn, aperture_sn, annulus_sn, bad_sn = pf.fluxes(data, "SN", w)
#fluxes[3], flux_errs[3] = pf.get_flux(sub_im_sn, aperture_sn, annulus_sn)

# Loops through each reference star
for jj, key in enumerate(["star_1", "star_2", "star_3"]):

    # Takes the data to find a sub image around the star and return an optimal aperture and annulus to
    # calculate flux from
    sub_im, aperture, annulus, bad = pf.fluxes(data, key, w)


    # Calculates flux
    # If the bad value comes back true, flux is not calculated and kept as 0 which makes removing easier later
    if bad is False:
        fluxes[jj], flux_errs[jj] = pf.get_flux(sub_im, aperture, annulus)
        mag_sn[jj], mag_err_sn[jj] = pf.magnitude(fluxes[3], flux_errs[3], fluxes[jj],
                                                  flux_errs[jj], OBJECTS[key][1],
                                                  OBJECTS[key][2])


mag, mag_err = pf.mag_mean(mag_sn, mag_err_sn)

print(f"The calculated fluxes are:\n"
      f"{fluxes[0]} \u00B1 {flux_errs[0]} which is {flux_errs[0]/fluxes[0] * 100}\n"
      f"{fluxes[1]} \u00B1 {flux_errs[1]} which is {flux_errs[1]/fluxes[1] * 100}\n"
      f"{fluxes[2]} \u00B1 {flux_errs[2]} which is {flux_errs[1]/fluxes[2] * 100}\n")

print(f"The magnitude of the Supernova is: {mag} \u00B1 {mag_err}")
