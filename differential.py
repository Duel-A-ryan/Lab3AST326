"""===IMPORTS==="""
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import astropy
import numpy as np
from astropy.io import fits
import helpful_functions as pf
import warnings
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=astropy.wcs.FITSFixedWarning)

# Shape --> {key: [Coordinates in sky (degrees), magnitude, uncertainty in magnitude]}
OBJECTS = {
    "star_1": [SkyCoord(ra='0:56:49.70', dec='-37:01:38.31', frame="icrs", unit=(u.hourangle, u.deg)), 16.32, 0.07],
    "star_2": [SkyCoord(ra='0:56:46.43', dec='-37:02:29.50', frame="icrs", unit=(u.hourangle, u.deg)), 17.16, 0.08],
    "star_3": [SkyCoord(ra='0:56:58.27', dec='-36:58:16.60', frame="icrs", unit=(u.hourangle, u.deg)), 15.62, 0.02],
    "SN": [SkyCoord(ra="0:57:03.1827", dec='-37:02:23.654', frame="icrs", unit=(u.hourangle, u.deg)), 0, 0]}

"""==================== MAIN ======================="""

# Takes the chosen file and sets its data as a reference time to be used for determining time since start of supernova
h = fits.open("Data/Fits/AST325-326-SN-20150924.5410.fits")[0].header
ref_time = datetime.fromisoformat(h["DATE-OBS"])

# Read text file of all the file names
with open("Data/cleaned_filelist.txt") as f:
    filenames = [line for line in f.readlines()]

# Turns list into numpy array
filenames = np.array(filenames)

# Sets arrays to carry flux and magnitude data for the reference stars and supernova
# Rows 0,1,2 hold data for reference stars while row 4 is saved for flux of supernova
file_array = filenames
fluxes = np.zeros((4, len(file_array)))
flux_errs = np.zeros((4, len(file_array)))
mag = np.zeros(len(file_array))
mag_err = np.zeros(len(file_array))
time = []

# Begins a loop through each file
for ii, f in enumerate(file_array):

    # Opens file and saves data and header
    file = fits.open(f[:-1])[0]
    data, header = file.data, file.header
    w = WCS(header)

    # Takes the difference in date of the file and the reference time to give a time since supernova start in seconds
    image_date = datetime.fromisoformat(header["DATE-OBS"])
    time.append((image_date - ref_time).total_seconds())

    # Finding pixel center for detected supernova based on given RA,DEC value
    sn_pos_pxl = skycoord_to_pixel(OBJECTS["SN"][0], w)
    sn_x, sn_y = sn_pos_pxl[0].astype(int), sn_pos_pxl[1].astype(int)

    # Gathers sub image around SN and optimal aperture and annulus for calculations
    sub_im_sn, aperture_sn, annulus_sn, _ = pf.fluxes(data, "SN", w)
    fluxes[3, ii], flux_errs[3, ii] = pf.get_flux(sub_im_sn, aperture_sn, annulus_sn)

    # Makes zero arrays to hold magnitude and uncertainty from each reference star calculation to then be averaged later
    mag_sn = np.zeros(3)
    mag_err_sn = np.zeros(3)

    # Runs through the 3 reference stars
    for jj, key in enumerate(["star_1", "star_2", "star_3"]):

        # Takes the data to find a sub image around the star and return an optimal aperture and annulus to
        # calculate flux from
        sub_im, aperture, annulus, bad = pf.fluxes(data, key, w)

        # Calculates flux
        # If bad comes back as true, flux and magnitude are left as 0 to make removal easier
        if bad is False:
            fluxes[jj, ii], flux_errs[jj, ii] = pf.get_flux(sub_im, aperture, annulus)
            mag_sn[jj], mag_err_sn[jj] = pf.magnitude(fluxes[3, ii], flux_errs[3, ii], fluxes[jj, ii],
                                                      flux_errs[jj, ii], OBJECTS[key][1],
                                                      OBJECTS[key][2])

    # Updates magnitude and magnitude error
    mag[ii], mag_err[ii] = pf.mag_mean(mag_sn, mag_err_sn)

# Plots apparent magnitude light curve
plt.figure(figsize=(15, 5))
plt.errorbar(time, mag, mag_err, label="Magnitude", fmt=".", ecolor="red")

plt.ylim(25, 18)
plt.ylabel('Magnitude', size=16)
plt.xlabel('Time', size=16)
plt.title("Supernova Light Curve", size=18)
plt.grid(which='both', alpha=0.5)
plt.show()

"""
with open("Data/cleaned data/times", "w") as f:
    for line in time:
        f.write(str(line))
        f.write('\n')

with open("Data/cleaned data/mags", "w") as f:
    for line in mag:
        f.write(str(line))
        f.write('\n')

with open("Data/cleaned data/mag_uncs", "w") as f:
    for line in mag_err:
        f.write(str(line))
        f.write('\n')

with open("Data/cleaned data/testing", 'w') as f:
    for i in range(0, len(mag_err)):
        f.write(f"{time[i]}, {mag[i]}, {mag_err[i]}")
        f.write('\n')
"""

time = np.array(time)

with open(f"Data/cleaned data/fluxes_sn", 'w') as f:
    filter = (fluxes[3] / flux_errs[3]) > 3
    print(filter)
    time_filter = time[filter]
    flux = fluxes[3][filter]
    flux_err = flux_errs[3][filter]
    for j in range(0, len(flux)):
        f.write(f"{time_filter[j]}, {flux[j]}, {flux_err[j]}")
        f.write('\n')
