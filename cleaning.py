"""===IMPORTS==="""

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

from datetime import datetime

# Shape --> {key: [Coordinates in sky (degrees), magnitude, uncertainty in magnitude]}
REF = {
    "star_1": [SkyCoord(ra='0:56:49.70', dec='-37:01:38.31', frame="icrs", unit=(u.hourangle, u.deg)), 16.32, 0.07],
    "star_2": [SkyCoord(ra='0:56:46.43', dec='-37:02:29.50', frame="icrs", unit=(u.hourangle, u.deg)), 17.16, 0.08],
    "star_3": [SkyCoord(ra='0:56:58.27', dec='-36:58:16.60', frame="icrs", unit=(u.hourangle, u.deg)), 15.62, 0.02]}

"""==================== MAIN ======================="""
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

# Chooses which files will be iterated through
# Creates empty 2D arrays to hold calculated flux values for each reference star in each image
file_array = filenames
fluxes = np.zeros((3, len(file_array)))
flux_errs = np.zeros((3, len(file_array)))

# Begins a loop through each file
for ii, f in enumerate(file_array):

    # Opens file and saves data and header
    file = fits.open(f[:-1])[0]
    data, header = file.data, file.header

    # Saves the (w value?) to be used in determining coordinates
    w = WCS(header)

    # Loops through each reference star
    for jj, key in enumerate(["star_1", "star_2", "star_3"]):

        # Takes the data to find a sub image around the star and return an optimal aperture and annulus to
        # calculate flux from
        sub_im, aperture, annulus, bad = pf.fluxes(data, key, w)

        # Calculates flux
        # If the bad value comes back true, flux is not calculated and kept as 0 which makes removing easier later
        if bad is False:
            fluxes[jj, ii], flux_errs[jj, ii] = pf.get_flux(sub_im, aperture, annulus)

# Plot the fluxes gathered for all the data
plt.figure(figsize=(15, 5))
plt.plot(file_array, fluxes[0], label="Star 1", color="blue")
plt.scatter(file_array, fluxes[0], color="blue")
plt.plot(file_array, fluxes[1], label="Star 2", color="orange")
plt.scatter(file_array, fluxes[1], color="orange")
plt.plot(file_array, fluxes[2], label="Star 3", color="green")
plt.scatter(file_array, fluxes[2], color="green")
plt.yscale("log")
plt.ylim(1e3, 5e5)
plt.legend()
plt.savefig("Plots/Logarithmic_Flux_Plot")
plt.show()

# Saves the good files into new text file to be used for calculations in differential.py
with open("Data/cleaned_filelist.txt", "w") as f:
    for line in file_array[fluxes[1] > 1e4]:
        f.write(line[:-1])
        f.write('\n')
