from astropy.io import fits
import astropy.wcs
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit
import helpful_functions as pf
plt.rcParams.update({'font.size': 12})

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel

from os import listdir
from os.path import isfile, join
from datetime import datetime
from matplotlib.colors import LogNorm
import warnings

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=astropy.wcs.FITSFixedWarning)

with open("Data/AST325-326-SN-filelist.txt") as f:
    img_files = ["Data/Fits/" + line[:-1] for line in f.readlines()]


def read_file(filename):
    hdu = fits.open(filename)[0]
    data = hdu.data.astype(int)
    data[data < -1e3] = ((2 ** 15 - 1) - np.abs(data[data < -1e3])) + (2 ** 15 - 1)
    return data, hdu.header

    def ra_dec_string_to_skycoord(ra_str, dec_str):
        # Convert RA and DEC strings to SkyCoord object
        skycoord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
        return skycoord


def get_centroid(image):
    x_values = np.arange(image.shape[1]) - image.shape[1] // 2
    x_centroid = np.sum(x_values * image ** 2) / np.sum(image ** 2)
    y_centroid = np.sum(x_values * image.T ** 2) / np.sum(image ** 2)
    return np.array([y_centroid, x_centroid])


def gaus(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c


def get_aperture(shape, radius):
    aperture = np.zeros(shape)
    center = shape[0] // 2, shape[1] // 2
    for i in range(shape[0]):
        for j in range(shape[1]):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if distance < radius:
                aperture[j, i] = 1
    return aperture


def get_annulus(shape, in_radius, out_radius):
    annulus = np.zeros(shape)
    center = shape[0] // 2, shape[1] // 2
    for i in range(shape[0]):
        for j in range(shape[1]):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if distance > in_radius and distance < out_radius:
                annulus[j, i] = 1
    return annulus


def get_flux(sub_im, aperture, annulus, gain):
    N_annulus = annulus[annulus > 0].size
    N_aperture = aperture[aperture > 0].size
    bg = np.sum(sub_im * annulus) / N_annulus
    flux = np.sum((sub_im - bg) * aperture)
    flux_errs = np.sqrt((flux / gain) + N_aperture * (1 + ((np.pi / 2) * (N_aperture / N_annulus)) * (np.std(bg)) ** 2))
    return flux, flux_errs


def fluxes(data, w):
    bad_file = False

    boxsize = 50

    # locate the star
    ref_pos = skycoord_to_pixel(star, w)
    ref_x, ref_y = ref_pos

    # we can then put look at a box around the star to set up our flux code above
    ref_x_int, ref_y_int = ref_x.astype(int), ref_y.astype(int)

    # generate region around star
    sub_im = data[ref_y_int - boxsize: ref_y_int + boxsize + 1,
             ref_x_int - boxsize: ref_x_int + boxsize + 1]

    # Notice that the centroid is not at a pixel interval,
    # we should be fine shifting the image as best we can
    new_centroid = get_centroid(sub_im)
    print("New Centroid", new_centroid)

    # Checks if the final image is the correct shape
    if sub_im.shape == (2 * boxsize + 1, 2 * boxsize + 1):

        # Gets a tuple with the shape of the 2D array to then get a singular row going through the center of the object
        # Gaussian curve fit is then applied to find the standard deviation from the rows intensity data
        shape = sub_im.shape[0]
        gaus_x, gaus_y = np.arange(shape), sub_im[shape // 2, :]
        popt, _ = curve_fit(gaus, gaus_x, gaus_y, p0=[max(gaus_y), shape // 2, 10, 0], maxfev=3000)

        # Plots the Gaussian alongside the data points
        """
      plt.plot(gaus_x, gaus_y, '.', label="Data Points")
      plt.plot(gaus_x, gaussian(gaus_x, *popt), label=fr"Gaussian with $\sigma = {popt[2]:.3}$", color="red")
      plt.legend()
      """

        # Calculates the optimal radius for the aperture from the standard deviation
        radius = 2.355 * popt[2]

        # Checks if the standard deviation falls in the wanted range. If not the file is labeled bad and will be
        # discarded
        if 1.5 < popt[2] < 5:
            aperture = get_aperture(sub_im.shape, radius)
            annulus = get_annulus(sub_im.shape, radius + 4, radius + 7)
        else:
            bad_file = True
            aperture = get_aperture(sub_im.shape, 10)
            annulus = get_annulus(sub_im.shape, 12, 15)

        # Plot the star sub images TODO: hopefully centered next time

        plt.figure(figsize=(7, 7))
        plt.imshow(sub_im, cmap='inferno', interpolation="None")
        plt.title(f"Image #{i + 1} Ref Star #{j + 1}", size=20)
        plt.axvline(x=sub_im.shape[0] // 2, color='red')
        plt.axhline(y=sub_im.shape[0] // 2, color='red')
        plt.colorbar()
        plt.show()
        return sub_im, aperture, annulus, bad_file
    else:
        return sub_im, 0, 0, True


file_times = []
for f in img_files:
    header = fits.open(f)[0].header
    dt = datetime.fromisoformat(header["DATE-OBS"])
    file_times.append(dt)

# sort files by the same way
file_times, img_files = zip(*sorted(zip(file_times, img_files)))

img_files = np.array(img_files)
file_times = np.array(file_times)

star1 = SkyCoord('00h56m49.70s', '-37d01m38.31s')
star2 = SkyCoord('00h56m46.43s', '-37d02m29.50s')
star3 = SkyCoord('00h56m58.27s', '-36d58m16.60s')
supernova = SkyCoord(ra="0:57:03.1827", dec='-37:02:23.654', frame="icrs", unit=(u.hourangle, u.deg))

ref_stars = [star1, star2, star3]
# print(ref_stars)

boxsize = 50

sub_im_shape = (2 * boxsize + 1, 2 * boxsize + 1)
aperture = get_aperture(sub_im_shape, 12)
annulus = get_annulus(sub_im_shape, 20, 40)

fluxes = np.zeros((len(ref_stars), len(img_files)))
flux_errs = np.empty_like(fluxes)

for i, f in enumerate(img_files):

    # read the image
    data, header = read_file(f)
    # use the fits header information for cordinate transformations
    w = WCS(header)

    # In each image, we want to find the position of each star
    for j, star in enumerate(ref_stars):
        # locate the star
        ref_pos = skycoord_to_pixel(star, w)
        ref_x, ref_y = ref_pos

        # we can then put look at a box around the star to set up our flux code above
        ref_x_int, ref_y_int = ref_x.astype(int), ref_y.astype(int)

        # generate region around star
        sub_im = data[ref_y_int - boxsize: ref_y_int + boxsize + 1,
                 ref_x_int - boxsize: ref_x_int + boxsize + 1]

        if sub_im.shape == (101,101):
          centroid = np.round(get_centroid(sub_im), 0).astype(int)
          #print("Original Centroid", centroid)

          ref_x_int += centroid[1]
          ref_y_int += centroid[0]

          sub_im = data[ref_y_int - boxsize: ref_y_int + boxsize + 1,
                   ref_x_int - boxsize: ref_x_int + boxsize + 1]

          # Notice that the centroid is not at a pixel interval,
          # we should be fine shifting the image as best we can
          new_centroid = get_centroid(sub_im)
          #print("New Centroid", new_centroid)

          """
          plt.figure(figsize=(7, 7))
          plt.imshow(sub_im, cmap='inferno', interpolation="None")
          plt.title(f"Image #{i + 1} Ref Star #{j + 1}", size=20)
          plt.axvline(x=sub_im.shape[0] // 2, color='red')
          plt.axhline(y=sub_im.shape[0] // 2, color='red')
          plt.colorbar()
          #plt.show()
          """

          shape = sub_im.shape[0]
          x = np.arange(shape)
          y = sub_im[(shape // 2)]
          amp = np.max(y)
          #popt, pcov = curve_fit(gaus, x, y, p0=[amp, sub_im.shape[0] // 2, 10, 0])

          fluxes[j, i], flux_errs[j, i] = pf.get_flux(sub_im, aperture, annulus)

        """
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, 'x', color='k', label='Central Row')
        plt.plot(x, gaus(x, *popt), color='r', alpha=0.3, label=f'Best-fit Gaussian (STD {round(popt[2], 1)})')
        plt.legend()
        plt.show()
        """

plt.figure()
plt.plot(file_times, fluxes[0])
plt.yscale("log")
plt.ylim(1e3, 1e5)

plt.show()

