from astropy.wcs import WCS
import astropy
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc

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

"""==================== FUNCTIONS ======================="""


def show_ds9(data, title):
    """
    Graphs the DS9 style plot of the data using matplotlib.pyplot.imshow()

    :param data: 2D NumPy array, data being plotted
    :param title: str, Title for the plot
    """
    plt.figure()
    plt.imshow(data, vmin=-100,
               vmax=1e2,
               origin='lower',
               cmap='inferno'
               )
    plt.title(title)


def gaussian(x, a, x0, sigma, c):
    """
    Function of a Gaussian function used for fitting star data to find the best aperture radius

    :param x: float, x position
    :param a: float, peak of the gaussian
    :param x0: float, shift in x from 0
    :param sigma: float, standard deviation
    :param c: float, scale factor
    :return: float, y value from a gaussian distribution
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c


def centroid(image):
    """
    Finds the (x,y) pixel positions of the centroids in a 2D array based on weighted averages

    :param image: 2D numpy array, sub image data
    :return: float, Centroid of x and float, Centroid of y
    """
    # Sets values to hold the summations
    I, Ix, Iy = 0.0, 0.0, 0.0

    # Loops through the rows of the image
    for j in range(0, len(image)):

        # Loops through each pixel in the row
        for i in range(0, len(image[j])):
            # Sums the values from each pixel
            I += image[j, i]
            Ix += i * image[j, i]
            Iy += j * image[j, i]

    return Ix / I, Iy / I


def get_aperture(shape, center, radius):
    """
    Creates a binary mask in the shape of a circle with radius r centered in the middle

    :param shape: Tuple, shape of the array (rows, columns)
    :param center: Tuple, center coordinates of the circle (row, column).
    :param radius: int, radius of the circle.
    :return: 2D NumPy array, binary mask for the inner circle.
    """

    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    return mask.astype(np.uint8)


def get_annulus(shape, center, radii):
    """
    Creates a binary mask in the shape of a ring with inner radius radii[0] and outer radius radii[1] centered in the
    middle

    :param shape: Tuple, shape of the array (rows, columns)
    :param center: Tuple, center coordinates of the circle (row, column).
    :param radii: Tuple, inner radius and outer radius
    :return: 2D NumPy array, binary mask of the ring
    """
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    mask = (radii[0] ** 2 <= (x - center[1]) ** 2 + (y - center[0]) ** 2) & \
           ((x - center[1]) ** 2 + (y - center[0]) ** 2 <= radii[1] ** 2)
    return mask.astype(np.uint8)


# TODO: CHECK IF THE CALCULATIONS GO CORRECTLY
def get_flux(sub, app, ann, gain=4):
    """
    Calculates intensity of star and the uncertainty using the binary masks for the aperture and annulus overtop the
    sub image data.

    :param sub: 2D NumPy array, sub image data
    :param app: 2D NumPy array, aperture mask
    :param ann: 2D NumPy array, annulus mask
    :param gain: int, gain with a preset value of 4
    :return: float, flux and float, flux uncertainty
    """

    N_app = app[app > 0].size
    N_ann = ann[ann > 0].size
    if N_ann > 0:
        I_bg = np.sum(sub * ann) / N_ann
    else:
        I_bg = 0
        print("ERROR")
    F = np.sum(sub[app > 0] * app[app > 0] - I_bg)

    std_bg = np.std(ann)
    unc_F = np.sqrt((F / gain) + N_app * (np.pi / 2) * (N_app / N_ann) * std_bg ** 2)

    return F, unc_F


def magnitude(I1, uI1, I2, uI2, m2, um2):
    """
    Calculates the apparent magnitude of a celestial object based on data from itself and one reference object with
    known values.

    :param I1: float, intensity of celestial object
    :param uI1: float, uncertainty in intensity of celestial object
    :param I2: float, intensity of reference object
    :param uI2: float, uncertainty in intensity of reference object
    :param m2: float, apparent magnitude of reference object
    :param um2: float, uncertainty in apparent magnitude of reference object
    :return: float, apparent magnitude of celestial object and float, uncertainty in apparent magnitude of celestial
             object
    """
    # Uses equation listed on Slide __ of ___
    m1 = m2 - 2.512 * np.log10(I1 / I2)
    # Derived from information listen on Slide __ of ___
    um1 = um2 ** 2 + (-2.512 / (np.log(10) * I1)) ** 2 * uI1 ** 2 + (2.512 / (np.log(10) * I2)) ** 2 * uI2 ** 2
    return m1, um1


def fluxes(data, key, w):
    """
    Takes in fits data and for a given reference star makes a sub image around it. Finds the optimal radius for aperture
    and annulus masks and determines if a file is poor quality. This is done through the Gaussian curve fit made from a
    ross section across the center of the sub image. If the returned standard deviation is less than 5 but greater
    than 1.5, the file is considered good and bool is kept False.

    :param data: 2D NumPy array, fits file data
    :param key: str, key for OBJECTS dictionary
    :param w: #TODO: WHAT IS THIS??
    :return: 2D NumPy array, sub image centered on reference star,
             2D NumPy array,aperture data,
             2D NumPy array, annulus data,
             bool: True if file is considered bad
    """
    # Set bad_file to False until shown to be poor quality
    bad_file = False

    # Takes the (RA, DEC) of the object and finds the correlated pixels in the data set
    ref_pos = skycoord_to_pixel(OBJECTS[key][0], w)
    x_raw, y_raw = ref_pos
    x, y = x_raw.astype(int), y_raw.astype(int)

    # Setting the sub image size. This value will be the horizontal and vertical distance from the center point
    # The sub image is then made as a subset of the whole fits data centered on the found pixels
    boxsize = 50
    sub_im = data[y - boxsize: y + boxsize + 1, x - boxsize: x + boxsize + 1]

    # Finds the centroids of the plot and then calculates the pixel offset from the center and the centroid
    # The offset is then accounted for and a new sub image is created with the object now centered
    """
    cx, cy = centroid(sub_im)
    x_offset, y_offset = (x-cx).astype(int), (y-cy).astype(int)
    x, y = x+x_offset, y+y_offset
    sub_im = data[y - boxsize: y + boxsize + 1, x - boxsize: x + boxsize + 1]
    """

    # Checks if the final image is the correct shape
    if sub_im.shape == (2*boxsize+1, 2*boxsize+1):

        # Gets a tuple with the shape of the 2D array to then get a singular row going through the center of the object
        # Gaussian curve fit is then applied to find the standard deviation from the rows intensity data
        shape = sub_im.shape[0]
        gaus_x, gaus_y = np.arange(shape), sub_im[shape // 2, :]
        popt, _ = sc.curve_fit(gaussian, gaus_x, gaus_y, p0=[max(gaus_y), shape // 2, 10, 0], maxfev=3000)

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
            aperture = get_aperture(sub_im.shape, (50, 50), radius)
            annulus = get_annulus(sub_im.shape, (50, 50), (radius + 4, radius + 7))
        else:
            bad_file = True
            aperture = get_aperture(sub_im.shape, (50, 50), 10)
            annulus = get_annulus(sub_im.shape, (50, 50), (12, 15))

        # Plot the star sub images TODO: hopefully centered next time
        sub_im_plotting(sub_im, radius)

        return sub_im, aperture, annulus, bad_file
    else:
        return sub_im, 0, 0, True


def sub_im_plotting(sub_im, radius):
    plt.figure(figsize=(7, 7))
    plt.title(f"Radius of {radius:.4}")
    plt.imshow(sub_im, cmap='inferno', interpolation="None")
    plt.axvline(x=sub_im.shape[0] // 2, color='red')
    plt.axhline(y=sub_im.shape[0] // 2, color='red')
    plt.colorbar()
    plt.show()


def light_curve(t, C, t_1):
    """

    :param t:
    :param C:
    :param t_1:
    :return:
    """
    return C * (t - t_1) ** 2


def mag_to_flux(m, const=0):
    """
    Converts the magnitude to an intensity value for looking at the light curve
    :param m: apparent magnitude
    :param const: constant which can be set if needed
    :return: Intensity relating to the provided magnitude
    """
    return 10 ** (-m / 2.5) * 10 ** (const / 2.5)
