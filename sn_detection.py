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
file = "Data/Fits/AST325-326-SN-20150920.9014.fits"
f = fits.open(file)
data = f[0].data
header = f[0].header

dt = datetime.fromisoformat(header["DATE-OBS"])

pf.show_ds9(data, dt)

plt.figure(figsize=(12,5))
plt.hist(data.flatten(), bins = 'sturges', color = 'k', histtype='step')
plt.grid(alpha=0.5)
plt.yscale('log')
plt.xlabel("Pixel Value [ADU]", size = 16)
plt.ylabel("Number of Pixels", size = 16)
plt.show()


def two_d_gaussian(x, y, ux, uy, sx, sy, A):
    return A * np.exp(-1 * ((x - ux) ** 2 / (2 * sx ** 2) + (y - uy) ** 2 / (2 * sy ** 2)))


def gaus(x, a, x0, sigma, c):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c


def fake_star(amplitude, noise):
    axis = np.arange(-100, 101).astype(float)
    x, y = np.meshgrid(axis, axis)

    data = np.zeros_like(x)

    data += two_d_gaussian(x, y, 0, 0, 3, 3, amplitude)

    return data + np.random.normal(0, noise, x.shape) + 20


file = fits.open("Data/Fits/AST325-326-SN-20151009.0868.fits")[0]
data, header = file.data, file.header

data[data < 0] = 0  # Sets any negative values to zero
w = WCS(header)

# Finding pixel center for detected supernova
ref_pos = skycoord_to_pixel(OBJECTS["star_1"][0], w)
x_raw, y_raw = ref_pos
x, y = x_raw.astype(int), y_raw.astype(int)

# Setting the sub image size
boxsize = 50

sub_im = data[y - boxsize: y + boxsize + 1, x - boxsize: x + boxsize + 1]
pf.show_ds9(sub_im, "test")
shape = sub_im.shape[0]
x, y = np.arange(shape), sub_im[shape // 2, :]
popt, pcov = sc.curve_fit(pf.gaussian, x, y, p0=[max(y), shape//2, 10, 0])

print(popt)
plt.figure(figsize=(10, 5))
plt.plot(x, y, 'x', color='k', label='Central Row')
plt.plot(x, gaus(x, *popt), color='r', alpha=0.3, label=f'Best-fit Gaussian (STD {round(popt[2], 1)})')
plt.legend()
plt.show()

# potential = ["Data/Fits/AST325-326-SN-20150924.9819.fits", ]