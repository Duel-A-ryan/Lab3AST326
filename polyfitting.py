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
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings("ignore", category=astropy.wcs.FITSFixedWarning)

"""=====MAIN====="""

times = np.loadtxt("Data/cleaned data/times")
mags = np.loadtxt("Data/cleaned data/mags")
mag_err = np.loadtxt("Data/cleaned data/mag_uncs")

times = times[~np.isnan(mags)]
mag_err = mag_err[~np.isnan(mags)]
mags = mags[~np.isnan(mags)]

s2n = mags/mag_err

times = times[s2n >= 3]
mags = mags[s2n >= 3]

times = times[17< mags]
times = times[mags < 23]
mags = mags[17< mags]
mags = mags[mags < 23]

plt.figure(figsize=(15, 5))

plt.scatter(times, mags)
for n in range(2, 5):
    p, res, _, _, _ = np.polyfit(times, mags, n, full=True)
    pred = np.polyval(p, times)
    print(p, res)
    #plt.plot(times, pred, label=f"n={n}")
    #pf.reduced_chi_squared(mags, pred, -1, 1)

#plt.ylim(24, 18)
#plt.legend()
#plt.show()
