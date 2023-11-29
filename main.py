import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

"""==================== FUNCTIONS ======================="""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def show_ds9(data):
    plt.imshow(data, norm=LogNorm())
    plt.title("Log Plot")
    plt.figure()
    plt.imshow(data, vmin=100)
    plt.title("Normal Plot")
    plt.show()


# TODO: Just check if x,y coordinates are set up in correct order
def object_aperture(data, center, r):
    """
    :param data: 2D array of intensity data
    :param center: central points to gather intensities from
    :param r: radius set to gather points from
    :return: average intensity, uncertainty in the average intensity
    """
    points = []
    for j in range(0, len(data)):
        for i in range(0, len(data[0])):
            if (i - center[0]) ** 2 + (j - center[1]) ** 2 < r ** 2:
                points.append(data[i][j])

    average = np.mean(points)

    diff = [(points[i] - average) ** 2 for i in range(0, len(points))]
    sigma = np.sqrt(np.sum(diff) / (len(diff) - 1))

    return average, len(points), sigma


# Helper function to do the dividing because it isn't working for some reason
def help_divide(num1, num2):
    return num1 / num2


# TODO: Same this as above function
def sky_annulus(data, center, r_small, r_large):
    """
    :param data: 2D array of intensity data
    :param center: central points to gather intensities from
    :param r: radius set to gather points from
    :return: average intensity, uncertainty in the average intensity
    """
    points = []
    for j in range(0, len(data)):
        for i in range(0, len(data[0])):
            dot = (i - center[0]) ** 2 + (j - center[1]) ** 2
            if r_large ** 2 >= dot >= r_small ** 2:
                points.append(data[j][i])

    average = np.mean(points)

    diff = [(points[i] - average) ** 2 for i in range(0, len(points))]
    sigma = np.sqrt(np.sum(diff) / (len(diff) - 1))

    return average, len(points), sigma


# TODO: Determine this is correct
def stellar_intensity(aperture, num_aperture, annulus, unc_annulus, num_annulus):
    answer = (aperture * num_aperture) - num_aperture * annulus

    unc_source = np.sqrt(answer / 4)
    unc_background = np.sqrt(num_annulus * unc_annulus)

    unc_answer = np.sqrt(unc_source ** 2 + unc_background ** 2)

    return answer, unc_answer, answer / unc_answer


def magnitude(m_2, u_m_2, I_1, u_I_1, I_2, u_I_2):
    """
    :param m_2: Magnitude of known source
    :param I_1: Intensity of unknown source
    :param I_2: Intensity of known source
    :return: Magnitude of known source
    """
    m_1 = m_2 - 2.5 * np.log10(I_1 / I_2)
    u_m_1 = np.sqrt(u_m_2 ** 2 + (((314 * I_1) / 125) ** 2) * u_I_1 ** 2 + ((314 / (125 * I_2)) ** 2) * u_I_2 ** 2)

    return m_1, u_m_1


# NOTE: NOT MY FUNCTIONS. THIS SHOULD BE REWRITTEN AS MY OWN LATER TO AVOID PLAGIARISM
def find_centroids(array, threshold=0):
    rows, cols = array.shape
    visited = np.zeros_like(array, dtype=bool)
    centroids = []

    def dfs(row, col, object_region):
        if 0 <= row < rows and 0 <= col < cols and array[row, col] > threshold and not visited[row, col]:
            visited[row, col] = True
            object_region[row, col] = 1

            # Explore neighbors (up, down, left, right)
            dfs(row - 1, col, object_region)  # Up
            dfs(row + 1, col, object_region)  # Down
            dfs(row, col - 1, object_region)  # Left
            dfs(row, col + 1, object_region)  # Right

    for row in range(rows):
        for col in range(cols):
            if array[row, col] > threshold and not visited[row, col]:
                object_region = np.zeros_like(array)
                dfs(row, col, object_region)

                # Calculate centroid for the object
                centroid_row, centroid_col = find_centroid(object_region)
                centroids.append((centroid_row, centroid_col))

    return centroids


def find_centroid(array):
    rows, cols = array.shape
    total_intensity = np.sum(array)

    # Calculate weighted sum of row and column indices
    row_indices = np.arange(rows)
    col_indices = np.arange(cols)
    row_sum = np.sum(array * row_indices[:, np.newaxis])
    col_sum = np.sum(array * col_indices)

    # Calculate centroid coordinates
    centroid_row = row_sum / total_intensity
    centroid_col = col_sum / total_intensity

    return centroid_row, centroid_col


# Read text file of all the file names
with open("Data/AST325-326-SN-filelist.txt") as f:
    filenames = ["Data/Fits/" + line for line in f.readlines()]

# Opens file in filenames and save the 2D array data
num = 11
hdu = fits.open(filenames[num][:-1])
print(f"The file chosen is number {num-1} and is {filenames[num][10:-1]}")

# Save 2D array data and find the 4 stellar positions in terms of (x,y) pixel coordinates
data = hdu[0].data
centroids = find_centroids(data, 2000)
print(centroids)

show_ds9(data)

# Find the intensity and uncertainty of the four found centroids
I_1, num_1, sI_1 = object_aperture(data, centroids[0], 7)
unc_1 = help_divide(sI_1, np.sqrt(num_1))
print(f"The found intensity is {I_1:.3} \u00B1 {unc_1:.3}. This correlates to a margin of error of "
      f"{unc_1/I_1 * 100:.3}%")

sky_1, numsky_1, sky_s_1 = sky_annulus(data, centroids[0], 8, 11)
unc_sky_1 = help_divide(sky_s_1, np.sqrt(numsky_1))
print(f"The found sky annulus is {sky_1:.3} \u00B1 {unc_sky_1:.3}. This correlates to a margin of error of "
      f"{unc_sky_1/sky_1 * 100:.3}%")
"""
I_2, num_2, sI_2 = object_aperture(data, centroids[1], 7)
unc_2 = help_divide(sI_2, np.sqrt(num_2))
print(f"The found intensity is {I_2:.3} \u00B1 {unc_2:.3}. This correlates to a margin of error of "
      f"{unc_2/I_2 * 100:.3}%")

I_3, num_3, sI_3 = object_aperture(data, centroids[2], 7)
unc_3 = help_divide(sI_3, np.sqrt(num_3))
print(f"The found intensity is {I_3:.3} \u00B1 {unc_3:.3}. This correlates to a margin of error of "
      f"{unc_3/I_3 * 100:.3}%")


#IS NOT WORKING. LOOK INTO THAT
I_4, num_4, sI_4 = object_aperture(data, centroids[3], 7)
unc_4 = help_divide(sI_4, np.sqrt(num_4))
print(f"The found intensity is {I_4:.3} \u00B1 {unc_4:.3}. This correlates to a margin of error of "
      f"{unc_4/I_4 * 100:.3}%")
"""

# TODO: Continue from Lecture 9 Slides.
