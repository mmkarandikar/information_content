#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
from gen_gauss_2D import gauss_2D

import numpy as np
from scipy.signal import correlate2d

# Define the field
size = 128
field = gauss_2D(size=size, alpha=1)
field += gauss_2D(size=size, alpha=0)
field += gauss_2D(size=size, alpha=2)


# # Create a non-Gaussian component by adding a sinusoidal modulation
# x, y = np.meshgrid(np.arange(size), np.arange(size))
# field += 1 * np.sin(2 * np.pi * (x + y) / size)

# Calculate the autocorrelation function
autocorr = correlate2d(field, field, mode='same')
corrfunc = autocorr #/ (field.size * field.mean()**2)

# Calculate the distance from the center of the field to each point
x, y = np.meshgrid(np.arange(field.shape[0]), np.arange(field.shape[1]))
r = np.sqrt((x - field.shape[0]/2)**2 + (y - field.shape[1]/2)**2)

# Calculate the radial average of the correlation function
# Define the bins
bins = np.arange(0, np.ceil(r.max()), 1)
# Get the indices of the bins
digitized = np.digitize(r.flat, bins)
radial_avg = np.zeros(len(bins))
for i in range(1, len(bins)):
    # For each bin, find the indices of the correlation function
    # that fall within that bin, then calculate the mean of those indices
    radial_avg[i-1] = corrfunc.flat[digitized == i].mean()


def three_point_function(field, r_bins):
    # Compute the shape of the field
    nx, ny = field.shape

    # Generate a grid of indices
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

    # Compute the pairwise distances between all points
    dx = x[:, :, np.newaxis] - x[:, np.newaxis, :]
    dy = y[:, :, np.newaxis] - y[:, np.newaxis, :]
    distances = np.sqrt(dx**2 + dy**2)

    # Compute the product of the field at all points
    product = field[:, :, np.newaxis] * field[:, np.newaxis, :] * field[np.newaxis, :, :]

    # Compute the bin indices for each pair of points
    bin_indices = np.digitize(distances.flat, r_bins, right=True)

    # Compute the sum and count for each bin using bincount
    sums = np.bincount(bin_indices, weights=product.flat)
    counts = np.bincount(bin_indices)

    # Compute the average for each bin
    average = sums[1:] / counts[1:]

    return average


r_bins = np.arange(0, r.max(), 1)
three_pcf = three_point_function(field, r_bins)

# Plot the field and the correlation function
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(10, 6))#, sharex=True, sharey=True)
for j in range(2):
    ax[j].minorticks_on()
im = ax[0].imshow(field, cmap='rainbow')
# plt.colorbar(im, use_gridspec=True)
ax[0].set_title('Field', fontsize=16)

# im_corr = ax[1].imshow(corrfunc, cmap='gray')
ax[1].plot(bins[:-1], radial_avg[:-1], c='b', label='2-PCF')
ax[1].plot(r_bins[:-1], three_pcf[:-1], c='r', label='3-PCF')
ax[1].legend()
# plt.colorbar(im_corr, use_gridspec=True)
ax[1].set_title('Radially averaged n-PCFs', fontsize=16)
plt.tight_layout()
plt.show()
