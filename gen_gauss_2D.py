#!/usr/bin/python3

# Main dependencies
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def gauss_2D(size: int = 1024, alpha: float = 1, norm: bool = True, 
            visualise: bool = False, save_name: str = 'gauss_2D') -> np.array:

    """Returns a 2D Gaussian field of specified size from a power spectrum
        with scale alpha. If norm is True, the resulting field will have zero
        mean and unit standard deviation."""

    # Set up the k-space grid
    k_ind = np.mgrid[:size, :size] - int((size+1)/2)
    k_idx = np.fft.fftshift(k_ind)

    # Define the amplitude as a power law |k|^(-alpha)
    amplitude = np.power(k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha)
    amplitude[0,0] = 0

    # Generate the field and transford to real space
    noise = np.random.normal(size = (size, size)) + 1j * np.random.normal(size = (size, size))
    gfield = np.fft.ifft2(noise * amplitude).real

    # Normalise the field if specified
    if norm == True:
        # Sets the mean to zero and standard deviation to one
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)

    else:
        pass

    # Plot the field if specified
    if visualise == True:
        plt.rcParams.update({"text.usetex": True})
        plt.rcParams.update({"font.family": "serif"})
        fig, ax = plt.subplots()
        ax.minorticks_on()
        plt.imshow(gfield, cmap='rainbow', extent=(0, 0.5, 0, 0.5))
        ax.set_xlabel(r'$x/L$', fontsize=18)
        ax.set_ylabel(r'$y/L$', fontsize=18)
        cbar = plt.colorbar()
        cbar.set_label(r'$\delta(\mathbf{x})$', fontsize=20)
        plt.savefig('./plots/{}.png'.format(save_name),dpi=300)
        plt.close()

    else:
        pass

    return gfield



# # Generate a 2D density field with random values
# N = 512 # size of field
# field = gauss_2D(size=N, visualise=True)
# # field_k = np.fft.fft2(field)

# # # Bin the power in specified k-modes to get a 1D array
# # power_spectrum = np.abs(field_k)**2

# # # calculate the frequency grid
# # freqs = np.fft.fftfreq(N, d=1./N)
# # freqs_x, freqs_y = np.meshgrid(freqs, freqs)
# # freqs = np.sqrt(freqs_x**2 + freqs_y**2)

# # # calculate the radial average of the power spectrum
# # bins = np.arange(0, 1.5*np.max(freqs), 1)
# # digitized = np.digitize(freqs.flat, bins)
# # bin_means = [power_spectrum.flat[freqs.flat == i].mean() for i in range(1, len(bins))]
# # bin_centers = (bins[1:] + bins[:-1])/2

# # # plot the power spectrum
# # import matplotlib.pyplot as plt
# # plt.loglog(bin_centers, bin_means)
# # plt.xlabel('Frequency')
# # plt.ylabel('Power Spectrum')
# # plt.show()


# # def power(gfield: np.array) -> np.array:
# #     """Returns the power spectrum for a given density field."""