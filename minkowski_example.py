#!/usr/bin/python3

"""In this example script, we use the test case of randomly placed 
    Boolean disks to verify the calculated Minkowski functionals."""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from minkowski import calculate_minkowski

# Specify the field size and the number and radii of the disks
# In calculating the expected Minkowski functionals, we will ignore the
# overlap of the disks. Using a small value of the radius and fewer disks
#  yields a favourable comparison with the expected functionals. In case of
# overlap, the expected values are an overestimate.
N = 2048 # field size
radius = 200 # radius of a disk
num_disks = 5 # number of disks

# Generate a random placement of the disks
positions = np.random.uniform(low=radius, high=N-radius, size=(num_disks, 2))
distances = cdist(positions, positions)
binary = np.zeros((N, N), dtype=bool)
for i in range(num_disks):
    x, y = np.ogrid[-positions[i,0]:N-positions[i,0], -positions[i,1]:N-positions[i,1]]
    mask = x*x + y*y <= radius*radius
    binary[mask] = 1

# Calculate the Minkowski functionals of the disks
M0, M1, M2 = calculate_minkowski(binary, threshold=None, geom=True)

# We evaluate the expected values of the functionals
# In case of overlap, the expected values are an overestimate
# and the calculated values will be smaller
M0_e = np.round(num_disks * np.pi * radius**2, 3) # expected total area of the disks
M1_e = np.round(num_disks * 2 * np.pi * radius, 3) # expected circumference of the disks
M2_e = float(num_disks) # expected euler characteristic

# We plot the binary field with the randomly placed disks
# and display the calculated a expected values of the functionals
plt.rcParams.update({"text.usetex": True})
fig, ax = plt.subplots()
ax.set_title('Boolean disks', fontsize=16)
ax.minorticks_on()
image = plt.imshow(binary, cmap='viridis')
cbar = plt.colorbar(image)
cbar.set_label('Field Value', fontsize=14)
ax.set_xlabel(r'$x/L$', fontsize=14)
ax.set_ylabel(r'$y/L$', fontsize=14)
M0_str = "$M_{{0}} = ({}, {})$\n".format(M0, M0_e)
M1_str = "$M_{{1}} = ({}, {})$\n".format(M1, M1_e)
M2_str = "$M_{{2}} = ({}, {})$".format(M2, M2_e)
text_str = "MF = (calculated, expected)\n" + M0_str + M1_str + M2_str
ax.text(1.25, 0.85, text_str, transform=ax.transAxes, fontsize=11)
plt.tight_layout()
plt.savefig('./plots/MF_boolean_disks.png', bbox_inches='tight', dpi=300)
plt.close()