import numpy as np
import matplotlib.pyplot as plt
from gen_gauss_2D import gauss_2D


density_field = gauss_2D(128, alpha=1)

# Define the bins for the histogram
bins = np.linspace(np.min(density_field), np.max(density_field), 50)

# Calculate the histogram of the density field
hist, bins = np.histogram(density_field, bins=bins, density=True)

# Calculate the PDF from the histogram
PDF = hist * np.diff(bins)

# Plot the PDF
plt.plot(bins[:-1], PDF)
plt.xlabel('Density')
plt.ylabel('PDF')
plt.show()