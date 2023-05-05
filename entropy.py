#!/usr/bin/python3

import numpy as np
from scipy.stats import entropy as ent
from sklearn.metrics import mutual_info_score
from gen_gauss_2D import gauss_2D

# Generate a random field
N = 32
field = gauss_2D(N)

# Compute the power spectrum of the random field
field_k = np.fft.fft2(field)
power_spectrum = np.abs(field_k)**2


mi = mutual_info_score(field.flatten(), power_spectrum.flatten())

print('Mutual Information: {}'.format(mi))