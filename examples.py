#!/usr/bin/python3

# In this example script, we use the entropy_calc
# function from the library to calculate the
# the relative entropy of two different estimators.

# Import libraries
import numpy as np
from rel_entropy import entropy_calc

# Set mean and standard deviation for Gaussian distribution
# and generate n_samples random points.
mu, sigma, n_samples = 0, 0.5, 1000
data = np.random.uniform(mu, sigma, n_samples)

# Example 1: In this example, we use the usual unbiased and
# biased estimators of the sample variance. Both estimators
# will give similar distributions after the jackknife
# resampling, and the relative entropy should be zero.

# Define the two estimators
est_1 = lambda x: np.var(x, ddof=0) # biased
est_2 = lambda x: np.var(x, ddof=1) # unbiased

# Calculate the relative entropy and print the results
ent_12, ent_21 = entropy_calc(data, est_1, est_2, visualise=True, save_name='unbiased_biased')

print('S_12 = {}, S_21 = {}'.format(ent_12,
            ent_21))


# Example 2: In this example, we compare the unbiased estimator
# and the range, i.e. the difference between the maximum 
# and minumum values of the data. The latter will produce the 
# wrong estimate, and a different distribution, and we can see
# the utility of the relative entropy.

# Define the two estimators
est_1 = lambda x: np.var(x, ddof=0) # biased
est_2 = lambda x: np.max(x) - np.min(x) # range
# Calculate the relative entropy and print the results
ent_12, ent_21 = entropy_calc(data, est_1, est_2, visualise=False)
print('S_12 = {}, S_21 = {}'.format(ent_12,
            ent_21))


# Example 3: In this example, we compare the unbiased estimator
# and the interquartile range (75-25). The latter will produce the 
# wrong estimate, and a different distribution.

# Define the two estimators
est_1 = lambda x: np.var(x, ddof=0) # biased
est_2 = lambda x: np.percentile(x, 75) - np.percentile(x, 25) # interquartile range
# Calculate the relative entropy and print the results
ent_12, ent_21 = entropy_calc(data, est_1, est_2, visualise=True)
print('S_12 = {}, S_21 = {}'.format(ent_12,
            ent_21))
