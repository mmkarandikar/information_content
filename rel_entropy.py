#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from typing import Callable

def entropy_calc(data: np.array, est_1: Callable, est_2: Callable, 
                 visualise: bool = False, save_name: None = str) -> tuple:
    """Calculates the relative entropy of two input estimators of the variance
      of given input data."""
    # Define functions
    def jackknife_variance(data: np.array, variance_func: Callable) -> tuple:
        """Use jackknife resampling to construct a distribution of
            variance estimates from given data and estimating function. Returns
            the distribution and the estimated variance after resampling."""
        n = data.shape[0]
        indices = np.arange(n)
        variances = np.zeros(n)
        for i in range(n):
            mask = indices != i
            variances[i] = variance_func(data[mask])
        jack_var = np.var(variances, ddof=1)
        return jack_var, variances

    # Compute jackknife estimate of sample variance using both estimators
    jack_var_unbiased, var_dist_unbiased = jackknife_variance(data, est_1)
    jack_var_biased, var_dist_biased = jackknife_variance(data, est_2)

    # Calculate the relative entropy
    # First, bin the distributions to get a histogram
    bins = 100
    counts_unbiased = np.histogram(var_dist_unbiased, bins=bins, density=True)[0]
    counts_biased = np.histogram(var_dist_biased, bins=bins, density=True)[0]

    # Next, use the scipy function 'entropy' to get the relative entropy
    # The unit of entropy is bits
    eps = 1e-10 # Add a small number to the counts to avoid division by zero
    relative_entropy_12 = np.round(entropy(counts_unbiased+eps, counts_biased+eps, base=2), 4)
    relative_entropy_21 = np.round(entropy(counts_biased+eps, counts_unbiased+eps, base=2), 4)

    if visualise == True:
        # Plot histograms of jackknife variance estimates
        plt.rcParams.update({"text.usetex": True})
        fig, ax = plt.subplots()
        ax.hist(var_dist_unbiased, color='b', bins=20, alpha=1, 
                label='Unbiased estimator')
        ax.hist(var_dist_biased, color='r', bins=20, alpha=0.5,
                label='Biased estimator')
        ax.axvline(np.var(data, ddof=1), color='k', linestyle='dashed',
                    linewidth=1, label='True sample variance')
        ax.set_xlabel('Sample variance', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.text(x=0.25, y=0.25, 
                s='$S_{{12}}$ = {0:03f} \n $S_{{21}}$ = {0:03f}'.format(relative_entropy_12,
                relative_entropy_21), transform=ax.transAxes)
        plt.legend()
        if save_name is not None:
            plt.savefig('plots/{}.png'.format(save_name), bbox_inches='tight', dpi=300)
        else:
            plt.savefig('plots/example_3.png', bbox_inches='tight', dpi=300)
        plt.close()

    else:
        pass

    return relative_entropy_12, relative_entropy_21
