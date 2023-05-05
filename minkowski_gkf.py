#!/usr/bin/python3

# [1]: https://arxiv.org/pdf/1812.07310.pdf

import scipy
import numpy as np
import matplotlib.pyplot as plt
from minkowski import calculate_minkowski
from gen_gauss_2D import gauss_2D
from tqdm import tqdm
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit

## The MF1 of a GRF averaged over many realisations
#  as a function of the threshold is also a Gaussian! Not surprising?

N = 64
# alpha = 10
# alphas = [-2, -1, 0, 1, 2]

alphas = np.array([-0.5]) #np.arange(-1, 1, 0.25) #
color = iter(cm.rainbow(np.linspace(0, 1, alphas.size)))
fil = None

for MF in ['Q1']:#, 'Q1', 'Q2']:

    plt.rcParams.update({"text.usetex": True})
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.set_xlabel(r'$\nu$', fontsize=18)
    # ax.set_ylabel(r'$\lambda(\nu)$', fontsize=14)
    # ax.set_ylabel(r'$\langle \hat{Q}_{0}(\nu)\rangle_{\mathrm{r}}$', fontsize=18)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=16)
    # define a threshold range
    n_real = 250
    n_thresh = 50 #number of values to use as a threshold
    # field = gauss_2D(size=N, alpha=alpha)
    # thresh_range = np.linspace(field.min(), field.max(), n_thresh)
    thresh_range = np.linspace(-4, 4, n_thresh)
    sigmas, lambdas = [], []
    for alpha in (alphas):
        ax.set_title(rf'$\alpha = {alpha}$', fontsize=20)
        Q0_measured, Q1_measured, Q2_measured, Q0_err, Q1_err, Q2_err = [], [], [], [], [], []
        Q0_all_meas = np.empty(shape=(n_real, n_thresh))
        Q1_all_meas = np.empty(shape=(n_real, n_thresh))
        Q2_all_meas = np.empty(shape=(n_real, n_thresh))

        for j in tqdm(range(n_thresh)):
            sigmas_real = []
            thresh = thresh_range[j]
            # for each realisation and a single threshold, we store the Q-values
            Q0, Q1, Q2 = [], [], []
            for i in (range(n_real)):
                field = gauss_2D(l_box=100, size=N, alpha=alpha, norm=True)
                Q0_num, Q1_num, Q2_num = calculate_minkowski(field, thresh, norm=True)
                Q0.append(Q0_num)
                Q1.append(Q1_num)
                Q2.append(Q2_num)

                Q0_all_meas[i,j] = Q0_num
                Q1_all_meas[i,j] = Q1_num
                Q2_all_meas[i,j] = Q2_num

            Q0_measured.append(np.mean(Q0))
            Q1_measured.append(np.mean(Q1))
            Q2_measured.append(np.mean(Q2))

    Q0_measured = np.array(Q0_measured)
    Q1_measured = np.array(Q1_measured)
    Q2_measured = np.array(Q2_measured)

    if fil is None:
        fil = 1

    if MF == 'Q0':
        # Fitting with Q0
        def fitting_function_Q0(X, a0):
            return 0.5 - 0.5*scipy.special.erf(X / np.sqrt(2*a0))

        Q0_cov = np.cov(Q0_all_meas.T)
        Q0_cov[Q0_cov < 1e-10] = 0 # define a threshold, here 1e-10 
        # find the indices of the non-zero diagonal elements
        non_zero_diag = np.nonzero(np.diagonal(Q0_cov))[0]
        new_size = Q0_cov.shape[0] - len(non_zero_diag)
        Q0_cov_cut = Q0_cov[non_zero_diag][:, non_zero_diag]
        print(np.linalg.det(Q0_cov_cut))
        C, cov = curve_fit(fitting_function_Q0, thresh_range[non_zero_diag], Q0_measured[non_zero_diag], sigma=Q0_cov_cut, method='lm', absolute_sigma=True)
        Q0_fit = fitting_function_Q0(thresh_range, C[0])
        ax.set_ylabel(r'$\langle \hat{Q}_{0}(\nu)\rangle_{\mathrm{r}}$', fontsize=18)
        ax.plot(thresh_range, Q0_measured, c='b', lw=1.5, ls='solid', label=r'measured')
        ax.plot(thresh_range, Q0_fit, c='r', lw=1.5, ls='dashed', label=r'fit')
        ax.text(0.65, 0.75, rf'$\sigma = {np.round(C[0],3)}\pm{np.round(cov[0][0], 6)}$', transform=ax.transAxes, fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()

        lam_exp = np.sqrt(np.abs(alpha + 3 / (2 * np.pi * fil**2)))
        print(rf'sigma_exp = {lam_exp}, sigma_mea = {C[0]}')
        plt.show()
        # plt.savefig('plots/Q0_fit.png', dpi=300)
        # plt.close()

    elif MF == 'Q1':
        # Fitting with Q1
        def fitting_function_Q1(X, a0, a1):
            return (np.sqrt(np.pi)*a0/(4*np.sqrt(2))) * np.exp(-X**2 / (2*a1))

        def model(thresh: np.ndarray, lam: float = 1, sigma: float = 1) -> np.ndarray:
            return (np.sqrt(np.pi)*lam/(4*np.sqrt(2))) * np.exp(-thresh**2 / 2)

        # helper function that calculates the lambda for a given alpha
        def lamb_mik(alpha, k_min, k_max):
            xi_dd = np.abs(-(np.power(k_max, alpha + 4) - np.power(k_min,alpha + 4))/(alpha + 4))
            xi = (np.power(k_max, alpha + 2) - np.power(k_min, alpha + 2))/(alpha + 2)
            return np.sqrt(xi_dd/(2*np.pi*xi))

        k_min = 2*np.pi / 100
        k_max = (N//2 - 1) * k_min
        print(k_min, k_max)
        lambda_true = lamb_mik(alphas[0], k_min, k_max)
        Q1_model = model(thresh_range, lambda_true)

        # Ignore the points where sigma is zero
        Q1_cov = np.cov(Q1_all_meas.T)
        Q1_cov[Q1_cov < 1e-8] = 0 # define a threshold, here 1e-10 
        # find the indices of the non-zero diagonal elements
        non_zero_diag = np.nonzero(np.diagonal(Q1_cov))[0]
        new_size = Q1_cov.shape[0] - len(non_zero_diag)
        Q1_cov_cut = Q1_cov[non_zero_diag][:, non_zero_diag]
        C, cov = curve_fit(fitting_function_Q1, thresh_range[non_zero_diag], Q1_measured[non_zero_diag], sigma=Q1_cov_cut, method='lm', absolute_sigma=True)
        Q1_fit = fitting_function_Q1(thresh_range, C[0], C[1])
        ax.set_ylabel(r'$\langle \hat{Q}_{1}(\nu)\rangle_{\mathrm{r}}$', fontsize=18)
        ax.plot(thresh_range, Q1_measured, c='b', lw=1.5, ls='solid', label=r'measured')
        ax.plot(thresh_range, Q1_model, c='k', lw=1.5, ls='dashdot', label=r'model')
        ax.plot(thresh_range, Q1_fit, c='r', lw=1.5, ls='dashed', label=r'fit')
        ax.text(0.65, 0.75, rf'$\lambda = {np.round(C[0],3)}\pm{np.round(cov[0][0], 6)}$', transform=ax.transAxes, fontsize=14)
        ax.text(0.65, 0.65, rf'$\sigma = {np.round(C[1],3)}\pm{np.round(cov[1][1], 6)}$', transform=ax.transAxes, fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        sigma_exp = 0.5 * fil**(-alpha - 2) * scipy.special.gamma(alpha/2 + 1)
        lam_exp = np.sqrt(np.abs(alpha + 3 / (2 * np.pi * fil**2)))
        print(rf'lambda_exp = {lam_exp}, lambda_mea = {C[0]}')
        print(rf'sigma_exp = {sigma_exp}, sigma_mea = {C[1]}')
        plt.show()
        # plt.savefig('plots/Q1_fit.png', dpi=300)
        # plt.close()

    elif MF == 'Q2':
        # Fitting with Q2
        def fitting_function_Q2(X, a0, a1):
            return a0**2 * X * np.exp(-X**2 / (2*a1)) / (np.sqrt(2*np.pi*a1))

        # Ignore the points where sigma is zero
        Q2_cov = np.cov(Q2_all_meas.T)
        Q2_cov[Q2_cov < 1e-8] = 0 # define a threshold, here 1e-10 
        # find the indices of the non-zero diagonal elements
        non_zero_diag = np.nonzero(np.diagonal(Q2_cov))[0]
        new_size = Q2_cov.shape[0] - len(non_zero_diag)
        Q2_cov_cut = Q2_cov[non_zero_diag][:, non_zero_diag]
        C, cov = curve_fit(fitting_function_Q2, thresh_range[non_zero_diag], Q2_measured[non_zero_diag], sigma=Q2_cov_cut, method='lm', absolute_sigma=True)
        Q2_fit = fitting_function_Q2(thresh_range, C[0], C[1])
        ax.set_ylabel(r'$\langle \hat{Q}_{1}(\nu)\rangle_{\mathrm{r}}$', fontsize=18)
        ax.plot(thresh_range, Q2_measured, c='b', lw=1.5, ls='solid', label=r'measured')
        ax.plot(thresh_range, Q2_fit, c='r', lw=1.5, ls='dashed', label=r'fit')
        ax.text(0.65, 0.35, rf'$\lambda = {np.round(C[0],3)}\pm{np.round(cov[0][0], 6)}$', transform=ax.transAxes, fontsize=14)
        ax.text(0.65, 0.25, rf'$\sigma = {np.round(C[1],3)}\pm{np.round(cov[1][1], 6)}$', transform=ax.transAxes, fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        sigma_exp = 0.5 * fil**(-alpha - 2) * scipy.special.gamma(alpha/2 + 1)
        lam_exp = np.sqrt(np.abs(alpha + 3 / (2 * np.pi * fil**2)))
        print(np.linalg.inv(Q2_cov_cut))
        print(rf'lambda_exp = {lam_exp}, lambda_mea = {C[0]}')
        print(rf'sigma_exp = {sigma_exp}, sigma_mea = {C[1]}')
        plt.show()
        # plt.savefig('plots/Q2_fit.png', dpi=300)
        # plt.close()


# # # d2_xi_0 = lambda_fit**2 * (2*np.pi*sigma_fit)
# # # print(d2_xi_0)

# # # # def fitting_function_Q2(X, a0, a1):
# # # #     return (a0 / np.sqrt(a1)) * X * np.exp(-X**2 / (2*a1))

# # # # def fitting_function_Q2(X, a0, a1):
# # # #     return a0/np.sqrt(a1) * X * np.exp(-X**2 / (2*a1))

# # # # C, cov = curve_fit(fitting_function_Q2, thresh_range, Q2_measured, sigma=Q2_err, method='lm', absolute_sigma=True)
# # # # lambda_fit_Q2, sigma_fit_Q2 = C
# # # # # fit_Q2 = (lambda_fit_Q2 / np.sqrt(sigma_fit_Q2)) * thresh_range * np.exp(-thresh_range**2 / (2*sigma_fit_Q2))
# # # # fit_Q2 = lambda_fit_Q2 / np.sqrt(sigma_fit_Q2) * thresh_range * np.exp(-thresh_range**2 / (2*sigma_fit_Q2))

# # # # print(sigma_fit, sigma_fit_Q2)

# # # Q2_est = -0.75 * np.array(Q1_measured) * thresh_range # * lambda_fit * thresh_range #/ (np.sqrt(sigma_fit))
# # ax.plot(thresh_range, Q0_measured, c='b', lw=1.5, ls='solid', label=r'measured')
# # ax.plot(thresh_range, Q1_measured, c='b', lw=1.5, ls='solid', label=r'measured')

# # ax.plot(thresh_range, fit_Q1, c='r', lw=1.5, ls='dashdot', label=r'fit')

# # c0 = 1
# # lam = 0.35
# # Q0_est = 0.5 - scipy.special.erf(thresh_range/(np.sqrt(2*c0)))/np.sqrt(np.pi)
# # Q1_est = np.pi*lam / (4*np.sqrt(2*np.pi)) * np.exp(-thresh_range**2 / (2*0.65))

# # ax.plot(thresh_range, Q0_est, c='k', lw=1.5, ls='dashed', label=r'estimated')
# # ax.plot(thresh_range, Q1_est, c='k', lw=1.5, ls='dashed', label=r'estimated')

# # plt.savefig('plots/Q0_alpha_dep.png', dpi=300)
# # plt.close()
