#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from gen_gauss_2D import gauss_2D
from minkowski import calculate_minkowski
from scipy.special import erf

# Initiate the Gaussian random field
N = 2048
# alpha = 1.4
# field = gauss_2D(N, alpha=alpha, norm=False)
# thresh = np.percentile(field, 6)

# Standard example of overlapping Boolean disks
# Their MFs are analytically known
from scipy.spatial.distance import cdist
radius = 50 
num_disks = 18
positions = np.random.uniform(low=radius, high=N-radius, size=(num_disks, 2))
distances = cdist(positions, positions)
binary = np.zeros((N, N), dtype=bool)
for i in range(num_disks):
    x, y = np.ogrid[-positions[i,0]:N-positions[i,0], -positions[i,1]:N-positions[i,1]]
    mask = x*x + y*y <= radius*radius
    binary[mask] = 1

M0, M1, M2 = calculate_minkowski(binary, threshold=None)
print(M2 * np.pi)

# M0_GKF = 0.5 - erf(thresh/np.sqrt(2))*0.5
# M1_GKF = np.exp(-thresh**2 / 2) / (3*np.pi)
# M2_GKF = np.exp(-thresh**2 / 2) * 2 * thresh / (2 * np.pi)**(3/2)
# print(M0_GKF, M0/N**2)
# print(M1_GKF, M1/N**2)
# print(M2_GKF, M2)

image = plt.imshow(binary, cmap='gray')
cbar = plt.colorbar(image)
plt.show()

# alpha_list = np.arange(0.1, 2, 0.1) #[0.2, 0.4, 0.6, 0.8, 1]    
# for alpha in alpha_list:


# M0, M1, M2 = [], [], []
# thresh_range = np.linspace(field.min(), field.max(), 100)

# for threshold in thresh_range:
#     binary = field >= threshold
#     MF = mk.functionals(np.ascontiguousarray(binary))
#     M0.append(MF[0])
#     M1.append(MF[1])
#     M2.append(MF[2])

# M0 = np.array(M0)
# M1 = np.array(M1)
# M2 = np.array(M2)

# n_interations = 100
# j = 0
# while j < n_interations:
#     field = gauss_2D(N, alpha=alpha, norm=False)
#     threshold = np.percentile(field, 50)
#     binary = field >= threshold
#     M0, M1, M2 = mk.functionals(np.ascontiguousarray(binary))
#     # M0 /= N**2
#     M1 *= (2*np.pi/(4*N))
#     # M2 *= (np.pi/N**2) 
#     # print(M0, M1, M2)
#     # print(M1)
#     M1_list.append(M1)
#     j += 1

#     # fig, ax = plt.subplots()
#     # binary_im = ax.imshow(binary)
#     # cbar = fig.colorbar(binary_im, ax=ax)
#     # plt.show()

# plt.hist(M1_list, density=True)
# plt.show()

# # M1_list = np.array(M1_list)
# # alpha_list = np.array(alpha_list)

# # from scipy.optimize import curve_fit
# # def fitting_function(alpha_list, a0, a1):
# #     return a0*(alpha_list**a1)
# # guesses = 1, 1
# # FF = curve_fit(fitting_function, (alpha_list), M1_list, guesses, sigma=1e-5*np.ones(M1_list.size), method='lm')
# # c, n = FF[0]
# # cov = FF[1]
# # fit = fitting_function(alpha_list, c, n)
# # print(c, n)


# # fig, ax = plt.subplots()
# # plt.rcParams.update({"text.usetex": True})
# # ax.minorticks_on()
# # ax.plot(alpha_list, M1_list, c='b', lw=2, label='measured')
# # ax.plot(alpha_list, fit, c='r', lw=2, ls='dashed', label='fit')

# # ax.set_xlabel(r'$\alpha$')
# # ax.set_ylabel(r'$M_{1}$')
# # plt.show()


# # plt.rcParams.update({"text.usetex": True})
# # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# # for j in range(2):
# #     ax[j].minorticks_on()
# # # ax[0].plot(thresh_range, M0/np.max(M0), c='b', lw=2, label=r'$M_{0}$')
# # # ax[0].plot(thresh_range, M1/np.max(M1), c='k', ls='dashdot', lw=2, label=r'$M_{1}$')
# # ax[0].plot(thresh_range, M2, c='r', ls='dotted', lw=2, label=r'$M_{{2}}$, max = {}'.format(M2.max()))

# # ax[0].set_xlabel('Threshold', fontsize=14)
# # ax[0].set_ylabel('Functionals', fontsize=14)
# # ax[0].legend()
# # # ax[0].axvline(thresh_range[thresh_range.size//2], ls='dashed', lw=1.5, c='seagreen')
# # field_im = ax[1].imshow(field, cmap='viridis')
# # cbar = fig.colorbar(field_im, ax=ax[1], fraction=0.046)
# # cbar.set_label(r'$\delta(\mathbf{x})$', fontsize=14)
# # ax[1].set_title(r'Overdensity, $\alpha = {}$'.format(alpha), fontsize=16)
# # ax[1].set_xlabel(r'$x/L$', fontsize=14)
# # ax[1].set_ylabel(r'$y/L$', fontsize=14)
# # plt.tight_layout()
# # plt.savefig('plots/M2_alpha_{}.png'.format(alpha), bbox_inches='tight', dpi=300)
# # plt.close()
# # plt.show()
