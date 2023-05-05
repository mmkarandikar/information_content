import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gen_gauss_2D import gauss_2D
from minkowski import calculate_minkowski

def generate_data(thresh_range: np.ndarray, alpha_true: float = -1, l_box: float = 2*np.pi,
                     field_size: int = 64, n_real: int = 200) -> np.ndarray:

    seeds = np.random.randint(1, 2**32-1, n_real)
    n_thresh = thresh_range.shape[0]

    # Q1[i,j] is the Q1 for the ith realisation and jth threshold value
    Q1 = np.empty(shape=(n_real, n_thresh))
    sigma_field = []
    for i in tqdm(range(n_real)):
        field = gauss_2D(size=field_size, l_box=l_box,
                    alpha=alpha_true, seed=seeds[i], norm=True)
        sigma_field.append(np.std(field))
        Q1[i,:] = [calculate_minkowski(field=field, threshold=thresh_range[j], norm=True)[2]
                    for j in range(n_thresh)]

    # Center the data by subtracting the mean from each column
    Q1_centered = np.subtract(Q1, np.mean(Q1, axis=0))

    # Ignore the points where sigma is zero
    Q1_cov = np.cov(Q1_centered.T, ddof=1)

    Q1_cov[Q1_cov < 1e-10] = 0 # define a threshold for zeroness 
    # find the indices of the non-zero diagonal elements
    non_zero_diag = np.nonzero(np.diagonal(Q1_cov))[0]
    Q1_cov = Q1_cov[non_zero_diag][:, non_zero_diag]
    return Q1[:, non_zero_diag], Q1_cov, non_zero_diag, sigma_field, field

l_box, prior_size, field_size, n_real, n_thresh = 100, 100, 128, 1000, 100
# something to do with the field normalization is throwing the model off
thresh = np.linspace(-4, 4, n_thresh)
prior = np.linspace(0, 2, prior_size)
alpha_true = 0.75

# # Generate and save data
# Q1_data, Q1_cov, indices, sigmas, field = generate_data(thresh, alpha_true, l_box=l_box,
#                              field_size=field_size, n_real=n_real)

# df = np.array([Q1_data, Q1_cov, indices, sigmas, field], dtype=object)
# file = open('Q1_data.npy', 'wb')
# np.save(file, df)
# file.close()

# Read data
file = open('Q1_data.npy', 'rb')
Q1_data, Q1_cov, indices, sigmas, field = np.load(file, allow_pickle=True)
file.close()

# helper function that calculates the lambda for a given alpha
def lamb_mik(alpha, k_min, k_max):
    xi_dd = np.abs((np.power(k_max,alpha + 4) - np.power(k_min,alpha + 4))/(alpha + 4))
    xi = (np.power(k_max,alpha + 2) - np.power(k_min,alpha + 2))/(alpha + 2)
    return np.sqrt(xi_dd/(2*np.pi*xi))

k_min = 2*np.pi / l_box
k_max = (field_size//2 - 1) * k_min
lambda_true = lamb_mik(alpha_true, k_min, k_max)

# def model(thresh: np.ndarray, lam: float = 1, sigma: float = 1) -> np.ndarray:
#     # return (np.sqrt(np.pi)*lam/(4*np.sqrt(2))) * np.exp(-thresh**2 / 2)
#     return lam * np.exp(-thresh**2 / 2)


def model(thresh: np.ndarray, lam: float = 1, sigma: float = 1) -> np.ndarray:
    # return (np.sqrt(np.pi)*lam/(4*np.sqrt(2))) * np.exp(-thresh**2 / 2)
    return lam**2 * thresh * np.exp(-thresh**2 / 2) / (np.sqrt(2*np.pi))


def get_lamb_dist(data_vec, data_cov, thresh_range, model, prior, indices, ind_data):
    def compute_likelihood(data, model, cov, n_thresh):
        correction= 1 - ((data_vec.shape[1] + 1)/(data_vec.shape[0] - 1))
        cov_det = np.linalg.det(cov)
        try:
            precision = correction * np.linalg.inv(cov)
        except:
            print("Singular matrix", cov_det)

        loglikeli = - (data-model).dot(precision.dot(data-model)) / 2
        return loglikeli

    log_lamb_posterior = [compute_likelihood(data_vec[ind_data,:], model(thresh_range, prior[j])[indices], data_cov,
                         thresh_range.size) for j in tqdm(range(prior.size))]
    lamb_posterior = np.exp(np.array(log_lamb_posterior))

    return lamb_posterior / np.sum(lamb_posterior)


ind_data = 0
# posterior = get_lamb_dist(Q1_data, Q1_cov, thresh, model, prior, indices, ind_data)

k = np.linspace(k_min, k_max, field_size)
xi = np.var(field)
print(xi)
xi_dd = np.var(np.real(np.fft.ifft(np.fft.fft(field) * (k))))

lambda_guess = np.sqrt(np.abs(xi_dd) / (2*np.pi*xi))

plt.rcParams.update({"text.usetex": True})
fig, ax = plt.subplots(nrows=2)
# ax.minorticks_on()

# ax.set_xlabel(r'$\lambda$', fontsize=18)
# ax.set_ylabel(r'$p(\lambda|\theta)$', fontsize=18)
# ax.tick_params(axis='both', which='both', direction='in', labelsize=16)
# ax.plot(prior, posterior, c='b', lw=1.5)
# ax.axvline(lambda_true, c='k', ls='dashed', lw=1)
# # plt.legend(fontsize=14)

# # mean = np.mean(Q1_data, axis=0)
# # std = (np.std(Q1_data, axis=0))
# # ax.errorbar(thresh[indices], mean, yerr=std, lw=1.5, color='k')
# # ax.plot(thresh, data, lw=1.5, color='b', ls='dashdot')

ax[0].plot(thresh[indices], Q1_data[0,:], lw=1.5, color='b', ls='dashdot', label='Data')
ax[0].plot(thresh, model(thresh, lambda_guess), lw=1.5, color='k', ls='dashed', label=rf'$\lambda_{{\mathrm{{guess}}}} = {lambda_guess}$')
ax[0].plot(thresh, model(thresh, lambda_true), lw=1.5, color='r', label=rf'$\lambda_{{\mathrm{{model}}}} = {np.round(lambda_true)}$')
# ax.set_xlabel(r'$\nu$', fontsize=18)
# ax.set_ylabel(r'$Q_{1}$', fontsize=18)

ax[1].plot(thresh[indices], model(thresh[indices], lambda_guess)/ Q1_data[0,:] , lw=1.5, c='k', ls='dashed')
ax[1].plot(thresh[indices], model(thresh[indices], lambda_true)/ Q1_data[0,:] , lw=1.5, c='r')


ratio = model(thresh[indices], lambda_guess)/ Q1_data[0,:]

print(np.mean(ratio[ratio.size//4:ratio.size//2]))

# ax[1].set_ylim(0, 20)
# ax[0].set_ylim(0, 0.4)

plt.legend()
plt.savefig('ratio_plus.png')
# plt.show()

# print(Q1_data)
# print(Q1_cov)
# Q1_corr = np.corrcoef(Q1_cov, rowvar=False)
# plt.rcParams.update({"text.usetex": True})
# fig, ax = plt.subplots()
# ax.set_title('Data Covariance', fontsize=14)
# ax.minorticks_on()
# ax.tick_params(axis='both', which='both', direction='out', labelsize=12)
# im = plt.imshow(Q1_corr)
# # im = plt.imshow(Q1_cov)

# plt.colorbar(im)
# plt.show()