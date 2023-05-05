import numpy as np

# # Define the limits of the wavevector k
# size = 64
# l_box = 100
# k_min = 2*np.pi / l_box
# k_max = (size//2 - 1) * k_min
# k = np.linspace(k_min, k_max, num=size)
# A = 

def calculate_l(n):
    # Define the power-law index and the number of grid points
    alpha = n
    l_box = 100
    N = 256 # change to desired grid size
    k_min = 2*np.pi / l_box
    k_max = (N//2 - 1) * k_min

    # Define the k-space grid
    k = np.linspace(k_min, k_max, N)
    kx, ky = np.meshgrid(k, k)
    
    # Define the power spectrum and the correlation function
    P = (N**6 / l_box**3) * (kx**(alpha) + ky**(alpha))
    xi = np.real(np.fft.ifft2(P))
    xi2 = np.real(np.fft.ifft2(-P * (kx**2 + ky**2)))
    
    # Calculate l
    xi0 = np.abs(xi[N//2,N//2])
    xi2_0 = np.abs(xi2[N//2,N//2])
    l = np.sqrt(xi2_0 / (2*np.pi*xi0))
    
    return l, xi0, xi2_0

print(calculate_l(-0.5))