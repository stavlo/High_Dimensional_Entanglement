import os
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.io
import scipy.ndimage
import optimization
import plots_functions
from sum_coordination import *
import DoubleGaussian


def extract_number(filename):
    """Extract the trailing number before the extension in a filename."""
    match = re.search(r'_(\d+)\.mat$', filename)
    return int(match.group(1)) if match else float('inf')

def plot_SPAD_visibility():
    folder_path_K = 'npj'
    file_names = sorted([f for f in os.listdir(folder_path_K) if f.endswith('.mat')], key=extract_number)
    Nset = 0 # 0 = autoconv, 1 = autocorr
    DXW, DYW = 32, 64
    vecimage = np.linspace(0, DYW * DXW, DYW * DXW + 1)
    Rd = np.zeros((DXW, DYW))
    fpath = os.path.join(folder_path_K, file_names[Nset])
    cov_mat = scipy.io.loadmat(fpath)
    I4D_K = cov_mat['I4D_final']
    I4D_K_opt, loss_list = optimization.optimize_x(I4D_K, 1e-5, 0.0,learning_rate=10 ** -4.5 ,max_iter=50, upper_triangular=False, SPAD=True) # fig 2

    autoconv_opt = convolution_reader(I4D_K_opt, Rd, vecimage)
    # autoconv_opt = correlation_reader(I4D_K_opt, Rd)
    autoconv = convolution_reader(I4D_K, Rd, vecimage)
    # autoconv = correlation_reader(I4D_K, Rd)

    half_box = 6 // 2

    center_y, center_x = np.unravel_index(autoconv.argmax(), autoconv.shape)
    y1 = center_y - half_box
    y2 = center_y + half_box
    x1 = center_x - half_box
    x2 = center_x + half_box
    mask = np.ones_like(autoconv, dtype=bool)
    mask[y1:y2, x1:x2] = False
    SNR = np.max(autoconv[~mask]) / np.std(autoconv[mask])
    V = np.max(autoconv[~mask]) / np.mean(autoconv[mask])
    SNR_l1 = np.max(autoconv_opt[~mask]) / np.std(autoconv_opt[mask])
    V_l1 = np.max(autoconv_opt[~mask]) / np.mean(autoconv_opt[mask])

    epsilon = 1e-10
    plt.figure()
    plt.subplot(1,2,1)
    # plt.imshow(autoconv_opt, cmap='hot')
    plt.imshow(np.log(np.abs(autoconv_opt + epsilon)), cmap='hot', vmin=-7, vmax=1)
    plt.colorbar()
    plt.title(f'L1 optimize SNR {SNR_l1:.1} & Visibility {V:.1}')
    plt.subplot(1,2,2)
    # plt.imshow(autoconv, cmap='hot')
    plt.imshow(np.log(np.abs(autoconv + epsilon)), cmap='hot', vmin=-7, vmax=1)
    plt.colorbar()
    plt.title(f'Original SNR {SNR:.1} & Visibility {V_l1:.1}')

    plt.suptitle("Minus Coordinations Heatmaps Log Scale", fontsize=14)

    plt.show(block=True)

def plot_EPR_SPAD():
    folder_path_K = 'npj'
    file_names = sorted([f for f in os.listdir(folder_path_K) if f.endswith('.mat')], key=extract_number)
    DXW, DYW = 32, 64
    vecimage = np.linspace(0, DYW * DXW, DYW * DXW + 1)
    Rd = np.zeros((DXW, DYW))
    Kpath = os.path.join(folder_path_K, file_names[0])
    Ppath = os.path.join(folder_path_K, file_names[1])
    cov_mat_K = scipy.io.loadmat(Kpath)
    cov_mat_P = scipy.io.loadmat(Ppath)
    I4D_K = cov_mat_K['I4D_final']
    I4D_K_opt, loss_list = optimization.optimize_x(I4D_K, 1e-5, 0.0, learning_rate=10 ** -4.5, max_iter=50,
                                                   SPAD=True)  # fig 2
    I4D_P = cov_mat_P['I4D_final']
    for i in range(I4D_P.shape[0]):
        if i == 0:
            I4D_P[i,i] = I4D_P[i,i + 1]
        elif i == I4D_P.shape[0] - 1:
            I4D_P[i,i] = I4D_P[i,i - 1]
        else:
            I4D_P[i,i] = 0.5*(I4D_P[i,i - 1] + I4D_P[i,i + 1])
    I4D_P_opt, loss_list = optimization.optimize_x(I4D_P, 1e-5, 0.0, learning_rate=10 ** -4.5, max_iter=50,
                                                   SPAD=True)  # fig 2

    autoconv_opt = convolution_reader(I4D_K_opt, Rd, vecimage)
    autocorr_opt = correlation_reader(I4D_P_opt, Rd)
    autoconv = convolution_reader(I4D_K, Rd, vecimage)
    autocorr = correlation_reader(I4D_P, Rd)
    window_size = np.array(autoconv.shape).max() + 1
    K_avg_sigma, _, _, _, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autoconv, window_size=window_size, show=False, SPAD=True)
    P_avg_sigma, _, _, _, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autocorr, window_size=window_size, show=False, SPAD=True)
    K_avg_sigma_l1, _, _, _, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autoconv_opt, window_size=window_size, show=False, SPAD=True)
    P_avg_sigma_l1, _, _, _, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autocorr_opt, window_size=window_size, show=False, SPAD=True)


    sigma_pos_m, sigma_mom_rad_per_m = plots_functions.convert_pixel_units(P_avg_sigma, K_avg_sigma,
                                                                          pixel_size_m=150e-6, wavelength_m=694e-9,
                                                                          focal_length_m=200e-3, M=(100/35, 300/35))
    sigma_pos_m_l1, sigma_mom_rad_per_m_l1 = plots_functions.convert_pixel_units(P_avg_sigma_l1, K_avg_sigma_l1,
                                                                                pixel_size_m=150e-6, wavelength_m=694e-9,
                                                                                focal_length_m=200e-3, M=(100/35, 300/35))

    # Heisenberg EPR product (unitless, ~ hbar = 1)
    epr_product, _ = plots_functions.epr_calc(sigma_pos_m, sigma_mom_rad_per_m, 0, 0)
    epr_product_l1, _ = plots_functions.epr_calc(sigma_pos_m, sigma_mom_rad_per_m, 0, 0)

    d, _ = plots_functions.dim_calc(sigma_pos_m, sigma_mom_rad_per_m, 0, 0)
    d_l1, _ = plots_functions.dim_calc(sigma_pos_m_l1, sigma_mom_rad_per_m_l1, 0, 0)

    print(f'EPR = {epr_product} with dimensional witness {d}')
    print(f'EPR_l1 = {epr_product_l1} with dimensional witness {d_l1}')
    plt.show()

    return

def spectral_decomposition(cov):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
    # Get the indices that would sort the eigenvalues in descending order
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    # Sort eigenvalues and eigenvectors accordingly
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvectors, sorted_eigenvalues

def _plot_figure_for_matrix(M_raw, M_opt, fig_title):
    """
    Make a 2x2 figure:
      top row:   top-2 eigenvectors of raw matrix (reshaped)
      bottom row:top-2 eigenvectors of optimized matrix (reshaped)
    """
    # Decompose raw & optimized
    V_raw, E_raw = spectral_decomposition(M_raw)
    V_opt, E_opt = spectral_decomposition(M_opt)

    # Take top-2
    S, E = 0, 2
    E_raw2, V_raw2 = E_raw.T[S:E], V_raw.T[S:E]
    E_opt2, V_opt2 =  E_opt.T[S:E], V_opt.T[S:E]

    # Reshape eigenvectors to images
    imgs = [
        np.reshape(np.outer(V_raw2[0, :], V_raw2[0, :]) * E_raw2[0], (64,32,64,32)).T.sum(axis=(0, 1)),
        np.reshape(np.outer(V_raw2[1, :], V_raw2[1, :]) * E_raw2[1], (64,32,64,32)).T.sum(axis=(0, 1)),
        np.reshape(np.outer(V_opt2[0, :], V_opt2[0, :]) * E_opt2[0], (64,32,64,32)).T.sum(axis=(0, 1)),
        np.reshape(np.outer(V_opt2[1, :], V_opt2[1, :]) * E_opt2[1], (64,32,64,32)).T.sum(axis=(0, 1))
    ]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    titles = [
        rf"Raw: $v_1$   ($\lambda_1$={np.abs(E_raw2[0]):.3g})",
        rf"Raw: $v_2$   ($\lambda_2$={np.abs(E_raw2[1]):.3g})",
        rf"Optimized: $v_1$   ($\lambda_1$={np.abs(E_opt2[0]):.3g})",
        rf"Optimized: $v_2$   ($\lambda_2$={np.abs(E_opt2[1]):.3g})",
    ]

    for ax, img, t in zip(axes.ravel(), imgs, titles):
        im = ax.imshow(img, cmap='hot')
        ax.set_title(t, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(fig_title, fontsize=13)
    plt.show(block=True)

def plot_eigenvalues_SPAD():
    folder_path_K = 'npj'
    file_names = sorted(
        [f for f in os.listdir(folder_path_K) if f.endswith('.mat')],
        key=extract_number
    )
    if len(file_names) < 2:
        raise FileNotFoundError("Expected at least two .mat files (K then P) in 'npj'.")

    Kpath = os.path.join(folder_path_K, file_names[0])
    Ppath = os.path.join(folder_path_K, file_names[1])

    # Load
    cov_mat_K = scipy.io.loadmat(Kpath)
    cov_mat_P = scipy.io.loadmat(Ppath)
    I4D_K = cov_mat_K['I4D_final']
    I4D_P = cov_mat_P['I4D_final']

    # Optimize (l1 off if_log=False as in your snippet)
    I4D_K_opt, _ = optimization.optimize_x(
        I4D_K, 1e-5, 0.0, learning_rate=10 ** -4.5, max_iter=150, SPAD=True
    )
    I4D_P_opt, _ = optimization.optimize_x(
        I4D_P, 1e-5, 0.0, learning_rate=10 ** -4.5, max_iter=150, SPAD=True
    )

    # Figure for K
    _plot_figure_for_matrix(I4D_K, I4D_K_opt, "K: Top-2 Eigenvectors (Raw vs Optimized)")

    # Figure for P
    _plot_figure_for_matrix(I4D_P, I4D_P_opt, "P: Top-2 Eigenvectors (Raw vs Optimized)")


if __name__ == "__main__":
    plot_SPAD_visibility()
    plot_EPR_SPAD()
    plot_eigenvalues_SPAD()
