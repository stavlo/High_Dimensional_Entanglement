import os
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.io
import scipy.ndimage
import optimization
import utilities
from sum_coordination import *
import DoubleGaussian


def extract_number(filename):
    """Extract the trailing number before the extension in a filename."""
    match = re.search(r'_(\d+)\.mat$', filename)
    return int(match.group(1)) if match else float('inf')

def plot_fig3(lambda_power, ref_xy=(16,25), save_path="SPAD figures/Fig_3.pdf", epsilon=1e-10):

    folder_path_K = 'SPAD figures/npj'
    file_names = sorted([f for f in os.listdir(folder_path_K) if f.endswith('.mat')], key=extract_number)
    Nset = 0  # 0 = autocorr, 1 = autocorr
    fpath = os.path.join(folder_path_K, file_names[Nset])
    cov_mat = scipy.io.loadmat(fpath)
    cov = cov_mat['I4D_final']
    cov_opt, loss_list = optimization.optimize_x(cov, 10 ** (lambda_power), 0.0, learning_rate=10 ** -4,
                                                   max_iter=1000,
                                                   upper_triangular=True, SPAD=True)  # fig 2

    cov = np.reshape(np.asarray(cov, dtype=float), (64,32,64,32)).T
    cov_opt = np.reshape(np.asarray(cov_opt, dtype=float), (64,32,64,32)).T

    ref_x, ref_y = ref_xy

    # Build panels
    raw_proj = np.sum(cov, axis=(0,1))
    opt_proj = np.sum(cov_opt, axis=(0,1))
    raw_cond = cov[ref_x, ref_y]
    opt_cond = cov_opt[ref_x, ref_y]

    panels = [raw_proj, opt_proj, raw_cond, opt_cond]

    # Shared logarithmic normalization
    panels_log = [np.log10(np.abs(P) + epsilon) for P in panels]

    vmin = -5 #min(np.nanmin(P) for P in panels_log)
    vmax = 0 #max(np.nanmax(P) for P in panels_log)

    # Figure style
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(6.6, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 2])

    axes = [ax_a, ax_b, ax_c, ax_d]
    labels = ["a", "b", "c", "d"]

    last_im = None
    panel_fs = 16

    for ax, panel, label in zip(axes, panels_log, labels):
        last_im = ax.imshow(panel, origin="lower", cmap="hot", vmin=vmin, vmax=vmax)

        ax.plot(
            ref_y,
            ref_x,
            marker="+",
            markersize=10,
            markeredgewidth=2.0,
            color="cyan",
            linestyle="None",
        )
        if label in ['a', 'c']:
            ax.set_ylabel(r"$k_y$ [pixels]")
        if label in ['d', 'c']:
            ax.set_xlabel(r"$k_x$ [pixels]")

        ax.text(
            0.04,
            0.94,
            label,
            transform=ax.transAxes,
            fontsize=panel_fs,
            fontweight="bold",
            color="white",
            va="top",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5),
        )

    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label(r"Coincidence [arb. units]", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")

    plt.show()

    return fig, axes

def plot_SPAD_SNR(lambda_power):
    folder_path_K = 'SPAD figures/npj'
    file_names = sorted([f for f in os.listdir(folder_path_K) if f.endswith('.mat')], key=extract_number)
    Nset = 0 # 0 = autocorr, 1 = autocorr
    DXW, DYW = 32, 64
    vecimage = np.linspace(0, DYW * DXW, DYW * DXW + 1)
    Rd = np.zeros((DXW, DYW))
    fpath = os.path.join(folder_path_K, file_names[Nset])
    cov_mat = scipy.io.loadmat(fpath)
    I4D_K = cov_mat['I4D_final']
    I4D_K_opt, loss_list = optimization.optimize_x(I4D_K, 10**(lambda_power), 0.0, learning_rate=10 ** -4, max_iter=1000,
                                                   upper_triangular=True ,SPAD=True)  # fig 2
    if Nset == 1:
        autoconv_opt = correlation_reader(I4D_K_opt, Rd)
        autoconv = correlation_reader(I4D_K, Rd)
    else:
        autoconv_opt = convolution_reader(I4D_K_opt, Rd, vecimage)
        autoconv = convolution_reader(I4D_K, Rd, vecimage)

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

def plot_EPR_SPAD(lambda_power, learning_rate=10 ** -4):
    folder_path_K = 'SPAD figures/npj'
    file_names = sorted([f for f in os.listdir(folder_path_K) if f.endswith('.mat')], key=extract_number)
    DXW, DYW = 32, 64
    vecimage = np.linspace(0, DYW * DXW, DYW * DXW + 1)
    Rd = np.zeros((DXW, DYW))
    Kpath = os.path.join(folder_path_K, file_names[0])
    Ppath = os.path.join(folder_path_K, file_names[1])
    cov_mat_K = scipy.io.loadmat(Kpath)
    cov_mat_P = scipy.io.loadmat(Ppath)
    I4D_K = cov_mat_K['I4D_final']
    I4D_K_opt, loss_K = optimization.optimize_x(I4D_K, 10**(lambda_power), 0.0, learning_rate=learning_rate, max_iter=1000,
                                                   upper_triangular=True ,SPAD=True)  # fig 2
    I4D_P = cov_mat_P['I4D_final']
    for i in range(I4D_P.shape[0]):
        if i == 0:
            I4D_P[i,i] = I4D_P[i,i + 1]
        elif i == I4D_P.shape[0] - 1:
            I4D_P[i,i] = I4D_P[i,i - 1]
        else:
            I4D_P[i,i] = 0.5*(I4D_P[i,i - 1] + I4D_P[i,i + 1])
    I4D_P_opt, loss_P  = optimization.optimize_x(I4D_P, 10**(lambda_power), 0.0, learning_rate=learning_rate, max_iter=1000,
                                                   upper_triangular=True ,SPAD=True)  # fig 2

    autoconv_opt = convolution_reader(I4D_K_opt, Rd, vecimage)
    autocorr_opt = correlation_reader(I4D_P_opt, Rd)
    autoconv = convolution_reader(I4D_K, Rd, vecimage)
    autocorr = correlation_reader(I4D_P, Rd)
    window_size = np.array(autoconv.shape).max() + 1
    K_avg_sigma, _, _, _, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autoconv, window_size=window_size, show=False, SPAD=True)
    P_avg_sigma, _, _, _, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autocorr, window_size=window_size, show=False, SPAD=True)
    K_avg_sigma_l1, _, _, _, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autoconv_opt, window_size=window_size, show=False, SPAD=True)
    P_avg_sigma_l1, _, _, _, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autocorr_opt, window_size=window_size, show=False, SPAD=True)

    sigma_pos_m, sigma_mom_rad_per_m = utilities.convert_pixel_units(P_avg_sigma, K_avg_sigma,
                                                                          pixel_size_m=150e-6, wavelength_m=694e-9,
                                                                          focal_length_m=200e-3, M=(100/35, 300/35))
    sigma_pos_m_l1, sigma_mom_rad_per_m_l1 = utilities.convert_pixel_units(P_avg_sigma_l1, K_avg_sigma_l1,
                                                                                pixel_size_m=150e-6, wavelength_m=694e-9,
                                                                                focal_length_m=200e-3, M=(100/35, 300/35))

    # Heisenberg EPR product (unitless, ~ hbar = 1)
    epr_product, _ = utilities.epr_calc(sigma_pos_m, sigma_mom_rad_per_m, 0, 0)
    epr_product_l1, _ = utilities.epr_calc(sigma_pos_m_l1, sigma_mom_rad_per_m_l1, 0, 0)

    d, _ = utilities.dim_calc(sigma_pos_m, sigma_mom_rad_per_m, 0, 0)
    d_l1, _ = utilities.dim_calc(sigma_pos_m_l1, sigma_mom_rad_per_m_l1, 0, 0)

    print(f'EPR = {epr_product} with dimensional witness {d}')
    print(f'EPR_l1 = {epr_product_l1} with dimensional witness {d_l1}')
    plt.show()

    return
    # return {
    #     "lambda_power": lambda_power,
    #     "learning_rate": learning_rate,
    #
    #     "K_sigma_pix_raw": K_avg_sigma,
    #     "P_sigma_pix_raw": P_avg_sigma,
    #     "K_sigma_pix_l1": K_avg_sigma_l1,
    #     "P_sigma_pix_l1": P_avg_sigma_l1,
    #
    #     "sigma_pos_m_raw": sigma_pos_m,
    #     "sigma_mom_rad_per_m_raw": sigma_mom_rad_per_m,
    #     "sigma_pos_m_l1": sigma_pos_m_l1,
    #     "sigma_mom_rad_per_m_l1": sigma_mom_rad_per_m_l1,
    #
    #     "epr_product_raw": epr_product,
    #     "epr_product_l1": epr_product_l1,
    #
    #     "dim_witness_raw": d,
    #     "dim_witness_l1": d_l1,
    #
    #     "loss_K": loss_K,
    #     "loss_P": loss_P,
    # }

def spectral_decomposition(cov):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
    # Get the indices that would sort the eigenvalues in descending order
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    # Sort eigenvalues and eigenvectors accordingly
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvectors, sorted_eigenvalues

def plot_fig4(
    lambda_power,
    save_path="SPAD figures/Fig_4.pdf",
    use_abs=False,
):
    folder_path_K = 'SPAD figures/npj'
    file_names = sorted(
        [f for f in os.listdir(folder_path_K) if f.endswith('.mat')],
        key=extract_number
    )
    if len(file_names) < 2:
        raise FileNotFoundError("Expected at least two .mat files (K then P) in 'npj'.")

    Kpath = os.path.join(folder_path_K, file_names[0])
    cov_mat_K = scipy.io.loadmat(Kpath)
    I4D_K = cov_mat_K['I4D_final']

    I4D_K_opt, loss_list = optimization.optimize_x(I4D_K, 10**(lambda_power), 0.0, learning_rate=10 ** -4, max_iter=1000,
                                                   upper_triangular=True ,SPAD=True)


    # Decompose raw & optimized
    V_raw, E_raw = spectral_decomposition(I4D_K)
    V_opt, E_opt = spectral_decomposition(I4D_K_opt)

    # Take top-2
    S, E = 0, 2
    E_raw2, V_raw2 = E_raw.T[S:E], V_raw.T[S:E]
    E_opt2, V_opt2 = E_opt.T[S:E], V_opt.T[S:E]

    # Reshape eigenvectors to images
    imgs = [
        np.reshape(np.outer(V_raw2[0, :], V_raw2[0, :]) * E_raw2[0], (64, 32, 64, 32)).T.sum(axis=(0, 1)),
        np.reshape(np.outer(V_raw2[1, :], V_raw2[1, :]) * E_raw2[1], (64, 32, 64, 32)).T.sum(axis=(0, 1)),
        np.reshape(np.outer(V_opt2[0, :], V_opt2[0, :]) * E_opt2[0], (64, 32, 64, 32)).T.sum(axis=(0, 1)),
        np.reshape(np.outer(V_opt2[1, :], V_opt2[1, :]) * E_opt2[1], (64, 32, 64, 32)).T.sum(axis=(0, 1))
    ]

    if use_abs:
        imgs = [np.abs(im) for im in imgs]
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))

    axes = axes.ravel()

    labels = ["a", "b", "c", "d"]
    cmap = "hot"
    panel_fs = 16

    for ax, im, label in zip(axes, imgs, labels):

        im_obj = ax.imshow(
            im,
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
            aspect="auto",
        )
        if label in ['a', 'c']:
            ax.set_ylabel(r"$k_y$ [pixels]")
        if label in ['d', 'c']:
            ax.set_xlabel(r"$k_x$ [pixels]")

        # Panel label outside top-left
        ax.text(
            -0.18, 1.08, label,
            transform=ax.transAxes,
            fontsize=panel_fs,
            fontweight="bold",
            va="top",
            ha="left",
            clip_on=False,
        )

        # Individual colorbar for each image
        cbar = fig.colorbar(
            im_obj,
            ax=ax,
            fraction=0.046,
            pad=0.035,
        )
        cbar.set_label("Eigenmode amp. [arb. units]", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(
        left=0.10,
        right=0.96,
        bottom=0.12,
        top=0.90,
        wspace=0.45,
        hspace=0.45,
    )

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.show()

    return fig, axes

if __name__ == "__main__":
    lambda_power = -9.4
    plot_fig3(lambda_power)
    plot_fig4(lambda_power)
    # plot_SPAD_SNR(lambda_power)
    plot_EPR_SPAD(lambda_power)