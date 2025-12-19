import matplotlib.pyplot as plt
from I4D_Edit import *
import re
import pickle
from utilities import dataset_creation, epr_calc, dim_calc, convert_pixel_units

def plot_sum_coordination(folder_path, file_name, load_if_exists=True):
    save_path = os.path.join(folder_path, file_name)
    if load_if_exists:
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                total = pickle.load(f)
            print(f"Loaded cached results from: {save_path}")
            cov_array = np.array([cov for _, cov, _, _, _, _, _ in total])
            cov_array_l1 = np.array([cov_l1 for _, _, _, _, cov_l1, _, _ in total])
            names = np.array([fname for fname, _, _, _, _, _, _ in total])
            num_img = np.array([int(re.search(r"(\d+)", name).group(1)) for name in names]) * 1718

    fig, axes = plt.subplots(2, 8, figsize=(10, 8), constrained_layout=True)
    cnt = 0
    start = 0
    jump = 1
    num_fig = axes.shape[1] * 2
    last_im = None  # <-- store the last image handle for shared colorbar

    for ax0, ax1, autoconv, autoconv_l1, title  in zip(axes[0], axes[1],
                                                      cov_array[start::jump], cov_array_l1[start::jump], num_img[start::jump]):

        autoconv_l1 = autoconv_l1#[20:-20,20:-20]
        autoconv = autoconv#[20:-20,20:-20]

        im = ax0.imshow(autoconv, cmap='hot')
        # im = ax0.imshow(autoconv, cmap='jet', vmin=np.min(cov_array[6:]), vmax=np.max(cov_array[6:]))
        # im = ax0.imshow(np.log10(autoconv + np.abs(np.min(autoconv)) + 1e-9), cmap='hot', vmin=4.5, vmax=6.5)
        # fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
        # ax0.set_title(f'N = {title}, Sigma = {sigma:.3}', fontsize=12)
        ax0.set_title(f'N = {title}', fontsize=12)
        ax0.set_xticks([])
        ax0.set_yticks([])

        im = ax1.imshow(autoconv_l1, cmap='hot')
        # im = ax1.imshow(autoconv_l1, cmap='jet', vmin=np.min(cov_array[6:]), vmax=np.max(cov_array[6:]))
        # im = ax1.imshow(np.log10(autoconv_l1 + np.abs(np.min(autoconv_l1)) + 1e-9), cmap='hot', vmin=4.5, vmax=6.5)
        # fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        # ax1.set_title(f'N = {title}, Sigma = {sigma_l1:.3}', fontsize=12)
        # ax1.set_title(f'N = {title}', fontsize=12)
        ax1.set_xticks([])
        ax1.set_yticks([])

        last_im = im  # <-- keep handle to the last image
        cnt += 1
        if cnt == num_fig:
            break

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), location='right', shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Sum Coordinations Heatmaps", fontsize=14)
    plt.show(block=True)

def plot_EPR_from_folder(folder_path_K, folder_path_P, file_name_K, file_name_P):
    sigma_array_K, sigma_array_K_err, l1_sigma_array_K, sigma_l1_array_K_err, names = dataset_creation(folder_path_K, file_name_K, load_if_exists=True)
    sigma_array_P, sigma_array_P_err, l1_sigma_array_P, sigma_l1_array_P_err, names = dataset_creation(folder_path_P, file_name_P, load_if_exists=True)
    sigma_array_P_W, sigma_array_P_W_err, _, _, names = dataset_creation(folder_path_P, file_name_P, load_if_exists=True, window=100)
    sigma_array_K_W, sigma_array_K_W_err, _, _, names = dataset_creation(folder_path_K, file_name_K, load_if_exists=True, window=100)

    plt.figure()
    plt.subplot(1,2,1)
    plt.errorbar(names, sigma_array_K_W, yerr=sigma_array_K_W_err, fmt='o', label='Raw Data window', capsize=4)
    plt.errorbar(names, sigma_array_K, yerr=sigma_array_K_err, fmt='o', label='Raw Data', capsize=4)
    plt.errorbar(names, l1_sigma_array_K, yerr=sigma_l1_array_K_err, fmt='o', label='L1-Regularized', capsize=4)
    plt.xlabel('# Acquired Frames')
    plt.ylabel(r'$\sigma_{avg}$')
    # plt.title('Gaussian Fit Width with Error Bars')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.subplot(1,2,2)
    plt.errorbar(names, sigma_array_P_W, yerr=sigma_array_P_W_err, fmt='o', label='Raw Data window', capsize=4)
    plt.errorbar(names, sigma_array_P, yerr=sigma_array_P_err, fmt='o', label='Raw Data', capsize=4)
    plt.errorbar(names, l1_sigma_array_P, yerr=sigma_l1_array_P_err, fmt='o', label='L1-Regularized', capsize=4)
    plt.xlabel('# Acquired Frames')
    plt.ylabel(r'$\sigma_{avg}$')
    # plt.title('Gaussian Fit Width with Error Bars')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-1.0, 2.0)

    plt.show(block=False)

    sigma_pos_m, sigma_mom_rad_per_m = convert_pixel_units(sigma_array_P, sigma_array_K[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)
    sigma_pos_m_err, sigma_mom_rad_per_m_err = convert_pixel_units(sigma_array_P_err, sigma_array_K_err[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)
    sigma_pos_m_W, sigma_mom_rad_per_m_W = convert_pixel_units(sigma_array_P_W, sigma_array_K_W[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)
    sigma_pos_m_W_err, sigma_mom_rad_per_m_W_err = convert_pixel_units(sigma_array_P_W_err, sigma_array_K_W_err[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)
    sigma_pos_m_l1, sigma_mom_rad_per_m_l1 = convert_pixel_units(l1_sigma_array_P, l1_sigma_array_K[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)
    sigma_pos_m_l1_err, sigma_mom_rad_per_m_l1_err = convert_pixel_units(sigma_l1_array_P_err, sigma_l1_array_K_err[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)

    epr_product_W, epr_product_W_err= epr_calc(sigma_pos_m_W, sigma_mom_rad_per_m_W, sigma_pos_m_W_err, sigma_mom_rad_per_m_W_err)
    epr_product, epr_product_err= epr_calc(sigma_pos_m, sigma_mom_rad_per_m, sigma_pos_m_err, sigma_mom_rad_per_m_err)
    epr_product_l1, epr_product_l1_err= epr_calc(sigma_pos_m_l1, sigma_mom_rad_per_m_l1, sigma_pos_m_l1_err, sigma_mom_rad_per_m_l1_err)

    d_W, d_W_err = dim_calc(sigma_pos_m_W, sigma_mom_rad_per_m_W, sigma_pos_m_W_err, sigma_mom_rad_per_m_W_err)
    d, d_err = dim_calc(sigma_pos_m, sigma_mom_rad_per_m, sigma_pos_m_err, sigma_mom_rad_per_m_err)
    d_l1, d_l1_err = dim_calc(sigma_pos_m_l1, sigma_mom_rad_per_m_l1, sigma_pos_m_l1_err, sigma_mom_rad_per_m_l1_err)

    # x_values = np.array([int(name.split('_')[-1].split('.')[0]) * 1718 for name in names])[:]
    x_values = names
    start = 0
    plt.figure(figsize=(10, 5))
    plt.errorbar(x_values[start:], epr_product_W[start:], yerr=epr_product_W_err[start:], fmt='o', label='Raw data window', capsize=4)
    plt.errorbar(x_values[start:], epr_product[start:], yerr=epr_product_err[start:], fmt='o', label='Raw data', capsize=4)
    plt.errorbar(x_values[start:], epr_product_l1[start:], yerr=epr_product_l1_err[start:], fmt='s', label='L1 Regularization', capsize=4)
    plt.axhline(0.5, color='red', linestyle='--', label='Heisenberg Limit (0.5)')
    plt.xlabel("# Acquired Frames", fontsize=14)
    plt.ylabel("EPR Product (Δx·Δk)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-1.0, 10.0)
    plt.show()

    # Plot dimensionality d with error bars
    plt.figure(figsize=(10, 5))
    plt.errorbar(x_values[start:], d_W[start:], yerr=d_W_err[start:], fmt='o', label='Raw data window', capsize=4)
    plt.errorbar(x_values[start:], d[start:], yerr=d_err[start:], fmt='o', label='Raw data', capsize=4)
    plt.errorbar(x_values[start:], d_l1[start:], yerr=d_l1_err[start:], fmt='s', label='L1 Regularization', capsize=4)
    plt.xlabel("# Acquired Frames")
    plt.ylabel("Entanglement Dimensionality Lower Bound")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(0.0, 11.0)
    plt.show()

    return d

def plot_SNR(folder_path, file_name):
    save_path = os.path.join(folder_path, file_name)
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            total = pickle.load(f)
        print(f"Loaded cached results from: {save_path}")

    cov_array = np.array([cov for _, cov, _, _, _, _, _ in total])
    cov_array_l1 = np.array([cov_l1 for _, _, _, _, cov_l1, _, _ in total])
    names = np.array([fname for fname, _, _, _, _, _, _ in total])
    X = np.array([int(re.search(r"(\d+)", name).group(1)) for name in names]) * 1718
    SNR = []
    SNR_l1 = []

    h, w = cov_array[0].shape
    half_box = 10 // 2
    center_y, center_x = h // 2, w // 2

    y1 = center_y - half_box
    y2 = center_y + half_box
    x1 = center_x - half_box
    x2 = center_x + half_box
    mask = np.ones_like(cov_array[0], dtype=bool)
    mask[y1:y2, x1:x2] = False
    for autoconv, autoconv_l1 in zip(cov_array, cov_array_l1):
        SNR.append(np.max(autoconv[~mask]) / np.std(autoconv[mask]))
        SNR_l1.append(np.max(autoconv_l1[~mask]) / np.std(autoconv_l1[mask]))

    plt.figure()
    plt.scatter(X,np.array(SNR), label='Raw Data')
    plt.scatter(X,np.array(SNR_l1), label='L1 Regularization')
    plt.legend(loc='lower right', fontsize=12)
    # plt.title('SNR = max / $std_{noise}$')
    # plt.xlabel('# Acquired frames', fontsize=15)
    plt.ylabel('SNR', fontsize=15)
    plt.show()


if __name__ == "__main__":
    folder_path_K = f'C:/Users/lotanstav/Desktop/Hugo_888_code/exp_final/FarField'
    folder_path_P = f'C:/Users/lotanstav/Desktop/Hugo_888_code/exp_final/NearField'
    dataset = 'l1_total_corr_wandb_window_100i_6.pkl'

    # plot_sum_coordination(folder_path_K, dataset)
    # plot_sum_coordination(folder_path_P, dataset)
    plot_EPR_from_folder(folder_path_K, folder_path_P, dataset, dataset)
    # plot_SNR(folder_path_K, dataset)
    # plot_SNR(folder_path_P, dataset)