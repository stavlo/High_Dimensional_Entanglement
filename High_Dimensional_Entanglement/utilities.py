import matplotlib.pyplot as plt
import scipy.io
import scipy.ndimage
from I4D_Edit import *
from sum_coordination import convolution_reader, correlation_reader
import DoubleGaussian
import re
import pickle
import optimization

def extract_number(filename):
    """Extract the trailing number before the extension in a filename."""
    match = re.search(r'_(\d+)\.mat$', filename)
    return int(match.group(1)) if match else float('inf')

def epr_calc(sigma_pos_m, sigma_mom_rad_per_m, sigma_pos_m_err, sigma_mom_rad_per_m_err):
    epr_product = sigma_pos_m * sigma_mom_rad_per_m
    epr_product_err = np.sqrt((sigma_mom_rad_per_m * sigma_pos_m_err) ** 2 +
                              (sigma_pos_m * sigma_mom_rad_per_m_err) ** 2)
    return epr_product, epr_product_err

def dim_calc(sigma_pos_m, sigma_mom_rad_per_m, sigma_pos_m_err, sigma_mom_rad_per_m_err):
    d = 1 / (np.exp(1) * sigma_pos_m * sigma_mom_rad_per_m)
    d_err = np.sqrt(
        (sigma_pos_m_err / (sigma_pos_m ** 2 * sigma_mom_rad_per_m)) ** 2 +
        (sigma_mom_rad_per_m_err / (sigma_pos_m * sigma_mom_rad_per_m ** 2)) ** 2) / np.exp(1)
    return d, d_err

def dataset_creation(folder_path, file_name, load_if_exists=True, proj='sum',window=500):
    save_path = os.path.join(folder_path, file_name)
    S = []
    Sl1 = []
    S_err = []
    Sl1_err = []

    if save_path and load_if_exists and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            total = pickle.load(f)
        S = np.array([sigma for _, _, sigma, _, _, _, _ in total])
        S_err = np.array([error for _, _, _, error, _, _, _ in total])
        Sl1 = np.array([sigma_l1 for _, _, _, _, _, sigma_l1, _ in total])
        Sl1_err = np.array([error_l1 for _, _, _, _, _, _, error_l1 in total])
        names = np.array([fname for fname, _, _, _, _, _, _ in total])

        print(f"Loaded cached results from: {save_path}")
        if window:
            autoconv_array = np.array([autoconv for _, autoconv, _, _, _, _, _ in total])
            autoconv_l1_array = np.array([autoconv_l1 for _, _, _, _, autoconv_l1, _, _ in total])
            i = 0
            for autoconv, autoconv_l1 in zip(autoconv_array, autoconv_l1_array):

                try:
                    avg_sigma, _, _, err, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autoconv, window_size=window,
                                                                                         show=False)
                    avg_sigma_l1, _, _, err_l1, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autoconv_l1,
                                                                                               window_size=window,
                                                                                               show=False)
                except:
                    print(f'Gaussian fit failed for {names[i]}')
                    avg_sigma, avg_sigma_l1 = 0, 0
                    err, err_l1 = 0, 0
                S[i] = avg_sigma
                S_err[i] = err
                Sl1[i] = avg_sigma_l1
                Sl1_err[i] = err_l1
                i = i + 1
    else:
        total = []
        file_names = sorted([f for f in os.listdir(folder_path) if (f.endswith('.mat') and f.startswith('Far'))], key=extract_number)
        DX, DY = 121, 121
        DXW, DYW = 90, 90
        vecimage = np.linspace(0, DYW * DXW, DYW * DXW + 1)
        Rd = np.zeros((DXW, DYW))

        for fname in file_names:
            fpath = os.path.join(folder_path, fname)
            I4D_K = I4D_calc(fpath, DX, DY, normalize=False, shifted=False)
            # 90x90
            # I4D_window = extract_ROI(I4D_K, (22, 112), (6, 96))
            I4D_window = extract_ROI(I4D_K, (21, 111), (5, 95))
            opt_I4D_K, loss = optimization.optimize_x(I4D_window, 10**(-4.8), 0.0,learning_rate=10 ** -1.403 ,max_iter=100)
            # opt_I4D_K, loss = optimization.optimize_x(I4D_window, 10 ** (-7.5), 0.0, learning_rate=10 ** 1.5,
            #                                           max_iter=1000,
            #                                           if_log=False, upper_triangular=True)
            if proj=='sum':
                autoconv_l1 = convolution_reader(opt_I4D_K, Rd, vecimage)
                autoconv = convolution_reader(I4D_window, Rd, vecimage)
            else:
                autoconv_l1 = correlation_reader(opt_I4D_K, Rd)
                autoconv = correlation_reader(I4D_window, Rd)
            try:
                avg_sigma, _, _, err, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autoconv, window_size=window, show=False)
            except:
                print(f'Gaussian fit failed for {fname}')
                avg_sigma, err = 0, 0
            try:
                avg_sigma_l1, _, _, err_l1, _, _ = DoubleGaussian.fit_2d_gaussian_windowed(autoconv_l1, window_size=window,
                                                                                           show=False)
            except:
                print(f'Gaussian fit failed for {fname} L1')
                avg_sigma_l1, err_l1 = 0, 0
            S.append(avg_sigma)
            S_err.append(err)
            Sl1.append(avg_sigma_l1)
            Sl1_err.append(err_l1)

            total.append((fname, autoconv, avg_sigma, err, autoconv_l1, avg_sigma_l1, err_l1))
            print(f'Finished calculating sum_corr in {fname}')
            if save_path:
                with open(save_path, 'wb') as f:
                    pickle.dump(total, f)
                print(f"Saved results to: {save_path}")

    names = np.array([fname for fname, _, _, _, _, _, _ in total])
    num_img = np.array([int(re.search(r"(\d+)(?=\.mat$)", name).group(1)) for name in names]) * 1718

    return np.array(S), np.array(S_err), np.array(Sl1), np.array(Sl1_err), np.array(num_img)

def plot_ROI(folder_path):
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')], key=extract_number)
    Nset = -2
    # DXW, DYW = 30, 30
    DXW, DYW = 90, 90
    DX, DY = 121, 121
    vecimage = np.linspace(0, DYW * DXW, DYW * DXW + 1)
    Rd = np.zeros((DXW, DYW))
    fpath = os.path.join(folder_path, file_names[Nset])
    I4D_K = I4D_calc(fpath, DX, DY, normalize=False, shifted=False)
    # 90x90
    I4D_window = extract_ROI(I4D_K, (21, 111), (5, 95))
    # autoconv
    # I4D_window = extract_ROI(I4D_K, (19, 109), (12, 102))

    # # 30x30
    # I4D_window = extract_ROI(I4D_K, (52, 82), (37, 67))
    autoconv = convolution_reader(I4D_window, Rd, vecimage)
    # autoconv = correlation_reader(I4D_window, Rd)
    plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(I4D_window, cmap='jet')
    # plt.subplot(1,2,2)
    plt.imshow(autoconv, cmap='hot')
    plt.colorbar()
    plt.title(file_names[Nset])
    plt.show(block=True)

def convert_pixel_units(sigma_pos_pix, sigma_mom_pix,pixel_size_m=16e-6, wavelength_m=808e-9, focal_length_m=100e-3, M=(1,1)):
    # Convert to physical units
    sigma_pos_m = sigma_pos_pix * pixel_size_m / M[0]
    sigma_mom_rad_per_m = sigma_mom_pix * (2 * np.pi / wavelength_m) * (pixel_size_m / focal_length_m) / M[1]
    return sigma_pos_m, sigma_mom_rad_per_m

if __name__ == "__main__":
    folder_path_P = r"C:\Users\lotanstav\Desktop\Hugo_888_code\exp_final\NearField"
    folder_path_K = r"C:\Users\lotanstav\Desktop\Hugo_888_code\exp_final\FarField"

    # # # dataset pass
    # folder_path_K = f'C:/Users/lotanstav/Desktop/Hugo_888_code/short_exp/FarField'
    # folder_path_P = f'C:/Users/lotanstav/Desktop/Hugo_888_code/short_exp/NearField'

    # creating the dataset from covariance matrices
    dataset_creation(folder_path_P, 'l1_total_corr_wandb_window_100i_3.pkl', load_if_exists=True, proj='minus')
    dataset_creation(folder_path_K, 'l1_total_corr_wandb_window_100i_3.pkl', load_if_exists=True, proj='sum')

