import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, offset):
    """
    2D Gaussian function without rotation.
    """
    x, y = xy
    return amp * np.exp(-(((x - x0)**2)/(2*sigma_x**2) + ((y - y0)**2)/(2*sigma_y**2))) + offset


def fit_2d_gaussian_windowed(data, window_size=60, normalize=False, show=False, SPAD=False):
    """
    Fit a 2D Gaussian to a small window around the peak in a 2D array.

    Parameters:
    - data (2D array): Input matrix to fit.
    - window_size (int): Size of the square region around the peak to fit.
    - normalize (bool): Whether to normalize data to [0, 1] before fitting.
    - show (bool): Whether to display plots of the fit.

    Returns:
    - popt (array): Best-fit parameters [amp, x0, y0, sigma_x, sigma_y, offset].
    - perr (array): 1σ standard errors for the fit parameters.
    """
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    # Optional normalization to [0, 1]
    if normalize:
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)

    ny, nx = data.shape
    if SPAD:
        max_y, max_x = np.unravel_index(np.argmax(data), data.shape)
    else:
        max_y, max_x = 90, 90

    # Define a window around the peak
    half_win = window_size // 2
    x_min = max(0, max_x - half_win)
    x_max = min(nx, max_x + half_win)
    y_min = max(0, max_y - half_win)
    y_max = min(ny, max_y + half_win)

    data_crop = data[y_min:y_max, x_min:x_max]
    yc, xc = np.mgrid[y_min:y_max, x_min:x_max]
    xdata = np.vstack((xc.ravel(), yc.ravel()))
    zdata = data_crop.ravel()

    # Initial parameter guess
    amp_guess = data_crop.max() - data_crop.min()
    offset_guess = data_crop.min()
    p0 = [amp_guess, max_x, max_y, 2.0, 2.0, offset_guess]

    bounds = (
        [0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    )

    # Fit
    popt, pcov = curve_fit(gaussian_2d, xdata, zdata, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    sigma_x, sigma_y = popt[3], popt[4]
    sigma_x_err, sigma_y_err = perr[3], perr[4]

    sigma_avg = 0.5 * (sigma_x + sigma_y)
    sigma_avg_err = 0.5 * np.sqrt(sigma_x_err ** 2 + sigma_y_err ** 2)

    # Build full-size fit image
    fit_crop = gaussian_2d((xc, yc), *popt).reshape(data_crop.shape)
    full_fit = np.zeros_like(data)
    full_fit[y_min:y_max, x_min:x_max] = fit_crop

    # Plotting
    if show:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axs[0].imshow(data, cmap='jet', origin='lower')
        # axs[0].scatter(popt[1], popt[2], color='r', label='Fit Center', s=30)
        axs[0].set_title(rf"$\sigma_x$ = {sigma_x:.4} +- {perr[3]:.3} $\sigma_y$ = {sigma_y:.4} +- {perr[4]:.3}"
                         + (" (Normalized)" if normalize else ""))
        # axs[0].legend()
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        axs[1].imshow(full_fit, cmap='jet', origin='lower')
        # axs[1].contour(full_fit, levels=8, colors='w', linewidths=0.5)
        axs[1].set_title("Fitted Gaussian")

        plt.tight_layout()
        plt.show(block=False)

    return sigma_avg, sigma_x, sigma_y, sigma_avg_err, sigma_x_err, sigma_y_err


def fit_gaussian_and_extract_sigma(data, normalize=False, show=False):
    """
    Fit a 2D Gaussian to a localized window of the data.
    Returns the average of σx and σy, and the individual values.
    """
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array.")


    if normalize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    ny, nx = data.shape
    # max_y, max_x = np.unravel_index(np.argmax(data), data.shape)

    yc, xc = np.mgrid[0:ny, 0:nx]
    xdata = np.vstack((xc.ravel(), yc.ravel()))
    zdata = data.ravel()

    # Initial guess
    amp = np.max(data) - np.min(data)
    offset = np.min(data)
    p0 = [amp, 30, 30, 1.0, 1.0, offset]
    bounds = ([0, 27, 27, 0.1, 0.1, -np.inf],
              [np.inf, 33, 33, 5, 5, np.inf])
    # p0 = [amp, 30, 30, 1.0, 1.0, offset]
    # bounds = ([0, 27, 27, 0.1, 0.1, -np.inf],
    #           [np.inf, 33, 33, 5, 5, np.inf])

    popt, _ = curve_fit(gaussian_2d, xdata, zdata, p0=p0, bounds=bounds)
    sigma_x, sigma_y = popt[3], popt[4]

    # Build full-size fit image
    full_fit = gaussian_2d((xc, yc), *popt).reshape(data.shape)

    if show:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(data, cmap='viridis', origin='lower')
        axs[0].scatter(popt[1], popt[2], color='r', label='Fit Center', s=30)
        axs[0].set_title("Original Data" + (" (Normalized)" if normalize else ""))
        axs[0].legend()

        axs[1].imshow(full_fit, cmap='viridis', origin='lower')
        axs[1].contour(full_fit, levels=8, colors='w', linewidths=0.5)
        axs[1].set_title("Fitted Gaussian")

        plt.tight_layout()
        plt.show()

    return 0.5 * (sigma_x + sigma_y), sigma_x, sigma_y
