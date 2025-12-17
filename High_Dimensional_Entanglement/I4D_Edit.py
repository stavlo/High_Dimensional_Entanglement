import numpy as np
import scipy.io
import scipy.ndimage
from scipy.io import savemat
import os
import pickle


def I4D_calc(path, DX, DY, normalize=False, shifted=False):
    mat_data = scipy.io.loadmat(path)
    SssT = np.array(mat_data['FarField_partial']['SssT'][0,0])
    BcssT = np.array(mat_data['FarField_partial']['BcssT'][0,0])
    I4D = SssT - BcssT
    I4D = mean4d_x_smoothing(I4D, DX, DY)
    if shifted:
        I4D += np.abs(np.min(I4D)) + 1e-9
    if normalize:
        I4D /= np.sum(I4D)

    return I4D


def mean4d_x_smoothing(I4D, DX, DY):
    """
    Smooths a reshaped 4D correlation matrix along the X-direction (columns)
    by averaging each column with its left and right neighbors.
    Then subtracts the global mean.

    Parameters:
    - I4D : ndarray of shape [DY*DX, DY*DX]
        The reshaped 4D correlation matrix (flattened to 2D).
    - DX, DY : int
        Spatial dimensions of the original 2D image grid.

    Returns:
    - I4D2 : ndarray of same shape as I4D
        Smoothed and mean-subtracted correlation matrix.
    """

    I4D2 = np.copy(I4D)

    # Interior smoothing: for kx in 2 to DX-1 (1-based in MATLAB → 1 to DX-2 in Python)
    for kx in range(1, DX-1):
        for ky in range(DY):
            XsYs = (kx) * DY + ky  # Equivalent to (kx-1)*DY + ky in 1-based indexing
            Mcond = I4D[XsYs, :].reshape((DY, DX), order='F')  # Reshape to 2D
            Mcond[:, kx] = (Mcond[:, kx - 1] + Mcond[:, kx + 1]) / 2
            I4D2[XsYs, :] = Mcond.flatten(order='F')  # Flatten back in column-major

    # Left boundary (kx = 0)
    for ky in range(DY):
        XsYs = 0 * DY + ky
        Mcond = I4D[XsYs, :].reshape((DY, DX), order='F')
        Mcond[:, 0] = Mcond[:, 1]
        I4D2[XsYs, :] = Mcond.flatten(order='F')

    # Right boundary (kx = DX-1)
    for ky in range(DY):
        XsYs = (DX - 1) * DY + ky
        Mcond = I4D[XsYs, :].reshape((DY, DX), order='F')
        Mcond[:, DX - 1] = Mcond[:, DX - 2]
        I4D2[XsYs, :] = Mcond.flatten(order='F')

    # # Global mean subtraction
    # I4D2 -= np.mean(np.mean(I4D2, axis=0))

    return I4D2


def extract_ROI(I4D, row_range, col_range, N=121):
    row_start, row_end = row_range
    col_start, col_end = col_range

    row_indices = np.arange(row_start, row_end)
    col_indices = np.arange(col_start, col_end)

    # Create a grid of (row, col) and convert to flat indices in row-major order
    rr, cc = np.meshgrid(row_indices, col_indices, indexing='ij')
    flat_indices = (rr * N + cc).ravel()

    # Extract submatrix using np.ix_ for 2D slicing
    I4D_window = I4D[np.ix_(flat_indices, flat_indices)]

    return I4D_window


def conver_npy2mat(folder_path, file_name, save_name):
    file_path = os.path.join(folder_path, file_name)
    save_path = os.path.join(folder_path, save_name)
    if  os.path.exists(file_path):
        with open(save_path, 'rb') as f:
            total = pickle.load(f)
        print(f"Loaded cached results from: {save_path}")

    cov_array = np.array([cov for _, cov, _, _, _ in total])
    cov_array_l1 = np.array([cov_l1 for _, _, _, cov_l1, _ in total])
    names = np.array([fname for fname, _, _, _, _ in total])

    mat_dict = {
        'cov_array': cov_array,
        'cov_array_l1': cov_array_l1,
        'names': names
    }
    # Save to .mat file
    savemat(save_path, mat_dict)
