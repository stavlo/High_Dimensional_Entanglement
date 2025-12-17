from I4D_Edit import *

def convolution_reader(I4D, Idis, vecimage):
    """
    Compute the 2D convolution image from a sparse 4D correlation matrix.

    Parameters:
    - I4D : 2D ndarray [len(vecimage) x len(vecimage)]
        The compressed 4D correlation matrix (flattened).
    - Idis : 2D ndarray [DY x DX]
        An image used to extract the shape (DY, DX) of the original grid.
    - vecimage : 1D ndarray
        Linear indices (1-based, MATLAB style) of valid pixels in the original grid.

    Returns:
    - D4D : 2D ndarray [2*DY-1 x 2*DX-1]
        The reconstructed cross-convolution image.
    """

    # Get dimensions from the reference image
    DY, DX = Idis.shape
    D4D = np.zeros((2 * DY - 1, 2 * DX - 1))
    KK = -1  # Column counter for D4D

    # First loop: upper triangle
    for K1 in range(DX):
        KK += 1
        vyl = np.arange(K1 * DY, (K1 + 1) * DY)
        vyc = np.arange(DY)

        for K2 in range(K1 + 1):
            # Intersect vecimage with row/col indices
            Cyl, Cyykl, Cykl = np.intersect1d(vyl, vecimage, return_indices=True)
            Cyc, Cyykc, Cykc = np.intersect1d(vyc, vecimage, return_indices=True)

            Car = np.zeros((DY, DY))
            if len(Cyl) > 0 and len(Cyc) > 0:
                Car[np.ix_(Cyykl, Cyykc)] = I4D[np.ix_(Cykl, Cykc)]

            # Flip vertically and sum over antidiagonals
            D4D[:, KK] += sum_diagonals(np.flipud(Car)).T
            vyl -= DY
            vyc += DY

    # Second loop: lower triangle
    for K1 in range(1, DX):
        KK += 1
        vyc = np.arange(K1 * DY, (K1 + 1) * DY)
        vyl = np.arange(DY * (DX - 1), DY * DX)
        for K2 in range(DX - 1, K1 - 1, -1):
            Cyl, Cyykl, Cykl = np.intersect1d(vyl, vecimage, return_indices=True)
            Cyc, Cyykc, Cykc = np.intersect1d(vyc, vecimage, return_indices=True)

            Car = np.zeros((DY, DY))
            if len(Cyl) > 0 and len(Cyc) > 0:
                Car[np.ix_(Cyykl, Cyykc)] = I4D[np.ix_(Cykl, Cykc)]

            D4D[:, KK] += sum_diagonals(np.flipud(Car))
            vyl = vyl - DY
            vyc = vyc + DY

    return D4D


def correlation_reader(I4D, Idis):
    """
    Compute the 2D cross-correlation image from a 4D matrix stored in 2D format.

    Parameters:
    - I4D : 2D ndarray
        The compressed 4D matrix represented as 2D (flattened).
    - Idis : 2D ndarray
        Reference image used to extract shape (DY, DX) of the original spatial layout.

    Returns:
    - D4D : 2D ndarray
        The reconstructed cross-correlation image.
    """
    DY, DX = Idis.shape
    D4D = np.zeros((2 * DY - 1, 2 * DX - 1))
    KK = -1

    # First loop: top-left to bottom-right diagonals
    for K1 in range(DX):
        K11 = 0
        KK += 1
        for K2 in range(DX - 1, DX - K1 - 1, -1):
            v1_start = (K1 + K11) * DY
            v1_end = (K1 + K11 + 1) * DY
            v2_start = K2 * DY
            v2_end = (K2 + 1) * DY
            v1 = slice(v1_start, v1_end)
            v2 = slice(v2_start, v2_end)
            D4D[:, KK] += sum_diagonals(I4D[v1, v2])
            K11 -= 1

    # Second loop: bottom-left to top-right diagonals
    for K1 in range(1, DX):
        K11 = 0
        KK += 1
        for K2 in range(DX - K1):
            v1_start = (K1 + K11) * DY
            v1_end = (K1 + K11 + 1) * DY
            v2_start = K2 * DY
            v2_end = (K2 + 1) * DY
            v1 = slice(v1_start, v1_end)
            v2 = slice(v2_start, v2_end)
            D4D[:, KK] += sum_diagonals(I4D[v1, v2])
            K11 += 1

    return D4D


def sum_diagonals(M):
    """
    Sum all diagonals of a square matrix M, from bottom-left to top-right,
    i.e., diagonals with negative and positive offsets relative to the main diagonal.

    Parameters:
    - M : 2D ndarray (square)

    Returns:
    - Mc : 1D ndarray of shape (2*N - 1,)
        Diagonal sums, ordered from bottom-left (-N+1) to top-right (+N-1).
    """
    N1, N2 = M.shape
    Mc = np.zeros(2 * N1 - 1)
    offset = N1 - 1  # to shift index to zero-based array

    for k in range(-N1 + 1, N1):
        Mc[k + offset] = np.sum(np.diag(M, -k))
    return Mc

