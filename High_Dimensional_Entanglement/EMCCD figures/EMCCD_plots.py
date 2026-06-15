import matplotlib.pyplot as plt
from I4D_Edit import *
import re
import pickle
from utilities import dataset_creation, epr_calc, dim_calc, convert_pixel_units
import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt


def _extract_frame_number(fname, frames_per_file=1718):
    """
    Extract frame number from file name and multiply by frames_per_file.
    Example: FarField_partial_50.mat -> 50 * 1718
    """
    match = re.search(r"(\d+)(?=\.mat$)", str(fname))
    if match is None:
        return np.nan
    return int(match.group(1)) * frames_per_file


def plot_supp_projection_grid(
    folder_path,
    file_name,
    projection="sum",
    save_path="Fig_S1.pdf",
    selected_indices=None,
    start=0,
    step=1,
    max_cols=8,
    frames_per_file=1718,
    cmap="hot",
):
    pkl_path = os.path.join(folder_path, file_name)

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Could not find: {pkl_path}")

    with open(pkl_path, "rb") as f:
        total = pickle.load(f)

    # total structure:
    # (fname, autoconv, avg_sigma, err, autoconv_l1, avg_sigma_l1, err_l1)
    names = np.array([entry[0] for entry in total])
    raw_imgs = [np.asarray(entry[1], dtype=float) for entry in total]
    opt_imgs = [np.asarray(entry[4], dtype=float) for entry in total]

    frames = np.array([
        _extract_frame_number(name, frames_per_file=frames_per_file)
        for name in names
    ])

    if selected_indices is None:
        selected_indices = np.arange(start, len(raw_imgs), step)[:max_cols]
    else:
        selected_indices = np.asarray(selected_indices)

    raw_sel = [raw_imgs[i] for i in selected_indices]
    opt_sel = [opt_imgs[i] for i in selected_indices]
    frame_sel = frames[selected_indices]

    ncols = len(selected_indices)

    vmin, vmax = np.min(raw_sel[6:]), np.max(raw_sel[6:])

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(1.45 * ncols + 1.0, 3.2),
        sharex=True,
        sharey=True,
    )

    if ncols == 1:
        axes = np.array(axes).reshape(2, 1)

    last_im = None

    for j in range(ncols):
        for row, imgs in [
            (0, raw_sel),
            (1, opt_sel),
        ]:
            ax = axes[row, j]

            last_im = ax.imshow(
                imgs[j],
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )

            if row == 0:
                ax.set_title(rf"$N={int(frame_sel[j]):,}$", pad=4)

            # Keep only sparse ticks to show pixel units without crowding
            ny, nx = imgs[j].shape
            ax.set_xticks([0, nx // 2, nx - 1])
            ax.set_yticks([0, ny // 2, ny - 1])

            if row == 0:
                ax.tick_params(labelbottom=False)

            if j != 0:
                ax.tick_params(labelleft=False)
    if projection == 'sum':
        fig.supxlabel(r"$k_{x1}+k_{x2}$ [pixels]", fontsize=12)
        fig.supylabel(r"$k_{y1}+k_{y2}$ [pixels]", fontsize=12, x=0.04)
    else:
        fig.supxlabel(r"$x_{x1}-x_{x2}$ [pixels]", fontsize=12)
        fig.supylabel(r"$x_{y1}-x_{y2}$ [pixels]", fontsize=12, x=0.04)

    fig.subplots_adjust(
        left=0.09,
        right=0.88,  # important: reserve space for colorbar
        bottom=0.18,
        top=0.88,
        wspace=0.05,
        hspace=0.08,
    )

    # Dedicated colorbar axis: [left, bottom, width, height]
    cax = fig.add_axes([0.9, 0.2, 0.012, 0.67])

    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("Projected Correlation amp. [arb. units]", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.show()

    return fig, axes

def plot_sum_coordination(folder_path, file_name, selected_img = None, load_if_exists=True, if_plot=False):
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
    if if_plot:
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

            # im = ax0.imshow(autoconv, cmap='hot')
            im = ax0.imshow(autoconv, cmap='hot', vmin=np.min(cov_array[6:]), vmax=np.max(cov_array[6:]))
            # im = ax0.imshow(np.log10(autoconv + np.abs(np.min(autoconv)) + 1e-9), cmap='hot', vmin=4.5, vmax=6.5)
            # fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
            # ax0.set_title(f'N = {title}, Sigma = {sigma:.3}', fontsize=12)
            ax0.set_title(f'N = {title}', fontsize=12)
            ax0.set_xticks([])
            ax0.set_yticks([])

            # im = ax1.imshow(autoconv_l1, cmap='hot')
            im = ax1.imshow(autoconv_l1, cmap='hot', vmin=np.min(cov_array[6:]), vmax=np.max(cov_array[6:]))
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
        # fig.suptitle("Sum Coordinations Heatmaps", fontsize=14)
        plt.show(block=True)
    if selected_img is not None:
        return cov_array[selected_img], cov_array_l1[selected_img]
    else:
        return


def plot_EPR_from_folder(folder_path_K, folder_path_P, file_name_K, file_name_P):
    sigma_array_K, sigma_array_K_err, l1_sigma_array_K, sigma_l1_array_K_err, names = dataset_creation(folder_path_K, file_name_K, load_if_exists=True)
    sigma_array_P, sigma_array_P_err, l1_sigma_array_P, sigma_l1_array_P_err, names = dataset_creation(folder_path_P, file_name_P, load_if_exists=True)
    sigma_array_P_W, sigma_array_P_W_err, _, _, names = dataset_creation(folder_path_P, file_name_P, load_if_exists=True, window=100)
    sigma_array_K_W, sigma_array_K_W_err, _, _, names = dataset_creation(folder_path_K, file_name_K, load_if_exists=True, window=100)

    plot_sigma_supp_revised(
        frames=names,

        sigma_K=sigma_array_K,
        sigma_K_err=sigma_array_K_err,
        sigma_K_window=sigma_array_K_W,
        sigma_K_window_err=sigma_array_K_W_err,
        sigma_K_opt=l1_sigma_array_K,
        sigma_K_opt_err=sigma_l1_array_K_err,

        sigma_P=sigma_array_P,
        sigma_P_err=sigma_array_P_err,
        sigma_P_window=sigma_array_P_W,
        sigma_P_window_err=sigma_array_P_W_err,
        sigma_P_opt=l1_sigma_array_P,
        sigma_P_opt_err=sigma_l1_array_P_err,

        save_path="Fig_S3.pdf",
    )
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.errorbar(names, sigma_array_K_W, yerr=sigma_array_K_W_err, fmt='o', label='Raw Data window', capsize=4)
    # plt.errorbar(names, sigma_array_K, yerr=sigma_array_K_err, fmt='o', label='Raw Data', capsize=4)
    # plt.errorbar(names, l1_sigma_array_K, yerr=sigma_l1_array_K_err, fmt='o', label='L1-Regularized', capsize=4)
    # plt.xlabel('# Acquired Frames', fontsize=16)
    # plt.ylabel(r'$\sigma_{avg} [pixels]$', fontsize=16)
    # # plt.title('Gaussian Fit Width with Error Bars')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.subplot(1,2,2)
    # plt.errorbar(names, sigma_array_P_W, yerr=sigma_array_P_W_err, fmt='o', label='Raw Data window', capsize=4)
    # plt.errorbar(names, sigma_array_P, yerr=sigma_array_P_err, fmt='o', label='Raw Data', capsize=4)
    # plt.errorbar(names, l1_sigma_array_P, yerr=sigma_l1_array_P_err, fmt='o', label='L1-Regularized', capsize=4)
    # plt.xlabel('# Acquired Frames', fontsize=16)
    # plt.ylabel(r'$\sigma_{avg}$', fontsize=16)
    # # plt.title('Gaussian Fit Width with Error Bars')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # # plt.ylim(-1.0, 2.0)
    # plt.show(block=False)

    sigma_pos_m, sigma_mom_rad_per_m = convert_pixel_units(sigma_array_P, sigma_array_K[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)
    sigma_pos_m_err, sigma_mom_rad_per_m_err = convert_pixel_units(sigma_array_P_err, sigma_array_K_err[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)
    sigma_pos_m_W, sigma_mom_rad_per_m_W = convert_pixel_units(sigma_array_P_W, sigma_array_K_W[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)
    sigma_pos_m_W_err, sigma_mom_rad_per_m_W_err = convert_pixel_units(sigma_array_P_W_err, sigma_array_K_W_err[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)
    sigma_pos_m_l1, sigma_mom_rad_per_m_l1 = convert_pixel_units(l1_sigma_array_P, l1_sigma_array_K[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)
    sigma_pos_m_l1_err, sigma_mom_rad_per_m_l1_err = convert_pixel_units(sigma_l1_array_P_err, sigma_l1_array_K_err[:], pixel_size_m=13e-6, wavelength_m=808e-9, focal_length_m=100e-3)

    epr_product_W, epr_product_W_err= epr_calc(sigma_pos_m_W, sigma_mom_rad_per_m_W, sigma_pos_m_W_err, sigma_mom_rad_per_m_W_err)
    epr_product, epr_product_err= epr_calc(sigma_pos_m, sigma_mom_rad_per_m, sigma_pos_m_err, sigma_mom_rad_per_m_err)
    epr_product_l1, epr_product_l1_err= epr_calc(sigma_pos_m_l1, sigma_mom_rad_per_m_l1, sigma_pos_m_l1_err, sigma_mom_rad_per_m_l1_err)

    x_values = names
    plot_metric_vs_frames(
        frames=x_values,
        y_raw=epr_product,
        y_raw_err=epr_product_err,
        y_window=epr_product_W,
        y_window_err=epr_product_W_err,
        y_opt=epr_product_l1,
        y_opt_err=epr_product_l1_err,
        hline = 0.5,
        ylabel="EPR Product (Δx·Δk)",
        save_path="Fig_S4.pdf",
    )

    d_W, d_W_err = dim_calc(sigma_pos_m_W, sigma_mom_rad_per_m_W, sigma_pos_m_W_err, sigma_mom_rad_per_m_W_err)
    d, d_err = dim_calc(sigma_pos_m, sigma_mom_rad_per_m, sigma_pos_m_err, sigma_mom_rad_per_m_err)
    d_l1, d_l1_err = dim_calc(sigma_pos_m_l1, sigma_mom_rad_per_m_l1, sigma_pos_m_l1_err, sigma_mom_rad_per_m_l1_err)

    plot_metric_vs_frames(
        frames=x_values,
        y_raw=d,
        y_raw_err=d_err,
        y_window=d_W,
        y_window_err=d_W_err,
        y_opt=d_l1,
        y_opt_err=d_l1_err,
        ylabel="Entanglement dimensionality lower bound",
        save_path="Fig_4.pdf",
    )
    return d


def plot_sigma_supp_revised(
    frames,
    sigma_K,
    sigma_K_err,
    sigma_K_window,
    sigma_K_window_err,
    sigma_K_opt,
    sigma_K_opt_err,
    sigma_P,
    sigma_P_err,
    sigma_P_window,
    sigma_P_window_err,
    sigma_P_opt,
    sigma_P_opt_err,
    save_path="Supp_Fig_S3_sigma.pdf",
  ):

    frames = np.asarray(frames, dtype=float)

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

    datasets = [
        (
            axes[0],
            sigma_K,
            sigma_K_err,
            sigma_K_window,
            sigma_K_window_err,
            sigma_K_opt,
            sigma_K_opt_err,
            "a",
        ),
        (
            axes[1],
            sigma_P,
            sigma_P_err,
            sigma_P_window,
            sigma_P_window_err,
            sigma_P_opt,
            sigma_P_opt_err,
            "b",
        ),
    ]

    for ax, y_raw, y_raw_err, y_win, y_win_err, y_opt, y_opt_err, label in datasets:

        ax.errorbar(
            frames,
            y_raw,
            yerr=y_raw_err,
            fmt="o",
            ms=5,
            capsize=3,
            linestyle="None",
            label="Raw data",
        )

        ax.errorbar(
            frames,
            y_win,
            yerr=y_win_err,
            fmt="^",
            ms=5,
            capsize=3,
            linestyle="None",
            label="Raw data, reduced window",
        )

        ax.errorbar(
            frames,
            y_opt,
            yerr=y_opt_err,
            fmt="s",
            ms=5,
            capsize=3,
            linestyle="None",
            label=r"$\ell_1$ reconstruction",
        )

        ax.set_xlabel(r"# of Acquired frames")
        ax.set_ylabel(r"$\sigma_{\rm avg}$ [pixels]")
        ax.grid(True, alpha=0.25)

        ax.text(
            -0.18, 1.06, label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
            clip_on=False,
        )

    # One legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.18,
        top=0.78,
        wspace=0.32,
    )

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.show()

    return fig, axes

def plot_metric_vs_frames(
    frames,
    y_raw,
    y_raw_err=None,
    y_window=None,
    y_window_err=None,
    y_opt=None,
    y_opt_err=None,
    hline=None,
    ylabel=r"Lower bound, $d$",
    save_path="figure.pdf"
):

    frames = np.asarray(frames, dtype=float)
    y_raw = np.asarray(y_raw)

    if y_raw_err is None:
        y_raw_err = np.zeros_like(y_raw)

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        frames, y_raw, yerr=y_raw_err,
        fmt="o", ms=5, capsize=3, linestyle="None",
        label="Raw data",
    )

    if y_window is not None:
        if y_window_err is None:
            y_window_err = np.zeros_like(y_window)

        ax.errorbar(
            frames, y_window, yerr=y_window_err,
            fmt="^", ms=5, capsize=3, linestyle="None",
            label="Raw data, reduced window",
        )

    if y_opt is not None:
        if y_opt_err is None:
            y_opt_err = np.zeros_like(y_opt)

        ax.errorbar(
            frames, y_opt, yerr=y_opt_err,
            fmt="s", ms=5, capsize=3, linestyle="None",
            label=r"$\ell_1$ reconstruction",
        )

    ax.set_xlabel(r"# of Acquired frames")
    if ylabel == "Entanglement dimensionality lower bound":
        ax.set_ylabel(ylabel, labelpad=10, fontsize=9)
    else:
        ax.set_ylabel(ylabel, labelpad=10)

    if hline is not None:
        ax.axhline(
            hline,
            linestyle="--",
            linewidth=1.2,
            color="red",
            alpha=0.7,
            label='Heisenberg Limit (0.5)'
        )

    ax.grid(True, alpha=0.25)

    # Legend above the data
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    # Fix margins manually
    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.18,
        top=0.82,
    )

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.show()

    return fig, ax

def plot_SNR(folder_path, file_name, if_plot=False):
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
    if if_plot:
        plt.figure()
        plt.scatter(X,np.array(SNR), label='Raw Data')
        plt.scatter(X,np.array(SNR_l1), label='L1 Regularization')
        plt.legend(loc='lower right', fontsize=12)
        # plt.title('SNR = max / $std_{noise}$')
        # plt.xlabel('# Acquired frames', fontsize=15)
        plt.ylabel('SNR', fontsize=15)
        plt.show()
    return X, np.array(SNR), np.array(SNR_l1)

def plot_fig2(folder_path_K, folder_path_P, dataset, save_path="Fig_2.pdf"):
    N, SNR_K, SNR_K_opt = plot_SNR(folder_path_K, dataset)
    N, SNR_P, SNR_P_opt = plot_SNR(folder_path_P, dataset)

    img, img_opt = plot_sum_coordination(folder_path_K, dataset, selected_img=4)

    n = len(SNR_K)

    x = np.asarray(N)
    xlabel = "# of Acquired frames"

    image_index = n // 2

    x_mark = x[image_index]

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    panel_fs = 16
    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 0.05])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[:, 2])

    # ---------- panels a,b: SNR curves ----------
    ax_a.scatter(x, SNR_K, s=18, label="Raw data")
    ax_a.scatter(x, SNR_K_opt, s=18, marker="s", label=r"$\ell_1$ reconstruction")
    ax_a.axvline(x_mark, ls="--", lw=1, color="k")
    ax_a.scatter(x_mark, SNR_K_opt[image_index], s=90, marker="*", color="k", zorder=5)
    ax_a.scatter(x_mark, SNR_K[image_index], s=90, marker="*", color="k", zorder=5)
    ax_a.set_ylabel("SNR")
    ax_a.legend(frameon=False)
    ax_a.text(0.03, 0.93, "a", transform=ax_a.transAxes, fontsize=panel_fs, fontweight="bold", va="top")

    ax_b.scatter(x, SNR_P, s=18, label="Raw data")
    ax_b.scatter(x, SNR_P_opt, s=18, marker="s", label=r"$\ell_1$ reconstruction")
    ax_b.set_ylabel("SNR")
    ax_b.set_xlabel(xlabel)
    ax_b.legend(frameon=False)
    ax_b.text(0.03, 0.93, "b", transform=ax_b.transAxes, fontsize=panel_fs, fontweight="bold", va="top")

    # ---------- panels c,d: representative images ----------
    vmin = -1.5e-4 # min(np.nanmin(img), np.nanmin(img_opt))
    vmax = 1.5e5 #max(np.nanmax(img), np.nanmax(img_opt))

    im_c = ax_c.imshow(img, origin="lower", cmap="hot", vmin=vmin, vmax=vmax)
    ax_d.imshow(img_opt, origin="lower", cmap="hot", vmin=vmin, vmax=vmax)

    for ax, label in zip(
        [ax_c, ax_d],
        ["c", "d"],
    ):
        if label == "d":
            ax.set_xlabel(r"$k_{x1}+k_{x2}$ [pixels]")
        ax.set_ylabel(r"$k_{y1}+k_{y2}$ [pixels]")
        ax.text(
            0.04,
            0.94,
            label,
            transform=ax.transAxes,
            fontsize=panel_fs,
            fontweight="bold",
            color="white",
            va="top",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1),
        )
    # later:
    cbar = fig.colorbar(im_c, cax=cax)
    cbar.set_label("Projected Correlation Amplitude [arb. units]")
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    folder_path_K = f'C:/Users/lotanstav/Desktop/Hugo_888_code/exp_final/FarField'
    folder_path_P = f'C:/Users/lotanstav/Desktop/Hugo_888_code/exp_final/NearField'
    dataset = 'test_7_4_final.pkl'
    plot_fig2(folder_path_K, folder_path_P, dataset)
    plot_EPR_from_folder(folder_path_K, folder_path_P, dataset, dataset)

    plot_supp_projection_grid(
        folder_path=folder_path_K,
        file_name=dataset,
        projection="sum",
        save_path="Fig_S1.pdf",
        max_cols=8,
    )
    plot_supp_projection_grid(
        folder_path=folder_path_P,
        file_name=dataset,
        projection="minus",
        save_path="Fig_S2.pdf",
        max_cols=8,
    )
