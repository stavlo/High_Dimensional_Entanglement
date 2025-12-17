import torch
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import I4D_Edit
import os
import sum_coordination
import DoubleGaussian
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lasso_loss(X, X_Origin, lambda_, mu, SPAD=False):
    # Compute the Lasso loss
    lossMse = torch.nn.MSELoss()
    frobenius_norm = lossMse(X, X_Origin)
    if SPAD:
        l1_norm = torch.sum(torch.abs(X))
    else:
        l1_norm = torch.sum(torch.abs(X + torch.abs(torch.min(X))))
    # print(f'The L1 norm is {l1_norm}')
    # nuclear_norm = torch.sum(torch.abs(torch.linalg.eigvals(X)))  # Nuclear norm (sum of singular values)
    nuclear_norm = 0
    loss = 0.5 * frobenius_norm + lambda_ * l1_norm + mu * nuclear_norm
    if torch.isnan(frobenius_norm):
        print("NaN detected in Frobenius norm computation!")

    if torch.isnan(l1_norm):
        print("NaN detected in L1 norm computation!")

    if torch.isnan(loss):
        print("NaN detected in final loss computation!")
    return loss

def optimize_x(X_init, lambda_, mu, learning_rate=10 ** -3.8, max_iter=150, upper_triangular=False, SPAD=False):
    if upper_triangular:
        X_init = np.triu(X_init)

    X = torch.tensor(X_init, dtype=torch.float32, requires_grad=True, device=device)
    X_Origin = torch.tensor(X_init, dtype=torch.float32, device=device)
    X_final = X.clone().detach().cpu().numpy()
    loss_min = torch.inf
    # Define the optimizer
    optimizer = optim.Adam([X], lr=learning_rate, weight_decay=0)
    loss_list = []
    for i in range(max_iter):
        # Zero the gradients
        optimizer.zero_grad()
        # Compute the loss
        loss = lasso_loss(X, X_Origin, lambda_, mu, SPAD=SPAD)

        if torch.isnan(loss).any():
            print(f"NaN detected in loss at iteration {i}. Stopping optimization.")
            break
        # Backward pass to compute gradients
        loss.backward()

        if i != 0 and loss < loss_min:
            if upper_triangular:
                X_full = X + torch.triu(X, diagonal=1).T
                X_final = X_full.clone().detach().cpu().numpy()
            else:
                X_final = X.clone().detach().cpu().numpy()
            loss_min = loss

        # Update parameters
        optimizer.step()
        # # --- enforce symmetry in-place ---
        if upper_triangular == False:
            with torch.no_grad():
                X.data = 0.5 * (X.data + X.data.T)  # <- keeps the variable symmetric
        # Print the current loss
        loss_list.append(loss.detach().cpu().numpy())

        # print(f"Iteration {i + 1}: Loss = {loss.item():.10f}")

    return X_final, loss_list

def finding_hyperparameters(X_init, lambda_val, learning_rate, max_iter=100):
    # DXW, DYW = 121, 121
    DXW, DYW = 90, 90
    vecimage = np.linspace(0, DYW * DXW, DYW * DXW + 1)
    Rd = np.zeros((DXW, DYW))

    matplotlib.use('Agg')

    # Use sweep-configured values
    lambda_val = 10**(-(lambda_val))
    learning_rate = 10**(-(learning_rate))

    # # original_img = sum_coordination.convolution_reader(X_init, Rd, vecimage)
    # original_img = sum_coordination.correlation_reader(X_init, Rd)
    # sigma, x, y = DoubleGaussian.fit_rowcol_gaussian(original_img, window_size=None, x_y=(89, 89), show=False)
    # # sigma, x, y = DoubleGaussian.fit_rowcol_gaussian(original_img, window_size=60, x_y=(121, 121), show=False)
    # fig_data, ax_data = plt.subplots()
    # im = ax_data.imshow(original_img, cmap='jet')
    # fig_data.colorbar(im, ax=ax_data, fraction=0.046, pad=0.04)
    # # ax_data.set_title(f"Original sigma = {sigma}")
    # ax_data.set_title(f"Original sigma = {x}")
    # ax_data.axis('off')
    # wandb.log({"original_convolution_data": wandb.Image(fig_data)})
    # plt.close(fig_data)

    X_opt, _ = optimize_x(X_init, lambda_val, 0, learning_rate, max_iter)
    # conv_img = sum_coordination.correlation_reader(X_opt, Rd)
    # sigma_avg, sigmaX, sigmaY = DoubleGaussian.fit_rowcol_gaussian(conv_img, window_size=60, x_y=(121, 121), show=False)
    conv_img = sum_coordination.convolution_reader(X_opt, Rd, vecimage)
    sigma_avg, sigmaX, sigmaY = DoubleGaussian.fit_rowcol_gaussian(conv_img, window_size=None, x_y=(90, 90), show=False)

    score = sigma_avg
    wandb.log({"score": score, "sigmaX": sigmaX, "sigmaY": sigmaY})
    # Log convolution image
    fig, ax = plt.subplots()
    im = ax.imshow(conv_img, cmap='jet')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Convolution Image")
    ax.axis('off')
    wandb.log({"convolution_image": wandb.Image(fig)})
    plt.close(fig)

    # # Log convolution image
    # fig2, ax2 = plt.subplots()
    # im2 = ax2.imshow(conv_img - original_img, cmap='jet')
    # fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    # ax2.set_title("Diff Image")
    # ax2.axis('off')
    # wandb.log({"Diff_image": wandb.Image(fig2)})
    # plt.close(fig2)

def wandb_sweep():
    folder_path = [f'C:/Users/lotanstav/Desktop/Hugo_888_code/exp_final/FarField']
    # folder_path = [f'C:/Users/lotanstav/Desktop/Hugo_888_code/FF_10Hz_250mW_final']
    fpath = os.path.join(folder_path[0], 'FarField_partial_50.mat')
    I4D_K = I4D_Edit.I4D_calc(fpath, 121, 121, normalize=True)
    # 90x90
    I4D_window = I4D_Edit.extract_ROI(I4D_K, (21, 111), (5, 95))
    # # 30x30
    # I4D_window = I4D_Edit.extract_ROI(I4D_K, (52, 82), (37, 67))
    run = wandb.init()  # ✅ must come before accessing wandb.config
    config = wandb.config
    finding_hyperparameters(I4D_window, config.lambda_val, config.learning_rate,  int(config.max_iter))


# === SWEEP CONFIGURATION ===
sweep_config = {
    "method": "bayes",  # or "random" if you prefer
    "metric": {"name": "score", "goal": "minimize"},
    "parameters": {
        "lambda_val": {
            "distribution": "uniform",
            "min": 0,
            "max": 5
        },
        "learning_rate": {
            "distribution": "uniform",
            "min": 0,
            "max": 4
        },
        "max_iter": {
            "values": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        }
    }
}
# === RUN THE SWEEP ===
if __name__ == "__main__":
    # optuna_sweep()
    sweep_id = wandb.sweep(sweep_config, project="90x90_Plus_50s")
    wandb.agent(sweep_id, function=wandb_sweep, count=1000)

    # Nset 550 sigmaX = 2.186, sigmaY = 3.2565