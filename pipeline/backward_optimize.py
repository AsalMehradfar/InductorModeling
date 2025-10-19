import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.visual_utils import plot_optimization_history


FEATURE_COLS = ["freq_GHz", "W_um", "L_pH", "Lv_um", "Lh_um", "Lc_um"]

def optimize_lv_lh_lc(
    model: nn.Module,
    freq: float,
    W: float,
    L: float,
    mean: np.ndarray,
    std: np.ndarray,
    input_columns: list,
    device: str = "cpu",
    lr: float = 0.01,
    steps: int = 3000
):
    bounds = {
        "Lv": (W+2, 100),
        "Lh": (2*W+4, 100),
        "Lc": (1, 50),
    }
    model.to(device)
    model.eval()

    # Fixed inputs
    fixed = torch.tensor([[freq, W, L]], dtype=torch.float32, device=device)

    # Optimized variables
    lv = torch.tensor([40.0], dtype=torch.float32, requires_grad=True, device=device)
    lh = torch.tensor([40.0], dtype=torch.float32, requires_grad=True, device=device)
    lc = torch.tensor([20.0], dtype=torch.float32, requires_grad=True, device=device)

    optimizer = optim.Adam([lv, lh, lc], lr=lr)

    lv_history = []
    lh_history = []
    lc_history = []
    q_history = []

    for _ in range(steps):
        optimizer.zero_grad()

        x = torch.cat([fixed, lv.unsqueeze(0), lh.unsqueeze(0), lc.unsqueeze(0)], dim=1)

        # Normalize
        x_norm = (x - torch.tensor(mean, dtype=torch.float32, device=device)) / torch.tensor(std, dtype=torch.float32, device=device)

        q_pred = model(x_norm)
        loss = -q_pred.mean()  # Negative since we want to maximize

        loss.backward()
        optimizer.step()

        # Clamp to bounds
        lv.data.clamp_(*bounds["Lv"])
        lh.data.clamp_(*bounds["Lh"])
        lc.data.clamp_(*bounds["Lc"])

        lv_history.append(lv.item())
        lh_history.append(lh.item())
        lc_history.append(lc.item())
        q_history.append(q_pred.item())

    # plot_optimization_history(lv_history, lh_history, lc_history, q_history)

    return {
        "Lv": float(lv.item()),
        "Lh": float(lh.item()),
        "Lc": float(lc.item()),
        "Predicted_Q": float(-loss.item())
    }
