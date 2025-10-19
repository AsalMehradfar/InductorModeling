import matplotlib.pyplot as plt

def plot_loss_curves(train_losses, val_losses, title="Training Loss", save_path=None, log_scale=False):
    """
    Plots training and validation loss curves.

    Args:
        train_losses (list or array): Training loss values per epoch.
        val_losses (list or array): Validation loss values per epoch.
        title (str): Title of the plot.
        save_path (str or None): If provided, saves the plot to this file.
        log_scale (bool): Whether to use log scale on the y-axis.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if log_scale:
        plt.yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=400, bbox_inches='tight')
        print(f"[INFO] Saved Loss plot to {save_path}")
    plt.show()


def plot_optimization_history(lv_vals, lh_vals, lc_vals, q_vals):
    """
    Plots the evolution of Lv, Lh, Lc, and predicted Q over optimization steps.

    Args:
        lv_vals (list of float): History of Lv values
        lh_vals (list of float): History of Lh values
        lc_vals (list of float): History of Lc values
        q_vals (list of float): History of predicted Q values
    """
    steps = list(range(1, len(lv_vals) + 1))

    plt.figure(figsize=(12, 6))

    # Left: Lv, Lh, Lc
    plt.subplot(1, 2, 1)
    plt.plot(steps, lv_vals, label="Lv")
    plt.plot(steps, lh_vals, label="Lh")
    plt.plot(steps, lc_vals, label="Lc")
    plt.xlabel("Optimization Step")
    plt.ylabel("Value (μm)")
    plt.title("Evolution of Lv, Lh, Lc")
    plt.legend()
    plt.grid(True)

    # Right: Predicted Q
    plt.subplot(1, 2, 2)
    plt.plot(steps, q_vals, label="Predicted Q", color="purple")
    plt.xlabel("Optimization Step")
    plt.ylabel("Q Value")
    plt.title("Predicted Q vs Optimization Step")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("checkpoints/optimization_history.pdf", format='pdf', dpi=400, bbox_inches='tight')
    plt.show()