import numpy as np
import json
import os

def compute_normalization_stats(df, feature_cols):
    """
    Returns mean and std for the given columns as numpy arrays.
    """
    mean = df[feature_cols].mean().values.astype("float32")
    std = df[feature_cols].std().values.astype("float32") + 1e-8  # prevent div by zero
    return mean, std

def save_normalization_stats(mean, std, path="norm_stats.json"):
    stats = {
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    with open(path, "w") as f:
        json.dump(stats, f, indent=4)

def load_normalization_stats(path="norm_stats.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Normalization file not found: {path}")
    with open(path, "r") as f:
        stats = json.load(f)
    return np.array(stats["mean"], dtype=np.float32), np.array(stats["std"], dtype=np.float32)
