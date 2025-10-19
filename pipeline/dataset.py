import torch
from torch.utils.data import Dataset

class QDataset(Dataset):
    """
    Converts a pandas DataFrame to a PyTorch Dataset.
    Normalizes inputs using provided mean and std.
    """
    def __init__(self, df, mean=None, std=None, normalize=True):
        self.X = df[["W_um", "Lv_um", "Lh_um", "Lc_um", "freq_GHz", "L_pH"]].values.astype("float32")
        self.y = df["Q"].values.astype("float32")
        self.normalize = normalize

        if self.normalize:
            if mean is None or std is None:
                raise ValueError("Must provide mean and std if normalize=True")
            self.mean = mean
            self.std = std
            self.X = (self.X - self.mean) / self.std

        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
