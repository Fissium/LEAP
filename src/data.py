from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_data(
    data_dir: Path,
    train_filename: str,
    ss_filename: str,
    n_rows: int,
    train_val_split: tuple[float, float],
    features: list[str],
    targets: list[str],
) -> tuple[np.ndarray, ...]:
    df = pd.read_csv(
        data_dir.joinpath(train_filename), nrows=n_rows, usecols=range(1, 925)
    ).astype("float32")

    weights = pd.read_csv(
        data_dir.joinpath(ss_filename), nrows=1, usecols=range(1, 369)
    ).astype("float32")

    weights = weights.to_numpy().reshape(1, -1)

    X = df[features].to_numpy()
    y = df[targets].to_numpy() * weights

    train_size = int(len(X) * train_val_split[0])

    X_train, y_train = X[:train_size, :], y[:train_size, :]

    X_val, y_val = X[train_size:, :], y[train_size:, :]

    return X_train, y_train, X_val, y_val


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        """
        Initialize with NumPy arrays.
        """
        assert (
            x.shape[0] == y.shape[0]
        ), "Features and labels must have the same number of samples"
        self.x = x
        self.y = y

    def __len__(self):
        """
        Total number of samples.
        """
        return self.x.shape[0]

    def __getitem__(self, index):
        """
        Generate one sample of data.
        """

        x = self.x[index].reshape(1, -1)
        # Convert the data to tensors when requested
        x_seq = np.concatenate(
            (
                x[:, :360].reshape(6, 60),
                x[:, -180:].reshape(3, 60),
            ),
            axis=0,
        )
        x_scalar = x[:, 360:376].reshape(-1)
        # make x_scalar shape of (60, 16) filled with the same value
        x_scalar = np.repeat(x_scalar, 60, axis=0).reshape(16, 60)

        x = np.concatenate((x_seq, x_scalar), axis=0)

        return torch.from_numpy(x), torch.from_numpy(self.y[index])
