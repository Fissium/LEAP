from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
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
    batch_size: int = 500_000,
) -> tuple[np.ndarray, ...]:
    weights = pd.read_csv(
        data_dir.joinpath(ss_filename), nrows=1, usecols=range(1, 369)
    ).astype("float32")

    X = np.zeros((n_rows, len(features)), dtype=np.float32)
    y = np.zeros((n_rows, len(targets)), dtype=np.float32)

    reader = pl.scan_csv(
        Path(data_dir).joinpath(train_filename),
        n_rows=n_rows,
    )

    X = np.zeros((n_rows, len(features)), dtype=np.float32)
    y = np.zeros((n_rows, len(targets)), dtype=np.float32)

    for batch_start in range(0, n_rows, batch_size):
        batch_end = min(batch_start + batch_size, n_rows)

        df = reader.slice(batch_start, batch_end - batch_start).collect()

        X[batch_start:batch_end, :] = (
            df.to_pandas()[features].astype("float32").to_numpy()
        )
        y[batch_start:batch_end, :] = (
            df.to_pandas()[targets].astype("float32").to_numpy()
        )

        del df

    train_size = int(len(X) * train_val_split[0])

    X_train, y_train = X[:train_size, :], y[:train_size, :]

    X_val, y_val = X[train_size:, :], y[train_size:, :]

    return X_train, y_train, X_val, y_val, weights.to_numpy()


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize with NumPy arrays.
        """
        assert (
            X.shape[0] == y.shape[0]
        ), "Features and labels must have the same number of samples"
        self.X = X
        self.y = y

    def __len__(self):
        """
        Total number of samples.
        """
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """
        Generate one sample of data.
        """

        x = self.X[index].reshape(1, -1)
        y = self.y[index]

        x_seq = np.concatenate(
            (
                x[:, :360].reshape(6, 60),
                x[:, -180:].reshape(3, 60),
            ),
            axis=0,
        )
        x_scalar = x[:, 360:376].reshape(1, -1)
        x_scalar = np.repeat(x_scalar, 60, axis=1).reshape(16, 60)

        x_seq = np.concatenate((x_seq, x_scalar), axis=0)

        # y is 6 by 60 sequences, get the difference of each sequence
        y_delta_first = (
            np.diff(y[:360].reshape(6, 60), axis=1, prepend=0)
            .reshape(-1)
            .astype(np.float32)
        )

        y_delta_second = (
            np.diff(y_delta_first.reshape(6, 60), axis=1, prepend=0)
            .reshape(-1)
            .astype(np.float32)
        )

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(y),
            torch.from_numpy(y_delta_first),
            torch.from_numpy(y_delta_second),
        )
