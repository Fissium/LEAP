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
    batch_size: int = 100_000,
) -> tuple[np.ndarray, ...]:
    weights = pd.read_csv(
        data_dir.joinpath(ss_filename), nrows=1, usecols=range(1, 369)
    ).astype("float32")

    X = np.zeros((n_rows, len(features)), dtype=np.float32)
    y = np.zeros((n_rows, len(targets)), dtype=np.float32)
    offset = 0
    for batch_start in range(0, n_rows, batch_size):
        batch_end = min(batch_start + batch_size, n_rows)
        df = (
            pl.read_csv(
                data_dir.joinpath(train_filename),
                n_rows=batch_end - batch_start,
                columns=range(1, 925),
                row_index_offset=offset,
            )
            .to_pandas()
            .astype("float32")
        )
        X[batch_start:batch_end, :] = df[features].to_numpy()
        y[batch_start:batch_end, :] = df[
            targets
        ].to_numpy() * weights.to_numpy().reshape(1, -1)
        offset += batch_end - batch_start

    train_size = int(len(X) * train_val_split[0])

    X_train, y_train = X[:train_size, :], y[:train_size, :]

    X_val, y_val = X[train_size:, :], y[train_size:, :]

    return X_train, y_train, X_val, y_val, weights.to_numpy()


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

    def __getitem__(self, index) -> tuple[torch.Tensor, ...]:
        """
        Generate one sample of data.
        """

        x = self.x[index].reshape(1, -1)
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
