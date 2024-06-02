from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def read_data(
    data_dir: Path,
    train_filename: str,
    ss_filename: str,
    num_features: int,
    num_targets: int,
    n_rows: int,
    train_val_split: tuple[float, float],
    batch_size: int = 2_000_000,
    seed: int = 42,
) -> tuple:
    weights = pd.read_csv(
        data_dir.joinpath(ss_filename), nrows=1, usecols=range(1, 369)
    ).astype("float32")

    reader = pl.scan_csv(
        Path(data_dir).joinpath(train_filename),
        n_rows=n_rows,
    )

    X = np.zeros((n_rows, num_features), dtype=np.float32)
    y = np.zeros((n_rows, num_targets), dtype=np.float32)

    for batch_start in range(0, n_rows, batch_size):
        batch_end = min(batch_start + batch_size, n_rows)

        df = reader.slice(batch_start, batch_end - batch_start).collect()

        X[batch_start:batch_end, :] = (
            df.to_pandas().iloc[:, 1:557].astype("float32").to_numpy()
        )

        y[batch_start:batch_end, :] = (
            df.to_pandas().iloc[:, 557:].astype("float32").to_numpy()
        ) * weights.to_numpy().reshape(1, -1)

        del df

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=train_val_split[1],
        random_state=seed,
        shuffle=False,
    )

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

        x = self.X[index]
        y = self.y[index]

        # original sequences
        x_seq = np.concatenate(
            (
                x[:360].reshape(6, 60),
                x[376 : 376 + 180].reshape(3, 60),
            ),
            axis=0,
        )
        # velocity sequences
        x_seq_delta_first = np.diff(x_seq, axis=1, prepend=0).astype(np.float32)
        x_scalar = np.pad(
            x[360:376], (0, 60 - 16), mode="constant", constant_values=0
        ).reshape(1, -1)

        x_seq = np.concatenate((x_seq, x_seq_delta_first, x_scalar), axis=0)

        # y is 6 by 60 sequences, get the difference of each sequence
        y_delta_first = (
            np.diff(y[:360].reshape(6, 60), axis=-1, prepend=0)
            .reshape(-1)
            .astype(np.float32)
        )

        y_delta_second = (
            np.diff(y_delta_first.reshape(6, 60), axis=-1, prepend=0)
            .reshape(-1)
            .astype(np.float32)
        )

        return (
            torch.from_numpy(x_seq),
            torch.from_numpy(y),
            torch.from_numpy(y_delta_first),
            torch.from_numpy(y_delta_second),
        )
