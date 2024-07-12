import gc
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import rootutils
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.const import MAGIC_INDEXES  # noqa: E402
from src.utils import add_features  # noqa: E402


def make_scalers(
    n_rows: int | Literal["all"], data_dir: Path, batch_size: int = 200_000
) -> tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # load data from npy files
    if n_rows == "all":
        files = sorted(
            data_dir.joinpath("train").glob("batch_*.npy"),
            key=lambda x: int(x.stem.split("_")[1]),
        )
    else:
        assert n_rows % batch_size == 0, "num_rows must be divisible by batch_size"
        files = sorted(
            data_dir.joinpath("train").glob("batch_*.npy"),
            key=lambda x: int(x.stem.split("_")[1]),
        )[0 : n_rows // batch_size]

    for file in files:
        data = np.load(file)
        X_batch = add_features(X=data[:, :556]).reshape(data.shape[0], -1)
        y_batch = data[:, 556:]

        x_scaler.partial_fit(X_batch)
        y_scaler.partial_fit(y_batch)

        del data, X_batch, y_batch
        gc.collect()

    return x_scaler, y_scaler


def read_data(
    data_dir: Path,
    ss_filename: str,
    num_targets: int,
    num_features: int,
    n_rows: int | Literal["all"],
    train_val_split: tuple[float, float],
    batch_size: int = 200_000,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    StandardScaler,
]:
    weights = (
        pl.read_csv(data_dir.joinpath(ss_filename), n_rows=1, columns=range(1, 369))
        .to_numpy()
        .astype(np.float64)
    )

    x_scaler, y_scaler = make_scalers(
        n_rows=n_rows, batch_size=batch_size, data_dir=data_dir
    )

    if n_rows == "all":
        total_size = 10_091_520
    else:
        assert n_rows % batch_size == 0, "num_rows must be divisible by batch_size"
        total_size = n_rows

    val_size = int(total_size * train_val_split[1])
    train_size = total_size - val_size

    X = np.zeros((total_size, num_features * 60), dtype=np.float32)
    X_val_magic = np.zeros((val_size, len(MAGIC_INDEXES)), dtype=np.float64)
    y = np.zeros((total_size, num_targets), dtype=np.float32)
    y_val_init = np.zeros((val_size, num_targets), dtype=np.float64)

    # load data from npy files
    files = sorted(
        data_dir.joinpath("train").glob("batch_*.npy"),
        key=lambda x: int(x.stem.split("_")[1]),
    )
    if n_rows != "all":
        files = files[: n_rows // batch_size]

    for file in files:
        data = np.load(file)

        batch_start = int(file.stem.split("_")[1])
        batch_end = int(file.stem.split("_")[-1])

        X_batch = data[:, :556]
        y_batch = data[:, 556:]

        X[batch_start:batch_end, :] = x_scaler.transform(
            add_features(X=X_batch).reshape(X_batch.shape[0], -1)
        ).astype(np.float32)  # type: ignore

        y[batch_start:batch_end, :] = y_scaler.transform(y_batch).astype(np.float32)  # type: ignore

        if batch_end <= train_size:
            pass
        elif batch_start < train_size < batch_end:
            # Batch spans both training and validation sets
            train_end = train_size
            val_start = train_size
            X_val_magic[: batch_end - val_start, :] = X_batch[
                train_end - batch_start :, MAGIC_INDEXES
            ]
            y_val_init[: batch_end - val_start, :] = y_batch[train_end - batch_start :]

        else:
            # Entire batch belongs to the validation set
            val_start = batch_start - train_size
            val_end = batch_end - train_size
            X_val_magic[val_start:val_end, :] = X_batch[:, MAGIC_INDEXES]
            y_val_init[val_start:val_end, :] = y_batch

        del data, X_batch, y_batch
        gc.collect()

    return (
        X[:train_size],
        y[:train_size],
        X[train_size:],
        y[train_size:],
        weights,
        X_val_magic,
        y_val_init,
        x_scaler,
        y_scaler,
    )


class LeapDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize with NumPy arrays.
        """
        assert (
            X.shape[0] == y.shape[0]
        ), "Features and labels must have the same number of samples"
        self.X = X
        self.y = y
        self.num_features = X.shape[1]
        assert self.num_features % 60 == 0, "Number of features must be divisible by 60"

    def __len__(self):
        """
        Total number of samples.
        """
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """
        Generate one sample of data.
        """

        x = self.X[index].reshape(self.num_features // 60, 60).transpose(1, 0)
        y = self.y[index]

        # y is 6 by 60 sequences, get the difference of each sequence
        y_delta = np.diff(y[:360].reshape(6, 60), axis=-1, prepend=0)

        y_delta[:, 0] = 0  # first element is always 0

        y_delta = y_delta.flatten().astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(y_delta),
        )
