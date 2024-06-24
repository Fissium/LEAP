import gc
from pathlib import Path

import numpy as np
import polars as pl
import rootutils
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.const import MAGIC_INDEXES  # noqa: E402
from src.utils import add_features  # noqa: E402


def make_scalers(
    reader: pl.LazyFrame, n_rows: int, batch_size: int
) -> tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    for batch_start in range(0, n_rows, batch_size):
        batch_end = min(batch_start + batch_size, n_rows)

        data = (
            reader.slice(batch_start, batch_end - batch_start)
            .collect()
            .to_pandas()
            .iloc[:, 1:]
            .to_numpy()
        )

        X_batch = add_features(data[:, :556]).reshape(data.shape[0], -1)
        y_batch = data[:, 556:]

        x_scaler.partial_fit(X_batch)
        y_scaler.partial_fit(y_batch)

        del data, X_batch, y_batch
        gc.collect()

    return x_scaler, y_scaler


def read_data(
    data_dir: Path,
    train_filename: str,
    ss_filename: str,
    num_targets: int,
    num_features: int,
    n_rows: int,
    train_val_split: tuple[float, float],
    batch_size: int = 2_000_000,
    seed: int = 42,
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

    reader = pl.scan_csv(
        Path(data_dir).joinpath(train_filename),
        n_rows=n_rows,
    )

    x_scaler, y_scaler = make_scalers(reader, n_rows, batch_size)

    X = np.zeros((n_rows, num_features * 60), dtype=np.float32)
    X_magic = np.zeros((n_rows, len(MAGIC_INDEXES)), dtype=np.float64)
    y = np.zeros((n_rows, num_targets), dtype=np.float32)
    y_init = np.zeros((n_rows, num_targets), dtype=np.float64)

    for batch_start in range(0, n_rows, batch_size):
        batch_end = min(batch_start + batch_size, n_rows)

        data = (
            reader.slice(batch_start, batch_end - batch_start)
            .collect()
            .select(pl.col("*").exclude("sample_id"))
            .to_numpy()
        )

        X_batch = data[:, :556]
        y_batch = data[:, 556:]

        X_magic[batch_start:batch_end, :] = data[:, MAGIC_INDEXES]

        y_init[batch_start:batch_end, :] = y_batch

        X[batch_start:batch_end, :] = x_scaler.transform(
            add_features(X_batch).reshape(X_batch.shape[0], -1)
        ).astype(np.float32)  # type: ignore

        y[batch_start:batch_end, :] = y_scaler.transform(y_batch).astype(np.float32)  # type: ignore

        del data, X_batch, y_batch
        gc.collect()

    (
        X_train,
        X_val,
        y_train,
        y_val,
        _,
        X_val_magic,
        _,
        y_val_init,
    ) = train_test_split(
        X,
        y,
        X_magic,
        y_init,
        test_size=train_val_split[1],
        random_state=seed,
        shuffle=False,
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        weights,
        X_val_magic,
        y_val_init,
        x_scaler,
        y_scaler,
    )  # type: ignore


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
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(y_delta_first),
            torch.from_numpy(y_delta_second),
        )
