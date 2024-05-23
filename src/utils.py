import logging
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


TRICK_INDXS = [
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
]


def noise(x: np.ndarray, y: np.ndarray, sigma=0.01):
    gauss = np.random.normal(0, sigma, x.shape).astype(np.float32)
    x_ = x.copy()
    x_ = x + gauss
    return x_, y


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_features_targets(
    data_dir: Path, train_filename: str
) -> tuple[list[str], list[str]]:
    columns = pd.read_csv(data_dir.joinpath(train_filename), nrows=1).columns.to_list()
    features = columns[1:557]
    targets = columns[557:]
    return features, targets


def postprocessor(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_scaler,
    indxs: list[int] = TRICK_INDXS,
) -> Callable:
    def inner(
        y_pred: np.ndarray, y_true: np.ndarray, training: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        y_pred = y_scaler.inverse_transform(y_pred)
        y_true = y_scaler.inverse_transform(y_true)
        if training:
            y_pred[:, indxs] = -X_train / 1200
        else:
            y_pred[:, indxs] = -X_val / 1200
        scores = r2_score(y_true, y_pred, multioutput="raw_values")

        for idx, score in enumerate(scores):  # type: ignore
            if score <= 0:
                y_pred[:, idx] = 0

        return y_pred, y_true

    return inner


class EarlyStopping:
    """Early stops the training if val loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0,
        on_each_epoch: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        patience
            How long to wait after last time validation loss improved.
        verbose
            If True, prints a massage for each val loss improvement.
        delta
            Min Change in the monitored quality.
        path
            Path for the checkpoint to be saved.
        trace_func
            Trace print function.
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop: bool = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.on_each_epoch = on_each_epoch

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        epoch: int,
        path: Path,
    ) -> None:
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                f"\nEarlyStopping counter: {self.counter} out of {self.patience}",
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, path)
            self.counter = 0

    def save_checkpoint(
        self,
        val_loss: float,
        model: nn.Module,
        epoch: int,
        path: Path,
    ) -> None:
        """Saves model when validation loss decrease."""
        if self.verbose:
            logger.info(
                f"\nValidation metric increased ({self.val_loss_min:.6f} -->"
                f" {val_loss:.6f}).  Saving model ...",
            )
        if self.on_each_epoch:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                },
                path.joinpath(f"epoch_{epoch}.ckpt"),
            )
        else:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                },
                path.joinpath("best.ckpt"),
            )

        self.val_loss_min = val_loss
