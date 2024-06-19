import logging
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rootutils  # type: ignore
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.const import MAGIC_INDEXES  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def add_features(X: np.ndarray) -> np.ndarray:
    X_seq = np.concatenate(
        (
            X[:, :360].reshape(X.shape[0], 6, 60),
            X[:, 376 : 376 + 180].reshape(X.shape[0], 3, 60),
        ),
        axis=1,
    )

    X_seq_delta = np.diff(
        X_seq,
        axis=-1,
        prepend=0,
    ).astype(np.float32)

    X_scalar = np.pad(
        X[:, 360:376],
        ((0, 0), (0, 60 - 16)),
        mode="constant",
        constant_values=0,
    ).reshape(X.shape[0], 1, -1)

    return np.concatenate((X_seq, X_seq_delta, X_scalar), axis=1)


class XScaler:
    def __init__(self, min_std: float = 1e-8):
        self.mean: np.ndarray  # type: ignore
        self.std: np.ndarray  # type: ignore
        self.min_std = min_std

    def fit(self, X: np.ndarray) -> None:
        self.mean = X.mean(axis=0)
        self.std = np.maximum(X.std(axis=0), self.min_std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = (X - self.mean.reshape(1, X.shape[1], X.shape[2])) / self.std.reshape(
            1, X.shape[1], X.shape[2]
        )
        return X


class YScaler:
    def __init__(self, min_std: float = 1e-15, std: np.ndarray | None = None):
        if std is not None:
            self.std = std
        else:
            self.std = None
        self.min_std = min_std

    def fit(self, y: np.ndarray) -> None:
        if self.std is None:
            self.std = np.maximum(y.std(axis=0), self.min_std)

    def transform(self, y: np.ndarray) -> np.ndarray:
        return y / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y * self.std


def postprocessor(
    X_magic: np.ndarray,
    y_scaler: YScaler,
    weights: np.ndarray,
    idxs: list[int] = MAGIC_INDEXES,
) -> Callable:
    def inner(
        y_pred: np.ndarray, y_true: np.ndarray, is_traning: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        y_pred = y_scaler.inverse_transform(y_pred) * weights
        y_true = y_scaler.inverse_transform(y_true) * weights
        if not is_traning:
            y_pred[:, idxs] = -X_magic / 1200
        # skip nan values
        y_pred = np.nan_to_num(y_pred)

        scores = r2_score(y_true, y_pred, multioutput="raw_values")

        for idx, score in enumerate(scores):  # type: ignore
            if score <= 0:
                y_pred[:, idx] = 0

        return y_pred, y_true

    return inner


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):  # type: ignore
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


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


class BatchAccumulator:
    """
    Accumulate the y_true and y_pred values for each batch
    Note: only used for evaluation
    """

    def __init__(
        self, data_loader: torch.utils.data.DataLoader, is_training: bool = False
    ):
        drop_last: bool = data_loader.drop_last
        num_samples: int = len(data_loader.dataset)  # type: ignore
        output_size: int = 368
        batch_size: int = (
            data_loader.batch_size if data_loader.batch_size is not None else 1
        )
        self.is_training = is_training

        if not is_training:
            self.y_true = np.zeros(
                (
                    num_samples
                    if not drop_last
                    else batch_size * (num_samples // batch_size),
                    output_size,
                )
            )
            self.y_pred = np.zeros(
                (
                    num_samples
                    if not drop_last
                    else batch_size * (num_samples // batch_size),
                    output_size,
                )
            )
        else:
            self.y_true = np.zeros((batch_size, output_size))
            self.y_pred = np.zeros((batch_size, output_size))

    def update(self, y_true: np.ndarray, y_pred: np.ndarray, index: int) -> None:
        if not self.is_training:
            adj_batch_size = y_true.shape[0]
            self.y_pred[
                index * adj_batch_size : index * adj_batch_size + adj_batch_size
            ] = y_pred
            self.y_true[
                index * adj_batch_size : index * adj_batch_size + adj_batch_size
            ] = y_true

    def postprocess(self, postprocessor: Callable) -> None:
        self.y_pred, self.y_true = postprocessor(
            self.y_pred, self.y_true, self.is_training
        )
