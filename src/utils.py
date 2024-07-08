import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import rootutils  # type: ignore
import torch
from omegaconf import DictConfig
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.const import MAGIC_INDEXES  # noqa: E402


def dictconfig_to_dict(cfg: DictConfig) -> dict[str, Any]:
    ret = {}

    for k in cfg.keys():
        if isinstance(cfg[k], DictConfig) or isinstance(cfg[k], dict):
            ret[k] = dictconfig_to_dict(cfg[k])
        else:
            ret[k] = cfg[k]

    return ret


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
    )

    X_scalar = np.pad(
        X[:, 360:376],
        ((0, 0), (0, 60 - 16)),
        mode="constant",
        constant_values=0,
    ).reshape(X.shape[0], 1, -1)

    return np.concatenate((X_seq, X_seq_delta, X_scalar), axis=1)


def postprocessor(
    X_magic: np.ndarray,
    y_scaler: StandardScaler,
    weights: np.ndarray,
    y_init: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray, bool], tuple[np.ndarray, np.ndarray]]:
    """
    Postprocessor for the model's predictions (validation only)
    """

    def inner(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        is_traning: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        # skip nan values
        y_pred = np.nan_to_num(y_pred)

        if not is_traning:
            y_pred = y_scaler.inverse_transform(y_pred.astype(np.float64))

            scores = r2_score(
                y_init * weights, y_pred * weights, multioutput="raw_values"
            )
            mask = scores <= 0

            y_pred[:, mask] = 0
            y_pred[:, MAGIC_INDEXES] = -X_magic / 1200

            return y_pred * weights, y_init * weights

        return y_pred, y_true

    return inner


class CosineLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        lr_max: float,
        num_cycles: float = 0.50,
        num_training_steps: int = 50,
        warmup_method: str = "exp",
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.lr_max = lr_max
        self.num_cycles = num_cycles
        self.num_training_steps = num_training_steps
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore
        current_step = self.last_epoch + 1

        if current_step < self.num_warmup_steps:
            if self.warmup_method == "log":
                lr = self.lr_max * 0.10 ** (self.num_warmup_steps - current_step)
            elif self.warmup_method == "exp":
                lr = self.lr_max * 2 ** -(self.num_warmup_steps - current_step)
            else:
                raise NotImplementedError
        else:
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps)
            )
            lr = (
                max(
                    0.0,
                    0.5
                    * (
                        1.0
                        + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)
                    ),
                )
                * self.lr_max
            )

        return [lr for _ in self.optimizer.param_groups]


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)

    return torch.mean(_log_cosh(y_pred - y_true))


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


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
                ),
                dtype=np.float32,
            )
            self.y_pred = np.zeros(
                (
                    num_samples
                    if not drop_last
                    else batch_size * (num_samples // batch_size),
                    output_size,
                ),
                dtype=np.float32,
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
