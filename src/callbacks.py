import logging
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


class ModelCheckpoint:
    def __init__(
        self,
        filepath: str,
        min_delta: float = 0,
        mode: str = "min",
    ) -> None:
        self.min_delta = min_delta
        self.filepath = filepath
        self.mode = mode
        if self.mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = float("-inf")

    def __call__(
        self,
        val_score: float,
        model: nn.Module,
    ) -> None:
        if self._is_best(val_score):
            self.best_score = val_score
            self._save_checkpoint(model)

    def _is_best(self, current: float) -> bool:
        if self.mode == "min":
            return (self.best_score - current) > self.min_delta
        else:
            return (current - self.best_score) > self.min_delta

    def _save_checkpoint(self, model: nn.Module) -> None:
        if self.mode == "min":
            logger.info(
                f"\nVal score decreased ({self.best_score:.6f}). Saving model ...",
            )
        else:
            logger.info(
                f"\nVal score increased ({self.best_score:.6f}). Saving model ...",
            )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
            },
            Path(self.filepath).joinpath("best.ckpt"),
        )
