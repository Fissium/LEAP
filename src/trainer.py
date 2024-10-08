import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pkbar
import rootutils
import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.callbacks import ModelCheckpoint  # noqa: E402
from src.utils import BatchAccumulator  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
MoveToType = Any


class Trainer:
    """Train neural networks"""

    def __init__(
        self,
        model: nn.Module,
        loss_func: nn.Module,
        loss_func_delta: nn.Module,
        checkpoint_dir: Path,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
        score_funcs: dict[str, Callable] | None = None,
        epochs: int = 50,
        norm_value: float = 1.0,
        resume_training: bool = False,
        resume_training_ckpt: Path | None = None,
        checkpoint_callback: ModelCheckpoint | None = None,
        lr_scheduler: ReduceLROnPlateau | _LRScheduler | None = None,
        postprocessor: Callable | None = None,
        device: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model
            The Pytorch model / "Moduel" to train.
        loss_func
            The loss function.
        loss_func_delta
            The loss function for the first derivative.
        optimizer
            The optimizer.
        train_loader
            Pytorch Dataloader object.
        val_loader
            Optional PyTorch DataLoader to evaluate on after every epoch.
        score_funcs
            A dict of scoring functions to use to evalue the model performance.
        epochs
            The number of training epochs to perform.
        norm_value
            The value to normalize the gradients.
        device
            The compute location to perform training.
        checkpoint_dir
            Path for the checkpoint to be saved.
        resume_training
            Using the chechpoint_file's last state to resume trating.
        resume_training_ckpt
            The checkpoint file to resume training from.
        checkpoint_callback
            A callback to save the model during training.
        lr_scheduler
            Learning rate scheduler.
        postprocessor
            A function to apply to the model output before computing the metrics.

        """
        self.model = model
        self.loss_func = loss_func
        self.loss_func_delta = loss_func_delta
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.score_funcs = score_funcs
        self.epochs = epochs
        self.norm_value = norm_value
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.resume_training = resume_training
        self.resume_training_ckpt = resume_training_ckpt
        self.checkpoint_callback = checkpoint_callback
        self.lr_scheduler = lr_scheduler
        self.postprocessor = postprocessor

        # Create the checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
            },
            self.checkpoint_dir.joinpath("best.ckpt"),
        )

    def move_to_track(self) -> list[str]:
        to_track = ["epoch", "total_time", "train_loss"]
        if self.val_loader is not None:
            to_track.append("val_loss")
        if self.score_funcs is not None:
            for eval_score in self.score_funcs:
                to_track.append("train" + "_" + eval_score)
                if self.val_loader is not None:
                    to_track.append("val" + "_" + eval_score)
        return to_track

    def make_pbar(self, epoch: int) -> pkbar.Kbar:
        return pkbar.Kbar(
            target=len(self.train_loader),
            epoch=epoch,
            num_epochs=self.epochs,
            width=10,
            always_stateful=False,
        )

    def resume_traning_from_ckpt(self, results: dict) -> tuple[dict, int]:
        last_epoch = -1
        if self.resume_training and self.resume_training_ckpt is not None:
            if self.resume_training_ckpt.exists():
                checkpoint_dict = torch.load(
                    self.resume_training_ckpt,
                    map_location=self.device,
                )
                logger.info(f"Resume training using {self.resume_training_ckpt} file")

            else:
                msg = "Using 'resume_traning' you must provide the existing model file"
                raise OSError(
                    msg,
                )
            self.model.load_state_dict(checkpoint_dict["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            last_epoch = checkpoint_dict["epoch"]
            results = checkpoint_dict["results"]
            if last_epoch >= self.epochs:
                msg = "Epochs must be greater than the epoch param in your state_dict"
                raise ValueError(
                    msg,
                )
        return results, last_epoch

    def train(self) -> pd.DataFrame:
        to_track = self.move_to_track()
        total_train_time = 0.0  # How long have we spent in the training loop?
        # Initialize every item with an empty list
        results: dict[str, list[Any]] = {item: [] for item in to_track}

        # Place the model on the correct compute resource (CPU or GPU)
        self.model.to(self.device)

        # Resume training from checkpoint file
        results, last_epoch = self.resume_traning_from_ckpt(results)

        for epoch in range(last_epoch + 1, self.epochs):
            model = self.model.train()  # Put our model in training mode

            pbar = self.make_pbar(epoch)

            run_time, train_metrics = self.run_epoch(
                model,
                self.optimizer,
                self.norm_value,
                self.train_loader,
                self.loss_func,
                self.loss_func_delta,
                self.device,
                results,
                self.score_funcs,
                pbar=pbar,
                prefix="train",
            )

            total_train_time += run_time

            results["total_time"].append(total_train_time)
            results["epoch"].append(epoch)
            metrics = train_metrics.copy()

            if self.val_loader is not None:
                model = model.eval()
                with torch.no_grad():
                    _, val_metrics = self.run_epoch(
                        self.model,
                        self.optimizer,
                        self.norm_value,
                        self.val_loader,
                        self.loss_func,
                        self.loss_func_delta,
                        self.device,
                        results,
                        self.score_funcs,
                        pbar=pbar,
                        prefix="val",
                    )
                    metrics["val_loss"] = results["val_loss"][-1]
                    metrics.update(val_metrics)

                    if self.checkpoint_callback is not None:
                        self.checkpoint_callback(
                            model=model, val_score=metrics["val_r2"]
                        )

                    if self.lr_scheduler is not None:
                        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                            self.lr_scheduler.step(metrics["val_r2"])
                        else:
                            self.lr_scheduler.step()
                    metrics.update({"lr": self.optimizer.param_groups[0]["lr"]})

            pbar.add(
                1,
                values=list(metrics.items()),
            )

        return pd.DataFrame.from_dict(results)

    def run_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        norm_value: float,
        data_loader: torch.utils.data.DataLoader,
        loss_func: nn.Module,
        loss_func_delta: nn.Module,
        device: str,
        results: dict[str, list],
        score_funcs: dict[str, Callable] | None,
        pbar: pkbar.Kbar | None,
        prefix: Literal["train", "val"],
    ) -> tuple[float, dict[str, float]]:
        """
        Parameters
        ----------
        model
            The PyTorch model / "Module" to run for one epoch.
        optimizer
            The Pytorch optimizer.
        norm_value
            The value to normalize the gradients.
        data_loader
            DataLoader object that returns tuples of (input, label) pairs.
        loss_func
            The loss function.
        loss_func_delta
            The loss function for the first derivative.
        device
            The compute lodation to perform training.
        results
            A dict where to store results.
        score_funcs
            A dict of scoring functions to use to evalue the model performance
        pbar
            A progress bar object.
        prefix
            A string to pre-fix to any scores placed into the _results_ dictionary

        Returns
        -------
        tuple[float, dict[str, float]]
            Epoch training time, results
        """

        batch_accumulator = BatchAccumulator(
            data_loader=data_loader, is_training=model.training
        )

        running_loss = []
        metrics = {}
        start = time.time()
        for i, (inputs, y, y_delta) in enumerate(data_loader):
            # Move the batch to the device we are using.
            inputs = inputs.to(device)
            y = y.to(device)
            y_delta = y_delta.to(device)

            y_hat, y_hat_delta = model(inputs)  # this just computed f_Θ(x(i))
            # Compute loss.
            loss_ = loss_func(y_hat, y)
            loss_delta_first = loss_func_delta(y_hat_delta, y_delta)

            loss = loss_ + loss_delta_first

            if model.training:
                loss.backward()
                # Clip the gradients
                nn.utils.clip_grad_norm_(model.parameters(), norm_value, norm_type=2.0)  # type: ignore
                # Update the parameters
                optimizer.step()
                optimizer.zero_grad()

            # Now we are just grabbing some information we would like to have
            running_loss.append(loss.item())

            if pbar is not None and prefix == "train":
                pbar.update(i, values=[("loss", loss)])

            if score_funcs is not None and isinstance(y, torch.Tensor):
                # moving labels & predictions back to CPU
                y = y.detach().cpu().numpy()
                y_hat = y_hat.detach().cpu().numpy()

                batch_accumulator.update(y_true=y, y_pred=y_hat, index=i)

        if self.postprocessor is not None:
            batch_accumulator.postprocess(self.postprocessor)

        # end training epoch
        end = time.time()

        results[prefix + "_" + "loss"].append(np.mean(running_loss))
        if score_funcs is not None:
            for name, score_func in score_funcs.items():
                score = score_func(batch_accumulator.y_true, batch_accumulator.y_pred)
                if prefix == "train":
                    metrics[name] = score
                else:
                    metrics[prefix + "_" + name] = score
                try:
                    results[prefix + "_" + name].append(score)
                except KeyError:
                    results[prefix + "_" + name].append(float("NaN"))
        return end - start, metrics
