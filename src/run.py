import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn as nn
import torch.utils.data
from omegaconf import DictConfig
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import NumpyDataset, read_data  # noqa: E402
from src.trainer import Trainer  # noqa: E402
from src.utils import (  # noqa: E402
    XScaler,
    get_device,
    get_features_targets,
    postprocessor,
    seed_everything,
)

torch.set_float32_matmul_precision("high")
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    log_dir: Path,
) -> np.ndarray:
    state_dict = torch.load(log_dir.joinpath("best.ckpt"))["model_state_dict"]
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for inputs_seq, inputs_scalar, labels in tqdm(val_loader):
            preds.append(
                model(inputs_seq.to(device), inputs_scalar.to(device))
                .detach()
                .cpu()
                .numpy()
            )
            targets.append(labels.detach().cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    r2_scores = r2_score(targets, preds)
    logger.info(f"R2 Score before postprocessing: {r2_scores}")

    raw_r2_scores = r2_score(targets, preds, multioutput="raw_values")

    for idx, score in enumerate(raw_r2_scores):  # type: ignore
        if score <= 0:
            preds[:, idx] = 0  # type: ignore

    r2_scores = r2_score(targets, preds)
    logger.info(f"R2 Score after postprocessing: {r2_scores}")

    return raw_r2_scores  # type: ignore


def predict(
    data_dir: Path,
    log_dir: Path,
    test_filename: str,
    model: nn.Module,
    device: torch.device,
    features: list[str],
    xscaler: XScaler,
    batch_size: int,
) -> np.ndarray:
    state_dict = torch.load(log_dir.joinpath("best.ckpt"))["model_state_dict"]
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    X = (
        pd.read_csv(data_dir.joinpath(test_filename))[features]
        .astype("float32")
        .to_numpy()
    )

    X = xscaler.transform(X)

    test_dataset = NumpyDataset(X, np.zeros((X.shape[0], 1)))

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    preds = []
    with torch.no_grad():
        for inputs_seq, inputs_scalar, _ in tqdm(test_loader):
            preds.append(
                model(inputs_seq.to(device), inputs_scalar.to(device))
                .detach()
                .cpu()
                .numpy()
            )
    preds = np.concatenate(preds)

    return preds


def postprocessing(
    data_dir: Path,
    ss_filename: str,
    y: np.ndarray,
    raw_r2_scores: np.ndarray,
) -> np.ndarray:
    weights = (
        pd.read_csv(
            data_dir.joinpath(ss_filename),
            nrows=1,
            usecols=range(1, 369),
        )
        .astype("float32")
        .to_numpy()
        .reshape(1, -1)
    )

    for idx, weight in enumerate(weights[0]):
        if weight > 0:
            continue
        else:
            y[:, idx] = 0

    for idx, score in enumerate(raw_r2_scores):  # type:ignore
        if score <= 0:
            y[:, idx] = 0

    return y


def prepare_submission(
    data_dir: Path,
    log_dir: Path,
    ss_filename: str,
    submissions_filename: str,
    y_pred: np.ndarray,
) -> None:
    samples_submission = pd.read_csv(data_dir.joinpath(ss_filename))
    samples_submission.iloc[:, 1:] = y_pred
    samples_submission.to_parquet(log_dir.joinpath(submissions_filename), index=False)

    logger.info(f"Submission file saved to {log_dir.joinpath(submissions_filename)}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="run.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    device = get_device()

    features, targets = get_features_targets(
        data_dir=Path(cfg.dataset_root), train_filename=cfg.train_filename
    )

    logger.info(f"Lenght of features: {len(features)}")
    logger.info(f"Lenght of targets: {len(targets)}")

    logger.info("Loading data...")

    X_train, y_train, X_val, y_val = read_data(
        data_dir=Path(cfg.dataset_root),
        train_filename=cfg.train_filename,
        ss_filename=cfg.ss_filename,
        n_rows=cfg.dataset.n_rows,
        train_val_split=cfg.dataset.train_val_split,
        features=features,
        targets=targets,
    )

    logger.info("Data loaded.")

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"y_val shape: {y_val.shape}")

    xscaler = XScaler()
    xscaler.fit(X_train)

    X_train = xscaler.transform(X_train)
    X_val = xscaler.transform(X_val)

    train_dataset = NumpyDataset(X_train, y_train)
    val_dataset = NumpyDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )

    model = hydra.utils.instantiate(cfg.model)
    criterion = nn.MSELoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())()
    lr_scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)()

    early_stopping = hydra.utils.instantiate(cfg.early_stopping)

    trainer = Trainer(
        model=model,
        loss_func=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.trainer.epochs,
        score_funcs={
            "r2": r2_score,
        },
        device=device,
        checkpoint_dir=Path(cfg.trainer.checkpoint_dir),
        postprocessor=postprocessor,
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
    )
    history = trainer.train()
    history.to_csv(f"{Path(cfg.hydra_dir).joinpath('history.csv')}", index=False)

    logger.info("Training finished.")

    logger.info("Evaluating model...")

    raw_r2_scores = eval(
        model=model,
        val_loader=val_loader,
        device=device,
        log_dir=Path(cfg.hydra_dir),
    )

    logger.info("Model evaluated.")

    logger.info("Predicting test data...")

    preds = predict(
        data_dir=Path(cfg.dataset_root),
        log_dir=Path(cfg.hydra_dir),
        test_filename=cfg.test_filename,
        model=model,
        device=device,
        features=features,
        xscaler=xscaler,
        batch_size=cfg.dataset.batch_size,
    )

    preds = postprocessing(
        data_dir=Path(cfg.dataset_root),
        ss_filename=cfg.ss_filename,
        y=preds,
        raw_r2_scores=raw_r2_scores,
    )

    logger.info("Test data predicted.")

    logger.info("Preparing submission...")

    submissions_filename = "submission.parquet"

    prepare_submission(
        data_dir=Path(cfg.dataset_root),
        log_dir=Path(cfg.hydra_dir),
        ss_filename=cfg.ss_filename,
        submissions_filename=submissions_filename,
        y_pred=preds,
    )


if __name__ == "__main__":
    main()
