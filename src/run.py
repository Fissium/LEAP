import logging
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import polars as pl
import rootutils
import torch
import torch.nn as nn
import torch.utils.data
from omegaconf import DictConfig
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.const import MAGIC_INDEXES  # noqa: E402
from src.data import (  # noqa: E402
    NumpyDataset,
    read_data,
)
from src.trainer import Trainer  # noqa: E402
from src.utils import (  # noqa: E402
    XScaler,
    YScaler,
    add_features,
    get_device,
    postprocessor,
    seed_everything,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)


def eval(
    model: nn.Module,
    val_loader: DataLoader,
    X_magic: np.ndarray,
    weights: np.ndarray,
    y_scaler: YScaler,
    log_dir: Path,
    device: str,
) -> np.ndarray:
    state_dict = torch.load(log_dir.joinpath("best.ckpt"))["model_state_dict"]
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for inputs, labels, *_ in tqdm(val_loader):
            y_hat, *_ = model(inputs.to(device))
            preds.append(y_hat.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    preds = y_scaler.inverse_transform(preds) * weights
    targets = y_scaler.inverse_transform(targets) * weights

    r2_scores = r2_score(targets, preds)
    logger.info(f"R2 Score before postprocessing: {r2_scores}")

    preds[:, MAGIC_INDEXES] = -X_magic / 1200  # type: ignore

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
    device: str,
    x_scaler: XScaler,
    y_scaler: YScaler,
    batch_size: int,
    weights: np.ndarray,
) -> np.ndarray:
    state_dict = torch.load(log_dir.joinpath("best.ckpt"))["model_state_dict"]
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    X = pl.read_csv(data_dir.joinpath(test_filename), columns=range(1, 557)).to_numpy()

    X_magic = X[:, MAGIC_INDEXES] / 1200

    X = add_features(X=X.astype(np.float32))

    X = x_scaler.transform(X)

    test_dataset = NumpyDataset(X=X, y=np.zeros((X.shape[0], 368)))

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    preds = []
    with torch.no_grad():
        for inputs, *_ in tqdm(test_loader):
            y_hat, *_ = model(inputs.to(device))
            preds.append(y_hat.detach().cpu().numpy())
    preds = np.concatenate(preds)
    preds = y_scaler.inverse_transform(preds) * weights
    preds[:, MAGIC_INDEXES] = -X_magic / 1200  # type: ignore

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

    y[:, -8:] = np.clip(a=y[:, -8:], a_min=0, a_max=None)

    return y


def prepare_submission(
    data_dir: Path,
    log_dir: Path,
    ss_filename: str,
    submissions_filename: str,
    y_pred: np.ndarray,
) -> None:
    samples_submission = pd.read_csv(data_dir.joinpath(ss_filename))
    samples_submission[samples_submission.columns[1:]] = samples_submission[
        samples_submission.columns[1:]
    ].astype("float32")
    samples_submission.iloc[:, 1:] = y_pred
    samples_submission.to_parquet(log_dir.joinpath(submissions_filename), index=False)

    logger.info(f"Submission file saved to {log_dir.joinpath(submissions_filename)}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="run.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    device = get_device()

    logger.info(f"Lenght of features: {cfg.dataset.num_features}")
    logger.info(f"Lenght of targets: {cfg.dataset.num_targets}")

    logger.info("Loading data...")

    X_train, y_train, X_val, y_val, weights = read_data(
        data_dir=Path(cfg.dataset_root),
        train_filename=cfg.train_filename,
        ss_filename=cfg.ss_filename,
        n_rows=cfg.dataset.n_rows,
        train_val_split=cfg.dataset.train_val_split,
        seed=cfg.seed,
        num_targets=cfg.dataset.num_targets,
        num_features=cfg.dataset.num_features,
    )

    logger.info("Data loaded.")

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"y_val shape: {y_val.shape}")

    X_val_magic = (
        X_val[:, :3, :].reshape(X_val.shape[0], -1)[:, MAGIC_INDEXES]
        * weights[:, MAGIC_INDEXES]
    )

    x_scaler = XScaler()
    x_scaler.fit(X_train)

    X_train = x_scaler.transform(X_train)
    X_val = x_scaler.transform(X_val)

    logger.info("Loadding std of targets..")
    if Path(cfg.dataset_root).joinpath("y_std.npy").exists():
        y_std = np.load(Path(cfg.dataset_root).joinpath("y_std.npy"))
    else:
        logger.info("Calculating std of targets..")
        y_std = None

    y_scaler = YScaler(std=y_std)
    y_scaler.fit(y_train)

    # do not apply scaling to y because y is already scaled (see data.py)

    train_dataset = NumpyDataset(X=X_train, y=y_train)
    val_dataset = NumpyDataset(X=X_val, y=y_val)

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
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.L1Loss()
    criterion_delta_first = nn.L1Loss()
    criterion_delta_second = nn.L1Loss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())()
    lr_scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)()

    early_stopping = hydra.utils.instantiate(cfg.early_stopping)

    trainer = Trainer(
        model=model,
        loss_func=criterion,
        loss_func_delta_first=criterion_delta_first,
        loss_func_delta_second=criterion_delta_second,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.trainer.epochs,
        score_funcs={
            "r2": r2_score,
        },
        device=device,
        checkpoint_dir=Path(cfg.trainer.checkpoint_dir),
        postprocessor=postprocessor(
            X_magic=X_val_magic, y_scaler=y_scaler, weights=weights
        ),
        early_stopping=early_stopping,
        lr_scheduler=lr_scheduler,
        norm_value=cfg.trainer.norm_value,
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
        X_magic=X_val_magic,
        weights=weights,
        y_scaler=y_scaler,
    )

    logger.info("Model evaluated.")

    logger.info("Predicting test data...")

    preds = predict(
        data_dir=Path(cfg.dataset_root),
        log_dir=Path(cfg.hydra_dir),
        test_filename=cfg.test_filename,
        model=model,
        device=device,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        batch_size=cfg.dataset.batch_size,
        weights=weights,
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
