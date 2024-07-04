import logging
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import rootutils
import torch
import torch.nn as nn
import torch.utils.data
from omegaconf import DictConfig
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.const import MAGIC_INDEXES  # noqa: E402
from src.data import (  # noqa: E402
    LeapDataset,
    read_data,
)
from src.trainer import Trainer  # noqa: E402
from src.utils import (  # noqa: E402
    add_features,
    get_device,
    postprocessor,
    seed_everything,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval(
    model: nn.Module,
    val_loader: DataLoader,
    X_magic: np.ndarray,
    y_init: np.ndarray,
    weights: np.ndarray,
    y_scaler: StandardScaler,
    log_dir: Path,
    device: str,
) -> np.ndarray:
    state_dict = torch.load(log_dir.joinpath("best.ckpt"))["model_state_dict"]
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    preds = []
    with torch.no_grad():
        for inputs, *_ in tqdm(val_loader):
            y_hat, *_ = model(inputs.to(device))
            preds.append(y_hat.detach().cpu().numpy())
    preds = np.concatenate(preds)

    preds = y_scaler.inverse_transform(preds.astype(np.float64))  # type: ignore

    raw_r2_scores = r2_score(
        y_init * weights, preds * weights, multioutput="raw_values"
    )

    mask = raw_r2_scores <= 0
    preds[:, mask] = 0  # type: ignore

    preds[:, MAGIC_INDEXES] = -X_magic / 1200  # type: ignore

    score = r2_score(y_init * weights, preds * weights)
    logger.info(f"R2 Score: {score}")

    return raw_r2_scores  # type: ignore


def predict(
    data_dir: Path,
    log_dir: Path,
    test_filename: str,
    model: nn.Module,
    device: str,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    state_dict = torch.load(log_dir.joinpath("best.ckpt"))["model_state_dict"]
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    X = pl.read_csv(data_dir.joinpath(test_filename), columns=range(1, 557)).to_numpy()

    X_magic = X[:, MAGIC_INDEXES]

    X = add_features(X=X).reshape(X.shape[0], -1)

    X = x_scaler.transform(X).astype(np.float32)  # type: ignore

    test_dataset = LeapDataset(X=X, y=np.zeros((X.shape[0], 368), dtype=np.float32))

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    preds = []
    with torch.no_grad():
        for inputs, *_ in tqdm(test_loader):
            y_hat, *_ = model(inputs.to(device))
            preds.append(y_hat.detach().cpu().numpy())
    preds = np.concatenate(preds)
    preds = y_scaler.inverse_transform(preds.astype(np.float64))  # type: ignore

    return preds, X_magic


def postprocessing(
    y: np.ndarray,
    raw_r2_scores: np.ndarray,
    weights: np.ndarray,
    X_magic: np.ndarray,
) -> np.ndarray:
    mask = raw_r2_scores <= 0
    y[:, mask] = 0

    y[:, MAGIC_INDEXES] = -X_magic / 1200

    return y * weights


def prepare_submission(
    data_dir: Path,
    log_dir: Path,
    ss_filename: str,
    submissions_filename: str,
    y_pred: np.ndarray,
) -> None:
    samples_submission = pl.read_csv(data_dir.joinpath(ss_filename)).to_pandas()
    samples_submission[samples_submission.columns[1:]] = samples_submission[
        samples_submission.columns[1:]
    ].astype("float64")
    samples_submission.iloc[:, 1:] = y_pred
    samples_submission.to_parquet(log_dir.joinpath(submissions_filename), index=False)

    logger.info(f"Submission file saved to {log_dir.joinpath(submissions_filename)}")


def train(
    cfg: DictConfig,
) -> tuple[nn.Module, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    seed_everything(cfg.seed)

    device = get_device()

    logger.info(f"Lenght of features: {cfg.dataset.num_features}")
    logger.info(f"Lenght of targets: {cfg.dataset.num_targets}")

    logger.info("Loading data...")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        weights,
        X_val_magic,
        y_val_init,
        x_scaler,
        y_scaler,
    ) = read_data(
        data_dir=Path(cfg.dataset_root),
        ss_filename=cfg.ss_filename,
        n_rows=cfg.dataset.n_rows,
        train_val_split=cfg.dataset.train_val_split,
        num_targets=cfg.dataset.num_targets,
        num_features=cfg.dataset.num_features,
    )

    logger.info("Data loaded.")

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")

    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"y_val shape: {y_val.shape}")

    train_dataset = LeapDataset(X=X_train, y=y_train)
    val_dataset = LeapDataset(X=X_val, y=y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
    )

    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    criterion = hydra.utils.instantiate(cfg.criterion)
    criterion_delta_first = hydra.utils.instantiate(cfg.criterion_delta_first)
    criterion_delta_second = hydra.utils.instantiate(cfg.criterion_delta_second)
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
            X_magic=X_val_magic,
            weights=weights,
            y_scaler=y_scaler,
            y_init=y_val_init,
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
        y_init=y_val_init,
    )

    logger.info("Model evaluated.")

    return model, raw_r2_scores, weights, y_scaler, x_scaler


@hydra.main(version_base="1.3", config_path="../configs", config_name="run.yaml")
def main(cfg: DictConfig):
    model, raw_r2_scores, weights, y_scaler, x_scaler = train(cfg)

    device = get_device()

    logger.info("Predicting test data...")

    preds, X_test_magic = predict(
        data_dir=Path(cfg.dataset_root),
        log_dir=Path(cfg.hydra_dir),
        test_filename=cfg.test_filename,
        model=model,
        device=device,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        batch_size=cfg.trainer.batch_size,
    )

    preds = postprocessing(
        y=preds,
        raw_r2_scores=raw_r2_scores,
        weights=weights,
        X_magic=X_test_magic,
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
