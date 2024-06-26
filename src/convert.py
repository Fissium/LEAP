import gc
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base="1.3", config_path="../configs", config_name="convert.yaml")
def main(cfg: DictConfig):
    reader = pl.scan_csv(
        Path(cfg.dataset_root).joinpath(cfg.train_filename),
        n_rows=cfg.n_rows,
    )
    # Create a directory for the batch if it doesn't exist
    Path(cfg.dataset_root).joinpath("train").mkdir(parents=True, exist_ok=True)

    for batch_start in tqdm(
        range(0, cfg.n_rows, cfg.batch_size), total=cfg.n_rows // cfg.batch_size
    ):
        batch_end = min(batch_start + cfg.batch_size, cfg.n_rows)

        data = (
            reader.slice(batch_start, batch_end - batch_start)
            .select(pl.col("*").exclude("sample_id"))
            .collect(streaming=True)
            .to_numpy()
        )

        # Save the batch
        np.save(
            Path(cfg.dataset_root)
            .joinpath("train")
            .joinpath(f"batch_{batch_start}_{batch_end}.npy"),
            data,
        )

        del data
        gc.collect()


if __name__ == "__main__":
    main()
