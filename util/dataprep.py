from pathlib import Path
import logging

import numpy as np

from util.datasets import load_series_from_csv


def build_train_val_lists(
    data_dir: Path,
    value_col: str = "max",
    train_frac: float = 0.8,
    min_len: int = 5000,
    downsample: int = 1,
    max_files: int | None = None,
    logger: logging.Logger | None = None,
):
    csv_files = sorted(data_dir.rglob("*.csv"))
    if max_files is not None:
        csv_files = csv_files[:max_files]

    if logger:
        logger.info(f"Found {len(csv_files)} CSV files under {data_dir.resolve()}")

    tr_list, va_list = [], []
    used, skipped = 0, 0

    for p in csv_files:
        y = load_series_from_csv(p, value_col=value_col)
        if y is None:
            skipped += 1
            continue

        if downsample > 1:
            y = y[::downsample]

        if len(y) < min_len:
            skipped += 1
            continue

        cut = int(train_frac * len(y))
        y_tr, y_va = y[:cut], y[cut:]
        if len(y_va) < max(200, min_len // 4):
            skipped += 1
            continue

        tr_list.append(y_tr.astype(np.float64))
        va_list.append(y_va.astype(np.float64))
        used += 1

        if logger and used % 50 == 0:
            logger.info(f"Loaded {used} series...")

    if logger:
        logger.info(f"Series used: {used} | skipped: {skipped}")

    return tr_list, va_list, csv_files
