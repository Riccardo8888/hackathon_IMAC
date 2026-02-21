import os
import json
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import argparse
import numpy as np
from sklearn import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

try:
    import wandb
except Exception:
    wandb = None


def wandb_enabled() -> bool:
    if wandb is None:
        return False
    flag = os.environ.get("WANDB_DISABLED", "").strip().lower()
    return flag not in ("1", "true", "yes")


def wandb_init(cfg_dict: dict, run_name: str):
    if not wandb_enabled():
        return None
    project = os.environ.get("WANDB_PROJECT", "eeg-forecasting")
    entity = os.environ.get("WANDB_ENTITY", None)
    return wandb.init(project=project, entity=entity, name=run_name, config=cfg_dict, reinit=True)


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("eeg_wavenet")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    lg.addHandler(ch)

    fh = logging.FileHandler(out_dir / "run.log")
    fh.setFormatter(fmt)
    lg.addHandler(fh)
    return lg


# Data loading
def load_json_epochs(path: str):
    p = Path(path)
    if p.is_dir():
        cand = p / "data.json"
        if not cand.exists():
            raise FileNotFoundError(f"Directory given but no data.json found: {p}")
        p = cand
    obj = json.loads(p.read_text())
    return obj


def resolve_epoch_path(path: str) -> str:
    """Allow passing 'datasets/name' when file is 'datasets/name.json', or a directory."""
    p = Path(path)
    if p.exists():
        return str(p)
    pj = p.with_suffix(".json")
    if pj.exists():
        return str(pj)
    if p.is_dir():
        return str(p)
    raise FileNotFoundError(f"Could not find dataset at: {path} (also tried {pj})")

def split_epochs(n_epochs: int, train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 0):
    idx = np.arange(n_epochs)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(round(n_epochs * train_frac))
    n_val = int(round(n_epochs * val_frac))
    train_ids = idx[:n_train]
    val_ids = idx[n_train:n_train + n_val]
    test_ids = idx[n_train + n_val:]
    return train_ids, val_ids, test_ids


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False