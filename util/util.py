import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

def load_epochs_from_npz(path) -> list[np.ndarray]:
    data = np.load(path)
    return [data[k] for k in sorted(data.files)]

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

def read_window_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported window file type: {path}")

def list_window_files(window_path: str | Path, recursive: bool = True) -> list[Path]:
    p = Path(window_path)
    if p.is_file():
        return [p]
    pats = ["*/.parquet", "*/.csv.gz", "*/.csv"] if recursive else [".parquet", ".csv.gz", "*.csv"]
    files = []
    for pat in pats:
        files.extend(p.glob(pat))
    return sorted(set(files))

def load_centered_series_from_window_file(path: Path, value_col: str = "max") -> tuple[np.ndarray, float] | tuple[None, None]:
    """
    Returns:
      y_centered: y - mean(y)   (no scaling)
      mu: mean(y)              (for later un-centering in plots)
    """
    df = read_window_df(path)

    if "utc" in df.columns:
        df["utc"] = pd.to_datetime(df["utc"], errors="coerce")
        df = df.dropna(subset=["utc"]).sort_values("utc")

    if value_col not in df.columns:
        return None, None

    y = pd.to_numeric(df[value_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if y.size < 10:
        return None, None

    mu = float(np.mean(y))
    return (y - mu), mu
