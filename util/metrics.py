import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from core.inference import predict_sampled_window
from core.wavenet import WaveNetCategorical
from util import Cfg
from util.quantization import mu_law_decode_np


def metrics_1d(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")
    return {"MSE": mse, "MAE": mae, "Corr": corr, "N": float(len(y_true))}

def save_random_postcue_plots(
    model: WaveNetCategorical,
    epochs_1d: Sequence[np.ndarray],
    cfg,
    device: torch.device,
    out_dir: Path,
    n_plots: int = 10,
    split_name: str = "test",
    seed: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    if len(epochs_1d) == 0:
        logging.getLogger("eeg_wavenet").info(f"[PLOTS] No epochs available for {split_name}. Skipping.")
        return

    n_plots = int(min(n_plots, len(epochs_1d)))
    picks = rng.choice(len(epochs_1d), size=n_plots, replace=False)

    sfreq = float(cfg.sfreq)
    H = int(round(cfg.horizon_s * sfreq))
    t_ms = np.arange(H) * (1000.0 / sfreq)

    for j, ei in enumerate(picks, start=1):
        ep = epochs_1d[int(ei)]
        y_true, y_pred = predict_sampled_window(model, ep, cfg, device, n_paths=10, temp=0.5)

        plt.figure()
        plt.plot(t_ms, y_true, "b", label="True future")
        plt.plot(t_ms, y_pred, "r", label="Pred future")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (V)")
        plt.title(f"{split_name} | epoch_idx={int(ei)} | horizon={int(round(cfg.horizon_s*1000))}ms")
        plt.legend(loc="best")
        plt.tight_layout()

        fname = out_dir / f"{split_name}_rand{j:02d}_epoch{int(ei):05d}_h{int(round(cfg.horizon_s*1000))}ms.png"
        plt.savefig(fname, dpi=150)
        plt.close()

@torch.no_grad()
def eval_teacher_forced_metrics(model: WaveNetCategorical, val_loader: DataLoader, cfg: Cfg, device: torch.device,
                                max_batches: int = 200) -> dict:
    """
    Teacher-forced evaluation:
    - input is the shifted quantized sequence
    - model outputs logits for each timestep
    - we compute expected value (in continuous space) from the predicted categorical distribution
    - compare vs decoded true values
    """
    model.eval()
    rf = int(cfg.receptive_field)

    centers = np.linspace(cfg.amp_min, cfg.amp_max, cfg.n_bins, dtype=np.float64)
    centers_t = torch.tensor(centers, device=device, dtype=torch.float32)  # [K]

    y_true_all = []
    y_pred_all = []

    for bi, (xb, yb) in enumerate(val_loader):
        if bi >= max_batches:
            break

        xb = xb.to(device)  # [B,T] long
        yb = yb.to(device)  # [B,T] long

        logits = model(xb)  # [B,K,T]
        logits_v = logits[:, :, rf:]              # [B,K,T-rf]
        y_v = yb[:, rf:]                          # [B,T-rf]

        probs = torch.softmax(logits_v, dim=1)    # [B,K,T-rf]
        # expected value per timestep: sum_k p(k)*center(k)
        y_pred = (probs * centers_t.view(1, -1, 1)).sum(dim=1)  # [B,T-rf]

        # decode true labels back to continuous values
        y_true = mu_law_decode_np(
            y_v.detach().cpu().numpy(),
            mu=cfg.n_bins - 1,
            amp_max=cfg.amp_max,
        )  # [B,T-rf]
        y_true = y_true.reshape(-1)
        y_pred = y_pred.detach().cpu().numpy().reshape(-1)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    if not y_true_all:
        return {"MSE": float("nan"), "MAE": float("nan"), "Corr": float("nan"), "N": 0.0}

    y_true_cat = np.concatenate(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)
    return metrics_1d(y_true_cat, y_pred_cat)