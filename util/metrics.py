import logging
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt

from core.training import predict_sampled_window
from core.wavenet import WaveNetCategorical


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