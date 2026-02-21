import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.wavenet import WaveNetCategorical
from util.datasets import RandomWaveNetSegments
from util.metrics import metrics_1d
from util.quantization import mu_law_encode_np, mu_law_decode_np


def train_model(
    model: nn.Module,
    train_ds: RandomWaveNetSegments,
    val_loader: DataLoader,
    rf: int,
    cfg,
    device: torch.device,
    logger: logging.Logger,
    wb_run=None,
    ckpt_dir: Optional[Path] = None,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=float(cfg.lr_decay_gamma))

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    bad = 0

    last_path = None
    best_path = None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        last_path = ckpt_dir / "last.pt"
        best_path = ckpt_dir / "best.pt"

    if last_path is not None and last_path.exists():
        logger.info(f"Resuming from checkpoint: {last_path}")
        bundle = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(bundle["model"])
        opt.load_state_dict(bundle["opt"])
        if bundle.get("sched") is not None:
            try:
                sched.load_state_dict(bundle["sched"])
            except Exception as e:
                logger.info(f"Could not load scheduler state (restart sched): {e}")
        start_epoch = int(bundle.get("epoch", 0)) + 1
        best_val = float(bundle.get("best_val", best_val))
        best_epoch = int(bundle.get("best_epoch", best_epoch))
        bad = int(bundle.get("bad", bad))
        logger.info(f"Resume @ epoch={start_epoch} | best_val={best_val:.6e} (epoch={best_epoch}) | bad={bad}")

    for ep in range(start_epoch, cfg.epochs + 1):
        t0 = time.perf_counter()

        train_ds.resample()
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)

        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)

            logits = model(xb)
            logits_v = logits[:, :, rf:]
            y_v = yb[:, rf:]

            loss = F.cross_entropy(logits_v.permute(0, 2, 1).reshape(-1, cfg.n_bins), y_v.reshape(-1))
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                logits_v = logits[:, :, rf:]
                y_v = yb[:, rf:]
                loss = F.cross_entropy(logits_v.permute(0, 2, 1).reshape(-1, cfg.n_bins), y_v.reshape(-1))
                va_losses.append(float(loss.item()))

        tr = float(np.mean(tr_losses)) if tr_losses else float("inf")
        va = float(np.mean(va_losses)) if va_losses else float("inf")
        lr_now = float(opt.param_groups[0]["lr"])
        dt = time.perf_counter() - t0

        logger.info(
            f"epoch {ep:03d}/{cfg.epochs} | train_ce={tr:.6e} | val_ce={va:.6e} | lr={lr_now:.3e} | dt={dt:.2f}s"
        )
        if wb_run is not None:
            wb_run.log({"epoch": ep, "train_ce": tr, "val_ce": va, "lr": lr_now, "epoch_time_s": dt}, step=ep)

        improved = (va + 1e-12) < best_val
        if improved:
            best_val = va
            best_epoch = ep
            bad = 0
            if best_path is not None:
                torch.save(
                    {
                        "epoch": ep,
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "bad": bad,
                    },
                    best_path,
                )
        else:
            bad += 1

        sched.step()

        if last_path is not None:
            torch.save(
                {
                    "epoch": ep,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "bad": bad,
                },
                last_path,
            )

        if bad >= int(cfg.early_stop_patience):
            logger.info(
                f"early stop @ epoch={ep} (no val improvement for {cfg.early_stop_patience}). best_epoch={best_epoch} best_val={best_val:.6e}"
            )
            break

    if best_path is not None and best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"])

    logger.info(f"train done | best_val_ce={best_val:.6e} @ epoch={best_epoch}")
    if wb_run is not None:
        wb_run.summary["best_val_ce"] = float(best_val)
        wb_run.summary["best_epoch"] = int(best_epoch)

    return model

class FastWaveNetGenerator:
    """Efficient sample-by-sample generation for this specific WaveNet.

    Instead of running full forward on a length-(RF+1) sequence every time, we keep
    per-layer circular buffers to get x[t-d] for each dilation.

    Assumes:
      - kernel_size = 2
      - in/out channels are square (n_filters)
    """

    def __init__(self, model: WaveNetCategorical):
        self.m = model
        if int(model.kernel_size) != 2:
            raise ValueError("FastWaveNetGenerator currently assumes kernel_size=2")

        self.K = int(model.n_bins)
        self.F = int(model.n_filters)
        self.dils = list(model.dils)

    def init_state(self, batch_size: int, device: torch.device):
        bufs = []
        pos = []
        for d in self.dils:
            bufs.append(torch.zeros(batch_size, self.F, int(d), device=device))
            pos.append(0)
        return {"bufs": bufs, "pos": pos}

    def clone_state(self, state):
        return {"bufs": [b.clone() for b in state["bufs"]], "pos": list(state["pos"])}

    @torch.no_grad()
    def step(self, state, in_label: torch.Tensor) -> torch.Tensor:
        """One autoregressive step.

        Args:
          state: generator state
          in_label: (B,) long tensor, the current input label (shift-right)

        Returns:
          logits: (B, K) for the next sample
        """
        if in_label.dtype != torch.long:
            in_label = in_label.long()
        B = in_label.shape[0]

        # front 1x1 (one-hot * W + b)
        Wf = self.m.front_1x1.weight.squeeze(-1)  # (F,K)
        bf = self.m.front_1x1.bias  # (F,)
        x = Wf[:, in_label].permute(1, 0).contiguous() + bf  # (B,F)

        skip_sum = torch.zeros(B, self.F, device=x.device, dtype=x.dtype)

        # blocks
        for i, blk in enumerate(self.m.blocks):
            d = int(self.dils[i])
            buf = state["bufs"][i]
            p = int(state["pos"][i])

            x_in = x
            x_past = buf[:, :, p].clone()
            buf[:, :, p] = x_in
            state["pos"][i] = (p + 1) % d


            wf = blk.filt.conv.weight
            bfilt = blk.filt.conv.bias
            wg = blk.gate.conv.weight
            bgate = blk.gate.conv.bias

            wf0 = wf[:, :, 0]
            wf1 = wf[:, :, 1]
            wg0 = wg[:, :, 0]
            wg1 = wg[:, :, 1]

            f = F.linear(x_past, wf0) + F.linear(x_in, wf1) + bfilt
            g = F.linear(x_past, wg0) + F.linear(x_in, wg1) + bgate
            z = torch.tanh(f) * torch.sigmoid(g)

            wskip = blk.skip_1x1.weight.squeeze(-1)
            bskip = blk.skip_1x1.bias
            wres = blk.res_1x1.weight.squeeze(-1)
            bres = blk.res_1x1.bias

            skip = F.linear(z, wskip) + bskip
            res = F.linear(z, wres) + bres

            x = x_in + res
            skip_sum = skip_sum + skip

        x = F.relu(skip_sum)
        w1 = self.m.post_1.weight.squeeze(-1)
        b1 = self.m.post_1.bias
        x = F.relu(F.linear(x, w1) + b1)

        w2 = self.m.post_2.weight.squeeze(-1)
        b2 = self.m.post_2.bias
        logits = F.linear(x, w2) + b2
        return logits


@torch.no_grad()
def decode_next_from_logits(
        logits: torch.Tensor,
        decode: str,
        centers_np: np.ndarray,
        centers_t: torch.Tensor,
        cfg,
) -> Tuple[int, float]:
    probs = torch.softmax(logits[0], dim=-1)
    if decode == "argmax":
        q = int(torch.argmax(probs).item())
        x = float(centers_np[q])
        return q, x

    x = float(torch.sum(probs * centers_t).item())
    q = int(np.round((x - cfg.amp_min) / (cfg.amp_max - cfg.amp_min) * (cfg.n_bins - 1)))
    q = int(np.clip(q, 0, cfg.n_bins - 1))
    return q, x


# Post-cue window: predict next H samples right after cue
@torch.no_grad()
def predict_sampled_window(model, ep_1d, cfg, device, n_paths=10, temp=0.5):
    sfreq = cfg.sfreq
    cue_idx = int(round(cfg.cue_time_s * sfreq))
    H = int(round(cfg.horizon_s * sfreq))
    rf = cfg.receptive_field

    context = ep_1d[max(0, cue_idx - rf):cue_idx]
    if len(context) < rf:
        context = np.pad(context, (rf - len(context), 0), mode='constant')
    q_hist = mu_law_encode_np(context, mu=cfg.n_bins - 1, amp_max=cfg.amp_max)

    gen = FastWaveNetGenerator(model)
    model.eval()

    state = gen.init_state(batch_size=n_paths, device=device)

    logits = None
    for q in [0] + q_hist.tolist():
        q_batch = torch.full((n_paths,), q, device=device, dtype=torch.long)

        logits = gen.step(state, q_batch)

    path_results = torch.zeros((H, n_paths), device=device)

    for k in range(H):
        probs = torch.softmax(logits / temp, dim=-1)

        q_next = torch.multinomial(probs, 1).squeeze(-1)  # (10,)

        path_results[k, :] = q_next.float()

        logits = gen.step(state, q_next)

    path_results_np = path_results.cpu().numpy()  # (H, 10)
    decoded_paths = mu_law_decode_np(path_results_np, mu=cfg.n_bins - 1, amp_max=cfg.amp_max)

    y_pred = np.mean(decoded_paths, axis=1)
    y_true = ep_1d[cue_idx: cue_idx + H]

    return y_true, y_pred


def eval_postcue_window(
        model: WaveNetCategorical,
        epochs_1d: Sequence[np.ndarray],
        cfg,
        device: torch.device,
        decode: str = "expected",
) -> Dict[str, float]:
    yts, yps = [], []
    total = len(epochs_1d)
    lg = logging.getLogger("eeg_wavenet")

    lg.info(f"Starting eval on {total} epochs...")
    for i, ep in enumerate(epochs_1d):
        yt, yp = predict_sampled_window(model, ep, cfg, device, n_paths=10, temp=0.5)
        yts.append(yt)
        yps.append(yp)

        if (i + 1) % 20 == 0:
            lg.info(f"  [Post-Cue] Processed {i + 1}/{total} epochs...")

    return metrics_1d(np.concatenate(yts), np.concatenate(yps))


# Streaming latency-comp: for many t, predict x(t+h)
@torch.no_grad()
def eval_streaming_latency_comp(
        model: WaveNetCategorical,
        epochs_1d: Sequence[np.ndarray],
        cfg,
        device: torch.device,
        decode: str = "expected",
        post_cue_s: float = 3.1,
        step_ms: float = 20.0,
        max_epochs: Optional[int] = 200,
) -> Dict[str, float]:
    sfreq = float(cfg.sfreq)
    cue_idx = int(round(cfg.cue_time_s * sfreq))
    H = int(round(cfg.horizon_s * sfreq))
    step = int(round((step_ms / 1000.0) * sfreq))

    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    epochs_use = list(epochs_1d)[:max_epochs] if max_epochs else list(epochs_1d)
    model.eval()

    for ei, ep in enumerate(epochs_use, start=1):
        ep = np.asarray(ep, dtype=np.float64)
        T = len(ep)
        post_len = int(round(post_cue_s * sfreq))
        t_end = min(cue_idx + post_len - 1 - H, T - 1 - H)
        if t_end <= cue_idx: continue

        t_idxs = set(range(cue_idx, t_end + 1, step))

        q_all = mu_law_encode_np(ep, mu=cfg.n_bins - 1, amp_max=cfg.amp_max)

        gen = FastWaveNetGenerator(model)
        state = gen.init_state(batch_size=1, device=device)

        logits = gen.step(state, torch.tensor([0], device=device))

        for t in range(0, t_end + 1):
            logits = gen.step(state, torch.tensor([int(q_all[t])], device=device))

            if t not in t_idxs:
                continue

            state_c = gen.clone_state(state)
            logits_c = logits.clone()
            x_pred = 0

            for k in range(H):
                probs = torch.softmax(logits_c / 0.5, dim=-1)
                q_next = torch.multinomial(probs[0], 1).item()

                x_pred = mu_law_decode_np(q_next, mu=cfg.n_bins - 1, amp_max=cfg.amp_max)
                logits_c = gen.step(state_c, torch.tensor([q_next], device=device))

            y_true_all.append(float(ep[t + H]))
            y_pred_all.append(float(x_pred))

        if ei % 10 == 0:
            logging.getLogger("eeg_wavenet").info(f"Streaming eval: epoch {ei}/{len(epochs_use)}")

    return metrics_1d(np.array(y_true_all), np.array(y_pred_all))

