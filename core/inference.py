import numpy as np
import torch
from torch.nn import functional as F

from core.wavenet import WaveNetCategorical
from util.quantization import mu_law_encode_np, mu_law_decode_np


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
