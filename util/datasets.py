from typing import Sequence, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from quantization import mu_law_encode_np


class RandomWaveNetSegments(Dataset):
    def __init__(
        self,
        epochs_1d: Sequence[np.ndarray],
        seq_len: int,
        n_samples: int,
        amp_min: float,
        amp_max: float,
        n_bins: int,
        rng: np.random.Generator,
        fixed_pairs: Optional[List[Tuple[int, int]]] = None,
        max_tries: int = 200000,
    ):
        self.epochs = epochs_1d
        self.seq_len = int(seq_len)
        self.n_samples = int(n_samples)
        self.amp_min = float(amp_min)
        self.amp_max = float(amp_max)
        self.n_bins = int(n_bins)
        self.rng = rng
        self.max_tries = int(max_tries)

        self.pairs: List[Tuple[int, int]] = []
        if fixed_pairs is not None:
            self.pairs = list(fixed_pairs)
            self.n_samples = len(self.pairs)
        else:
            self.resample()

    def resample(self):
        self.pairs = []
        tries = 0
        N = len(self.epochs)

        while len(self.pairs) < self.n_samples and tries < self.max_tries:
            tries += 1
            e = int(self.rng.integers(0, N))
            x = self.epochs[e]
            T = len(x)
            if T <= self.seq_len:
                continue
            s = int(self.rng.integers(0, T - self.seq_len))
            seg = x[s:s + self.seq_len]
            if np.any(seg < self.amp_min) or np.any(seg > self.amp_max):
                continue
            self.pairs.append((e, s))

        if len(self.pairs) < self.n_samples:
            raise RuntimeError(
                f"Could only sample {len(self.pairs)}/{self.n_samples} clean segments "
                f"within range [{self.amp_min},{self.amp_max}] after {tries} tries. "
                f"Your amplitude range is probably wrong for your units."
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        e, s = self.pairs[idx]
        seg = self.epochs[e][s:s + self.seq_len].astype(np.float64)
        y = mu_law_encode_np(seg, mu=self.n_bins-1, amp_max=self.amp_max)
        x_in = np.empty_like(y)
        x_in[0] = 0
        x_in[1:] = y[:-1]
        return torch.from_numpy(x_in.astype(np.int64)), torch.from_numpy(y.astype(np.int64))