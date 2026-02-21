from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, d: int):
        super().__init__()
        self.k = k
        self.d = d
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.k - 1) * self.d
        x = F.pad(x, (pad, 0))
        return self.conv(x)

class WaveNetBlock(nn.Module):
    def __init__(self, n_filters: int, k: int, d: int):
        super().__init__()
        self.dilation = d
        self.filt = CausalConv1d(n_filters, n_filters, k, d)
        self.gate = CausalConv1d(n_filters, n_filters, k, d)
        self.res_1x1 = nn.Conv1d(n_filters, n_filters, kernel_size=1)
        self.skip_1x1 = nn.Conv1d(n_filters, n_filters, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.tanh(self.filt(x)) * torch.sigmoid(self.gate(x))
        skip = self.skip_1x1(z)
        res = self.res_1x1(z)
        return x + res, skip