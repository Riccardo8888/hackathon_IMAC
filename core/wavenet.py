import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layers import WaveNetBlock
from util.wavenet import dilations_1s_context


class WaveNetCategorical(nn.Module):
    """Input labels (B,T) in [0..K-1], shift-right convention.
    Output logits (B,K,T).
    """

    def __init__(self, n_bins: int = 256, n_filters: int = 16, kernel_size: int = 2):
        super().__init__()
        self.n_bins = n_bins
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dils = dilations_1s_context()

        self.front_1x1 = nn.Conv1d(n_bins, n_filters, kernel_size=1)
        self.blocks = nn.ModuleList([WaveNetBlock(n_filters=n_filters, k=kernel_size, d=d) for d in self.dils])
        self.post_1 = nn.Conv1d(n_filters, n_filters, kernel_size=1)
        self.post_2 = nn.Conv1d(n_filters, n_bins, kernel_size=1)

    def forward(self, x_labels: torch.Tensor) -> torch.Tensor:
        if x_labels.dtype != torch.long:
            x_labels = x_labels.long()
        x_oh = F.one_hot(x_labels, num_classes=self.n_bins).float().permute(0, 2, 1)  # (B,K,T)
        x = self.front_1x1(x_oh)

        skip_sum = 0.0
        for blk in self.blocks:
            x, skip = blk(x)
            skip_sum = skip_sum + skip

        x = F.relu(skip_sum)
        x = F.relu(self.post_1(x))
        logits = self.post_2(x)
        return logits
