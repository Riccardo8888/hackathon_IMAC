import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple


# - kernel=2
# - 10 blocks, dilations 1..512
def dilations_1s_context() -> List[int]:
    # RF = 1 + sum(dilations) = 1000 samples @ 1000 Hz
    return [1, 2, 4, 8, 16, 32, 64, 128, 256,
            256, 128, 64, 32, 8]


def receptive_field(kernel_size: int, dilations: Sequence[int]) -> int:
    return 1 + (kernel_size - 1) * int(np.sum(dilations))