from dataclasses import dataclass

@dataclass
class Cfg:
    # data
    sfreq: float = 1000.0
    cue_time_s: float = 1.0

    # evaluation
    context_s: float = 1.0
    horizon_s: float = 0.01

    # quantization range
    amp_min = 0
    amp_max =  15
    n_bins: int = 256

    # architecture
    kernel_size: int = 2
    n_blocks: int = 10
    n_filters: int = 40

    # training
    lr: float = 1e-3
    lr_decay_gamma: float = 0.99
    batch_size: int = 32
    epochs: int = 400

    seq_len: int = 2000
    train_samples_per_epoch: int = 12000
    val_samples_fixed: int = 4000

    early_stop_patience: int = 40
    grad_clip: float = 1.0

    receptive_field: int = 1024

# ----------------------------
# Config
# ----------------------------
@dataclass
class Cfg2:

    def __init__(self):
        pass

    n_bins: int = 256
    kernel_size: int = 2
    n_filters: int = 16

    # seq windows (will be overwritten after RF computed)
    seq_len: int = 256

    # training
    lr: float = 1e-3
    lr_decay_gamma: float = 0.99
    batch_size: int = 32
    epochs: int = 200
    train_samples_per_epoch: int = 8000
    val_samples_fixed: int = 2000
    early_stop_patience: int = 10
    grad_clip: float = 1.0

    # quantization range: normalized (z-score) data often lives around [-3, 3]
    amp_min: float = -3.0
    amp_max: float =  3.0

    receptive_field: int = 1024