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
    amp_min = -8.0e-05
    amp_max =  8.0e-05
    n_bins: int = 256

    # architecture
    kernel_size: int = 2
    n_blocks: int = 10
    n_filters: int = 40

    # training
    lr: float = 1e-3
    lr_decay_gamma: float = 0.99
    batch_size: int = 32
    epochs: int = 600

    seq_len: int = 2000
    train_samples_per_epoch: int = 12000
    val_samples_fixed: int = 4000

    early_stop_patience: int = 40
    grad_clip: float = 1.0

    receptive_field: int = 1024