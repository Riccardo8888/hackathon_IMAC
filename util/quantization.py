import numpy as np

def dequantize_centers(x_min: float, x_max: float, n_bins: int = 256) -> np.ndarray:
    return np.linspace(x_min, x_max, n_bins, dtype=np.float64)

def mu_law_encode_np(x, mu=255, amp_max=8e-5):
    """Normalized mu-law: maps [-amp_max, amp_max] to [0, mu]"""
    x = np.clip(x / amp_max, -1, 1)
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return ((x_mu + 1) / 2 * mu).astype(np.int64)

def mu_law_decode_np(q, mu=255, amp_max=8e-5):
    """Inverse mu-law: maps [0, mu] back to [-amp_max, amp_max]"""
    x_mu = (q / mu) * 2 - 1
    x_norm = np.sign(x_mu) * ((1.0 + mu)**np.abs(x_mu) - 1.0) / mu
    return x_norm * amp_max

