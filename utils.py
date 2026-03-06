"""Shared utilities for autoloop scripts."""

import gzip

import numpy as np


def compressibility(text: bytes) -> float:
    """Compression ratio: compressed_size / original_size. Lower = more compressible."""
    compressed = gzip.compress(text, compresslevel=6)
    return len(compressed) / len(text)


def eos_ema(eos: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average of a binary EOS signal.

    Returns array same length as input. Uses standard EMA with
    alpha = 2 / (span + 1).
    """
    alpha = 2.0 / (span + 1)
    out = np.empty_like(eos, dtype=np.float64)
    out[0] = float(eos[0])
    for i in range(1, len(eos)):
        out[i] = alpha * float(eos[i]) + (1 - alpha) * out[i - 1]
    return out
