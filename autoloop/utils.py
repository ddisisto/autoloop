"""Shared utilities for autoloop scripts."""

import gzip
import os

import numpy as np


def compressibility(text: bytes) -> float:
    """Compression ratio: compressed_size / original_size. Lower = more compressible."""
    if len(text) == 0:
        return 1.0
    compressed = gzip.compress(text, compresslevel=6)
    return len(compressed) / len(text)


# Cache for incompressible baselines keyed by byte length.
_baseline_cache: dict[int, float] = {}


def compressibility_baseline(n_bytes: int, n_samples: int = 8) -> float:
    """Compression ratio of incompressible (random) data at a given byte length.

    Averages over n_samples random byte strings to smooth out variance.
    Results are cached for the lifetime of the process.
    """
    if n_bytes <= 0:
        return 1.0
    if n_bytes in _baseline_cache:
        return _baseline_cache[n_bytes]
    ratios = []
    for _ in range(n_samples):
        raw = os.urandom(n_bytes)
        compressed = gzip.compress(raw, compresslevel=6)
        ratios.append(len(compressed) / n_bytes)
    baseline = float(np.mean(ratios))
    _baseline_cache[n_bytes] = baseline
    return baseline


def normalized_compressibility(text: bytes) -> float:
    """Compressibility normalized against incompressible baseline at matched length.

    Returns raw_ratio / baseline_ratio, so ~1.0 means incompressible and
    values < 1.0 indicate real structure. Removes gzip fixed-overhead bias
    that inflates ratios at short byte lengths.
    """
    if len(text) == 0:
        return 1.0
    raw_ratio = compressibility(text)
    baseline = compressibility_baseline(len(text))
    return raw_ratio / baseline


def fix_decoded_texts(tokenizer: object, token_ids: list[int], texts: list[str]) -> list[str]:
    """Fix U+FFFD replacement characters from single-token decoding.

    Byte-level BPE tokens that are part of multi-byte UTF-8 sequences produce
    U+FFFD when decoded individually.  This groups consecutive affected tokens
    and batch-decodes them to recover the actual characters.  The result is
    assigned to the first token of each group; the rest get empty strings.
    """
    result = list(texts)
    i = 0
    while i < len(result):
        if "\ufffd" in result[i]:
            j = i
            while j < len(result) and "\ufffd" in result[j]:
                j += 1
            batch = tokenizer.decode(token_ids[i:j])
            if "\ufffd" not in batch:
                result[i] = batch
                for k in range(i + 1, j):
                    result[k] = ""
            i = j
        else:
            i += 1
    return result


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
