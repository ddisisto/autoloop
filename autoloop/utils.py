"""Shared utilities for autoloop scripts."""

import gzip

import numpy as np


def compressibility(text: bytes) -> float:
    """Compression ratio: compressed_size / original_size. Lower = more compressible."""
    if len(text) == 0:
        return 1.0
    compressed = gzip.compress(text, compresslevel=6)
    return len(compressed) / len(text)


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
