"""Lempel-Ziv complexity over token ID sequences.

LZ76 complexity counts the number of distinct phrases when parsing a
sequence greedily: each phrase is the shortest substring not yet seen.
This directly measures process complexity — more principled than gzip
compression ratio and free of byte-encoding artifacts since it operates
on integer token IDs.
"""

import numpy as np
import pandas as pd


def lz76_complexity(tokens: np.ndarray) -> int:
    """Count distinct phrases in LZ76 parsing of an integer sequence.

    The LZ76 algorithm scans left to right, extending the current phrase
    until it forms a substring not previously seen, then emits it and
    starts a new phrase. The count of emitted phrases is the complexity.

    Args:
        tokens: 1-D array of integer token IDs.

    Returns:
        Number of distinct LZ76 phrases.
    """
    n = len(tokens)
    if n == 0:
        return 0

    seen: set[tuple[int, ...]] = set()
    complexity = 0
    start = 0

    while start < n:
        length = 1
        while start + length <= n:
            phrase = tuple(tokens[start:start + length])
            if phrase not in seen:
                seen.add(phrase)
                complexity += 1
                break
            length += 1
        else:
            # Remainder of sequence is already seen — still count it
            complexity += 1
        start += length

    return complexity


def sliding_lz_complexity(token_ids: pd.Series, window_size: int) -> np.ndarray:
    """LZ76 complexity over a sliding window of token IDs.

    Same convention as sliding_compressibility: positions 0..window_size-2
    are NaN (insufficient tokens for a full window).

    Args:
        token_ids: Series of integer token IDs.
        window_size: Number of tokens per window.

    Returns:
        Array of length len(token_ids) with LZ76 complexity values.
    """
    ids = token_ids.to_numpy()
    n = len(ids)
    results = np.full(n, np.nan)

    for i in range(window_size - 1, n):
        window = ids[i - window_size + 1:i + 1]
        results[i] = lz76_complexity(window)

    return results
