"""Shared utilities for autoloop scripts."""

import gzip


def compressibility(text: bytes) -> float:
    """Compression ratio: compressed_size / original_size. Lower = more compressible."""
    compressed = gzip.compress(text, compresslevel=6)
    return len(compressed) / len(text)
