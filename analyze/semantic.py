"""Semantic analysis computation functions.

Heaps' law fitting, vocabulary statistics, semantic coherence,
repetition onset detection, and attractor cataloging.

These are pure computation functions operating on token/text data.
The CLI wrapper lives at top-level semantic.py.
"""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class HeapsLaw:
    beta: float       # exponent: V(n) = K * n^beta
    K: float           # coefficient
    r_squared: float   # goodness of fit
    vocab_at_10k: int
    vocab_at_100k: int
    saturation_ratio: float  # vocab_100k / vocab_10k


@dataclass
class CoherenceProfile:
    mean_coherence: float    # average overlap between adjacent windows
    std_coherence: float
    min_coherence: float     # sharpest topic break
    coherence_curve: list[float]  # overlap at each window boundary


@dataclass
class RepetitionOnset:
    onset_step: int | None  # step where repetition ratio crosses threshold
    onset_fraction: float   # onset_step / total_steps
    final_rep_ratio: float  # repetition ratio in last window
    profile: list[float]    # repetition ratio at each checkpoint


def vocab_stats(tokens: list[str]) -> dict:
    """Compute vocabulary richness metrics from token list.

    Returns dict with n_words, n_unique, type_token_ratio, hapax_count, hapax_ratio.
    """
    text = "".join(tokens)
    words = [w.lower() for w in text.split() if len(w) > 1]
    wc = Counter(words)
    n_tokens = len(words)
    n_types = len(wc)
    hapax = sum(1 for c in wc.values() if c == 1)
    ttr = n_types / n_tokens if n_tokens > 0 else 0.0

    return {
        "n_words": n_tokens,
        "n_unique": n_types,
        "type_token_ratio": round(ttr, 4),
        "hapax_count": hapax,
        "hapax_ratio": round(hapax / n_types, 4) if n_types > 0 else 0.0,
        "top10": wc.most_common(10),
    }


def fit_heaps_law(tokens: list[str], checkpoints: int = 20) -> HeapsLaw:
    """Fit Heaps' law V(n) = K·n^β to vocabulary growth curve."""
    text = "".join(tokens)
    words = text.lower().split()
    n_words = len(words)
    step_size = max(1, n_words // checkpoints)

    ns = []
    vs = []
    seen: set[str] = set()
    vocab_10k = 0
    vocab_100k = 0

    for i, w in enumerate(words):
        seen.add(w)
        if (i + 1) % step_size == 0:
            ns.append(i + 1)
            vs.append(len(seen))
        if i + 1 == 10000:
            vocab_10k = len(seen)

    vocab_100k = len(seen)
    if vocab_10k == 0:
        vocab_10k = len(seen)

    if len(ns) < 3:
        return HeapsLaw(
            beta=0.0, K=0.0, r_squared=0.0,
            vocab_at_10k=vocab_10k, vocab_at_100k=vocab_100k,
            saturation_ratio=vocab_100k / vocab_10k if vocab_10k > 0 else 0.0,
        )

    log_n = np.log(np.array(ns, dtype=float))
    log_v = np.log(np.array(vs, dtype=float))

    n_pts = len(log_n)
    sum_x = log_n.sum()
    sum_y = log_v.sum()
    sum_xy = (log_n * log_v).sum()
    sum_x2 = (log_n ** 2).sum()

    denom = n_pts * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-12:
        beta = 0.0
        log_K = sum_y / n_pts
    else:
        beta = (n_pts * sum_xy - sum_x * sum_y) / denom
        log_K = (sum_y - beta * sum_x) / n_pts

    K = math.exp(log_K)

    y_mean = sum_y / n_pts
    ss_tot = ((log_v - y_mean) ** 2).sum()
    y_pred = log_K + beta * log_n
    ss_res = ((log_v - y_pred) ** 2).sum()
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return HeapsLaw(
        beta=beta, K=K, r_squared=r_sq,
        vocab_at_10k=vocab_10k, vocab_at_100k=vocab_100k,
        saturation_ratio=vocab_100k / vocab_10k if vocab_10k > 0 else 0.0,
    )


def measure_coherence(
    tokens: list[str], window_words: int = 500, stride_words: int = 250
) -> CoherenceProfile:
    """Measure semantic coherence as bigram overlap between sliding windows.

    For each pair of adjacent windows, computes Jaccard similarity of their
    bigram sets. High overlap = topic persistence; low overlap = topic shift.
    """
    text = "".join(tokens)
    words = text.lower().split()
    n_words = len(words)

    def bigram_set(word_list: list[str]) -> set[str]:
        return {f"{word_list[i]} {word_list[i+1]}" for i in range(len(word_list) - 1)}

    windows = []
    for start in range(0, n_words - window_words, stride_words):
        windows.append(bigram_set(words[start : start + window_words]))

    if len(windows) < 2:
        return CoherenceProfile(
            mean_coherence=0.0, std_coherence=0.0, min_coherence=0.0,
            coherence_curve=[],
        )

    overlaps = []
    for i in range(len(windows) - 1):
        a, b = windows[i], windows[i + 1]
        union = len(a | b)
        inter = len(a & b)
        overlaps.append(inter / union if union > 0 else 0.0)

    arr = np.array(overlaps)
    return CoherenceProfile(
        mean_coherence=float(arr.mean()),
        std_coherence=float(arr.std()),
        min_coherence=float(arr.min()),
        coherence_curve=overlaps,
    )


def detect_repetition_onset(
    tokens: list[str], window: int = 5000, stride: int = 5000, threshold: float = 0.5
) -> RepetitionOnset:
    """Detect where n-gram repetition begins in a token stream.

    Slides a window across the text, measuring what fraction of bigrams in each
    window have been seen 3+ times. Onset is first window exceeding threshold.
    """
    text = "".join(tokens)
    words = text.lower().split()
    n_words = len(words)
    profile = []
    onset_step = None

    for start in range(0, n_words - window, stride):
        chunk = words[start : start + window]
        bigrams = Counter()
        for i in range(len(chunk) - 1):
            bigrams[f"{chunk[i]} {chunk[i+1]}"] += 1

        n_bi = sum(bigrams.values())
        repeated = sum(c for c in bigrams.values() if c >= 3)
        ratio = repeated / n_bi if n_bi > 0 else 0.0
        profile.append(ratio)

        if onset_step is None and ratio >= threshold:
            onset_step = start

    final_chunk = words[max(0, n_words - window):]
    final_bigrams = Counter()
    for i in range(len(final_chunk) - 1):
        final_bigrams[f"{final_chunk[i]} {final_chunk[i+1]}"] += 1
    n_bi = sum(final_bigrams.values())
    final_rep = sum(c for c in final_bigrams.values() if c >= 3) / n_bi if n_bi > 0 else 0.0

    return RepetitionOnset(
        onset_step=onset_step,
        onset_fraction=onset_step / n_words if onset_step is not None else 1.0,
        final_rep_ratio=final_rep,
        profile=profile,
    )
