"""Semantic analysis: data types, loading, theme search, attractors.

Core analysis primitives for semantic analysis across runs. Dataclasses,
run loading, theme hit search, vocabulary stats, attractor cataloging,
and delegate wrappers for analyze.semantic functions.
"""

import glob
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .analyze.semantic import (
    HeapsLaw,
    CoherenceProfile,
    RepetitionOnset,
    fit_heaps_law as _fit_heaps,
    measure_coherence as _measure_coherence,
    detect_repetition_onset as _detect_repetition,
    vocab_stats as _vocab_stats,
)

log = logging.getLogger(__name__)


@dataclass
class RunInfo:
    path: str
    name: str
    L: int
    T: float
    seed: int
    text: str
    tokens: list[str]
    entropies: list[float]
    log_probs: list[float]


@dataclass
class ThemeHit:
    """A single occurrence of the theme in a run."""
    run_name: str
    L: int
    T: float
    char_pos: int
    token_idx: int
    context_before: str
    context_after: str
    local_entropy_mean: float
    local_entropy_std: float


@dataclass
class AttractorInfo:
    run_name: str
    L: int
    T: float
    mean_entropy: float
    is_collapsed: bool
    top_bigrams: list[tuple[str, int]]
    top_trigrams: list[tuple[str, int]]
    top_4grams: list[tuple[str, int]]
    repetition_ratio: float  # fraction of bigrams that appear 10+ times


@dataclass
class RunRepetitionOnset:
    run_name: str
    L: int
    T: float
    onset_step: int | None
    onset_fraction: float
    final_rep_ratio: float
    profile: list[float]


@dataclass
class RunHeapsLaw:
    run_name: str
    L: int
    T: float
    beta: float
    K: float
    r_squared: float
    vocab_at_10k: int
    vocab_at_100k: int
    saturation_ratio: float


@dataclass
class RunCoherenceProfile:
    run_name: str
    L: int
    T: float
    mean_coherence: float
    std_coherence: float
    min_coherence: float
    coherence_curve: list[float]


def parse_run_params(name: str) -> tuple[int, float, int]:
    """Extract L, T, seed from run filename like L0256_T0.80_S42."""
    m = re.match(r"L(\d+)_T([\d.]+)_S(\d+)", name)
    if not m:
        return 0, 0.0, 0
    return int(m.group(1)), float(m.group(2)), int(m.group(3))


def load_run(path: str) -> RunInfo | None:
    """Load a single run's experiment-phase data."""
    name = Path(path).stem
    L, T, seed = parse_run_params(name)

    df = pd.read_parquet(path)
    exp = df[df.phase == "experiment"].reset_index(drop=True)
    if len(exp) == 0:
        return None

    # Fall back to parquet columns if filename doesn't match standard pattern
    if L == 0:
        if "context_length" in df.columns and "temperature" in df.columns:
            L = int(exp["context_length"].iloc[0])
            T = float(exp["temperature"].iloc[0])
            # Try to extract seed from filename
            m = re.search(r"S(\d+)", name)
            seed = int(m.group(1)) if m else 0
        else:
            return None

    return RunInfo(
        path=path,
        name=name,
        L=L, T=T, seed=seed,
        text="".join(exp.decoded_text.tolist()),
        tokens=exp.decoded_text.tolist(),
        entropies=exp.entropy.tolist(),
        log_probs=exp.log_prob.tolist(),
    )


def find_theme_hits(
    run: RunInfo, theme: str, context_radius: int, entropy_window: int
) -> list[ThemeHit]:
    """Find all occurrences of theme in run text with context."""
    hits = []
    text_lower = run.text.lower()
    theme_lower = theme.lower()
    start = 0

    # Build char->token index mapping (cumulative char positions)
    char_positions: list[int] = []
    pos = 0
    for tok in run.tokens:
        char_positions.append(pos)
        pos += len(tok)

    while True:
        idx = text_lower.find(theme_lower, start)
        if idx == -1:
            break

        # Find which token this char position falls in
        token_idx = 0
        for i, cp in enumerate(char_positions):
            if cp > idx:
                break
            token_idx = i

        # Context strings
        ctx_before = run.text[max(0, idx - context_radius) : idx]
        ctx_after = run.text[idx + len(theme) : idx + len(theme) + context_radius]

        # Local entropy stats
        ew = entropy_window
        e_start = max(0, token_idx - ew)
        e_end = min(len(run.entropies), token_idx + ew)
        local_e = run.entropies[e_start:e_end]
        e_mean = sum(local_e) / len(local_e) if local_e else 0.0
        e_std = (sum((x - e_mean) ** 2 for x in local_e) / len(local_e)) ** 0.5 if local_e else 0.0

        hits.append(ThemeHit(
            run_name=run.name, L=run.L, T=run.T,
            char_pos=idx, token_idx=token_idx,
            context_before=ctx_before, context_after=ctx_after,
            local_entropy_mean=e_mean, local_entropy_std=e_std,
        ))
        start = idx + 1

    return hits


def vocab_stats(run: RunInfo) -> dict:
    """Compute vocabulary richness metrics for a run."""
    result = _vocab_stats(run.tokens)
    result.update(name=run.name, L=run.L, T=run.T, seed=run.seed)
    return result


def _is_content_token(tok: str) -> bool:
    """Check if a token carries semantic content (not punctuation/fragment)."""
    stripped = tok.strip()
    if not stripped:
        return False
    # Pure punctuation or digits
    if re.fullmatch(r"[^a-zA-Z]+", stripped):
        return False
    # Single character (after stripping)
    alpha = re.sub(r"[^a-zA-Z]", "", stripped)
    if len(alpha) < 2:
        return False
    return True


def neighbor_profile(
    hits: list[ThemeHit], all_runs: dict[str, RunInfo], token_radius: int = 10
) -> dict[float, Counter]:
    """For each T value, collect tokens neighboring theme hits."""
    by_t: dict[float, Counter] = {}
    for hit in hits:
        run = all_runs[hit.run_name]
        start = max(0, hit.token_idx - token_radius)
        end = min(len(run.tokens), hit.token_idx + token_radius + 1)
        neighbors = run.tokens[start:end]
        by_t.setdefault(hit.T, Counter()).update(
            tok.strip().lower() for tok in neighbors if _is_content_token(tok)
        )
    return by_t


def _shares_morph_root(word: str, theme: str, min_prefix: int = 4) -> bool:
    """Check if word shares a morphological root with theme via shared prefix."""
    if len(word) < min_prefix or len(theme) < min_prefix:
        return False
    shared = 0
    for i in range(min(len(word), len(theme))):
        if word[i] != theme[i]:
            break
        shared += 1
    return shared >= min_prefix


def neighbor_morphology(
    hits: list[ThemeHit],
    all_runs: dict[str, RunInfo],
    theme: str,
    token_radius: int = 10,
    top_k: int = 15,
) -> dict[float, tuple[float, float, int]]:
    """Per T: (morph_rank_ratio, subword_ratio, n_content_neighbors).

    morph_rank_ratio: fraction of top-K most frequent content neighbors that
        share morphological root with theme. Captures rank-based crossover
        (morph variants rise to top at high T).
    subword_ratio: fraction of all non-empty neighbor tokens that are subword
        continuations (no leading space).
    """
    theme_clean = re.sub(r"[^a-z]", "", theme.lower())
    by_t_content: dict[float, Counter] = {}
    by_t_subword: dict[float, list[int]] = {}

    for hit in hits:
        run = all_runs[hit.run_name]
        start = max(0, hit.token_idx - token_radius)
        end = min(len(run.tokens), hit.token_idx + token_radius + 1)

        for tok in run.tokens[start:end]:
            if not tok:
                continue
            is_subword = not tok[0].isspace()
            by_t_subword.setdefault(hit.T, []).append(1 if is_subword else 0)

            if not _is_content_token(tok):
                continue
            word = re.sub(r"[^a-z]", "", tok.strip().lower())
            if word == theme_clean:
                continue
            by_t_content.setdefault(hit.T, Counter())[word] += 1

    result = {}
    for t_val in sorted(set(by_t_content) | set(by_t_subword)):
        content = by_t_content.get(t_val, Counter())
        sub_list = by_t_subword.get(t_val, [])
        # Morph ratio among top-K most frequent neighbors
        top_words = [w for w, _ in content.most_common(top_k)]
        n_morph = sum(1 for w in top_words if _shares_morph_root(w, theme_clean))
        morph_r = n_morph / len(top_words) if top_words else 0.0
        sub_r = sum(sub_list) / len(sub_list) if sub_list else 0.0
        result[t_val] = (morph_r, sub_r, sum(content.values()))
    return result


def extract_ngrams(tokens: list[str], n: int) -> list[str]:
    """Extract character-level n-grams from token list (joined text, word-split)."""
    text = "".join(tokens)
    words = text.lower().split()
    if len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def attractor_catalog(runs: dict[str, RunInfo], entropy_threshold: float = 1.0) -> list[AttractorInfo]:
    """Identify and fingerprint attractors in each run."""
    results = []
    for run in runs.values():
        mean_e = sum(run.entropies) / len(run.entropies)
        is_collapsed = mean_e < entropy_threshold

        words = "".join(run.tokens).lower().split()

        bigrams = Counter()
        trigrams = Counter()
        fourgrams = Counter()
        for i in range(len(words) - 1):
            bigrams[f"{words[i]} {words[i+1]}"] += 1
        for i in range(len(words) - 2):
            trigrams[f"{words[i]} {words[i+1]} {words[i+2]}"] += 1
        for i in range(len(words) - 3):
            fourgrams[f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}"] += 1

        n_bigrams = sum(bigrams.values())
        heavy_bigrams = sum(c for c in bigrams.values() if c >= 10)
        rep_ratio = heavy_bigrams / n_bigrams if n_bigrams > 0 else 0.0

        results.append(AttractorInfo(
            run_name=run.name, L=run.L, T=run.T,
            mean_entropy=mean_e, is_collapsed=is_collapsed,
            top_bigrams=bigrams.most_common(5),
            top_trigrams=trigrams.most_common(5),
            top_4grams=fourgrams.most_common(5),
            repetition_ratio=rep_ratio,
        ))

    results.sort(key=lambda a: (a.L, a.T))
    return results


# ── Analyses 3-5: delegate to analyze.semantic ────────────────────
# CLI wrappers add run metadata (name, L, T) to package results.


def detect_repetition_onset(
    run: RunInfo, window: int = 5000, stride: int = 5000, threshold: float = 0.5
) -> RunRepetitionOnset:
    r = _detect_repetition(run.tokens, window=window, stride=stride, threshold=threshold)
    return RunRepetitionOnset(
        run_name=run.name, L=run.L, T=run.T,
        onset_step=r.onset_step, onset_fraction=r.onset_fraction,
        final_rep_ratio=r.final_rep_ratio, profile=r.profile,
    )


def fit_heaps_law(run: RunInfo, checkpoints: int = 20) -> RunHeapsLaw:
    h = _fit_heaps(run.tokens, checkpoints=checkpoints)
    return RunHeapsLaw(
        run_name=run.name, L=run.L, T=run.T,
        beta=h.beta, K=h.K, r_squared=h.r_squared,
        vocab_at_10k=h.vocab_at_10k, vocab_at_100k=h.vocab_at_100k,
        saturation_ratio=h.saturation_ratio,
    )


def measure_coherence(
    run: RunInfo, window_words: int = 500, stride_words: int = 250
) -> RunCoherenceProfile:
    c = _measure_coherence(run.tokens, window_words=window_words, stride_words=stride_words)
    return RunCoherenceProfile(
        run_name=run.name, L=run.L, T=run.T,
        mean_coherence=c.mean_coherence, std_coherence=c.std_coherence,
        min_coherence=c.min_coherence, coherence_curve=c.coherence_curve,
    )


def _discover_run_files(
    run_patterns: list[str] | None,
) -> list[str]:
    """Find parquet files from explicit patterns or default glob."""
    if run_patterns:
        files = []
        for pattern in run_patterns:
            files.extend(glob.glob(pattern))
        return files
    from .runlib import discover_runs
    return [str(p) for p in discover_runs()]


def _load_runs(
    files: list[str], seed_filter: int | None = None,
) -> dict[str, RunInfo]:
    """Load all runs, optionally filtering by seed."""
    all_runs: dict[str, RunInfo] = {}
    for f in sorted(files):
        run = load_run(f)
        if run is None:
            continue
        if seed_filter is not None and run.seed != seed_filter:
            continue
        all_runs[run.name] = run
    return all_runs
