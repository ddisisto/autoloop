#!/usr/bin/env python3
"""Semantic analysis of generated text across conditions.

Searches for a theme word/phrase across all runs, extracts context windows,
and computes vocabulary statistics by (L, T) condition.

Analyses:
  1. Theme search: hit counts, context windows, local entropy, neighbor tokens
  2. Attractor catalog: top n-grams per collapsed run, attractor fingerprinting
  3. Repetition onset: detect the step where n-gram repetition begins
  4. Heaps' law: fit vocabulary growth V(n) = K·n^β, report β by condition
  5. Semantic coherence: n-gram overlap between sliding windows

Usage:
    python semantic.py                          # default theme: "temperature"
    python semantic.py --theme "the"            # custom theme
    python semantic.py --theme "temperature" --context-radius 30
    python semantic.py --runs data/runs/L0256*.parquet
    python semantic.py --csv data/semantic.csv  # export metrics
"""

import argparse
import glob
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analyze.semantic import (
    HeapsLaw,
    CoherenceProfile,
    RepetitionOnset,
    fit_heaps_law as _fit_heaps,
    measure_coherence as _measure_coherence,
    detect_repetition_onset as _detect_repetition,
    vocab_stats as _vocab_stats,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
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

    cols = ["decoded_text", "entropy", "log_prob"]
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
            tok.strip().lower() for tok in neighbors if tok.strip()
        )
    return by_t


## ── Analysis 2: Attractor catalog ──────────────────────────────────


def extract_ngrams(tokens: list[str], n: int) -> list[str]:
    """Extract character-level n-grams from token list (joined text, word-split)."""
    text = "".join(tokens)
    words = text.lower().split()
    if len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


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


## ── Analyses 3-5: delegate to analyze.semantic ────────────────────
# CLI wrappers add run metadata (name, L, T) to package results.


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


## ── Report printing ──────────────────────────────────────────────


def print_report(
    theme: str,
    all_runs: dict[str, RunInfo],
    hits: list[ThemeHit],
    vocab: list[dict],
    neighbor_tokens: dict[float, Counter],
    attractors: list[AttractorInfo],
    repetitions: list[RepetitionOnset],
    heaps: list[HeapsLaw],
    coherence: list[CoherenceProfile],
) -> None:
    """Print the full analysis report."""
    print(f"\n{'='*70}")
    print(f"  SEMANTIC ANALYSIS: '{theme}'")
    print(f"  Runs analyzed: {len(all_runs)}")
    print(f"{'='*70}\n")

    # --- Hit counts by condition ---
    print("## Hit Counts by (L, T)\n")
    hit_grid: dict[tuple[int, float], int] = Counter()
    for h in hits:
        hit_grid[(h.L, h.T)] += 1

    if not hits:
        print(f"  No occurrences of '{theme}' found in any run.\n")
    else:
        # Get sorted unique L and T values
        ls = sorted(set(h.L for h in hits))
        ts = sorted(set(h.T for h in hits))

        # Print header
        header = f"{'L':>6}" + "".join(f"  T={t:.2f}" for t in ts)
        print(header)
        print("-" * len(header))
        for l_val in ls:
            row = f"{l_val:>6}"
            for t_val in ts:
                c = hit_grid.get((l_val, t_val), 0)
                row += f"  {c:>6}" if c > 0 else "       ."
                # pad to match header width
            print(row)

        print(f"\n  Total hits: {len(hits)}")
        print(f"  Runs with hits: {len(set(h.run_name for h in hits))}/{len(all_runs)}")

    # --- Aggregated by T ---
    print("\n## Hit Rate by Temperature\n")
    by_t: dict[float, list] = {}
    for h in hits:
        by_t.setdefault(h.T, []).append(h)
    all_ts = sorted(set(r.T for r in all_runs.values()))
    for t_val in all_ts:
        t_hits = by_t.get(t_val, [])
        t_runs = [r for r in all_runs.values() if r.T == t_val]
        total_chars = sum(len(r.text) for r in t_runs)
        rate = len(t_hits) / total_chars * 100000 if total_chars > 0 else 0
        bar = "#" * int(rate * 2)
        print(f"  T={t_val:.2f}: {len(t_hits):4d} hits across {len(t_runs):2d} runs "
              f"({rate:5.1f}/100k chars) {bar}")

    # --- Aggregated by L ---
    print("\n## Hit Rate by Context Length\n")
    by_l: dict[int, list] = {}
    for h in hits:
        by_l.setdefault(h.L, []).append(h)
    all_ls = sorted(set(r.L for r in all_runs.values()))
    for l_val in all_ls:
        l_hits = by_l.get(l_val, [])
        l_runs = [r for r in all_runs.values() if r.L == l_val]
        total_chars = sum(len(r.text) for r in l_runs)
        rate = len(l_hits) / total_chars * 100000 if total_chars > 0 else 0
        bar = "#" * int(rate * 2)
        print(f"  L={l_val:>4d}: {len(l_hits):4d} hits across {len(l_runs):2d} runs "
              f"({rate:5.1f}/100k chars) {bar}")

    # --- Sample contexts ---
    if hits:
        print(f"\n## Sample Contexts (up to 5 per T)\n")
        for t_val in sorted(by_t):
            t_hits = by_t[t_val]
            print(f"  --- T={t_val:.2f} ({len(t_hits)} total) ---")
            for h in t_hits[:5]:
                before = h.context_before.replace("\n", "\\n")
                after = h.context_after.replace("\n", "\\n")
                print(f"    [{h.run_name} step~{h.token_idx}] "
                      f"H_local={h.local_entropy_mean:.2f}±{h.local_entropy_std:.2f}")
                print(f"      ...{before}>>>{theme}<<<{after}...")
            print()

    # --- Local entropy around hits ---
    if hits:
        print("## Local Entropy Around Theme Hits\n")
        for t_val in sorted(by_t):
            t_hits = by_t[t_val]
            mean_h = sum(h.local_entropy_mean for h in t_hits) / len(t_hits)
            mean_std = sum(h.local_entropy_std for h in t_hits) / len(t_hits)
            print(f"  T={t_val:.2f}: avg local entropy = {mean_h:.3f} ± {mean_std:.3f} "
                  f"(n={len(t_hits)})")

    # --- Neighbor token profiles ---
    if neighbor_tokens:
        print("\n## Most Common Neighbor Tokens (±10 tokens from theme)\n")
        for t_val in sorted(neighbor_tokens):
            top = neighbor_tokens[t_val].most_common(20)
            top_str = ", ".join(f"'{w}'({c})" for w, c in top)
            print(f"  T={t_val:.2f}: {top_str}")
        print()

    # --- Vocabulary stats ---
    print("\n## Vocabulary Richness by Condition\n")
    # Sort by (L, T)
    vocab.sort(key=lambda v: (v["L"], v["T"]))
    print(f"  {'Name':35s} {'Words':>8s} {'Unique':>8s} {'TTR':>7s} "
          f"{'Hapax':>7s} {'Hap%':>6s}  Top 3 words")
    print("  " + "-" * 100)
    for v in vocab:
        top3 = ", ".join(f"{w}({c})" for w, c in v["top10"][:3])
        print(f"  {v['name']:35s} {v['n_words']:>8d} {v['n_unique']:>8d} "
              f"{v['type_token_ratio']:>7.4f} {v['hapax_count']:>7d} "
              f"{v['hapax_ratio']:>6.3f}  {top3}")

    # ── Attractor Catalog ──
    print(f"\n{'='*70}")
    print("  ATTRACTOR CATALOG")
    print(f"{'='*70}\n")

    collapsed = [a for a in attractors if a.is_collapsed]
    escaped = [a for a in attractors if not a.is_collapsed]
    print(f"  Collapsed runs (entropy < 1.0): {len(collapsed)}")
    print(f"  Escaped runs: {len(escaped)}\n")

    if collapsed:
        print(f"  {'Name':35s} {'H_mean':>7s} {'RepRatio':>9s}  Top trigram")
        print("  " + "-" * 80)
        for a in collapsed:
            top_tri = a.top_trigrams[0][0] if a.top_trigrams else "n/a"
            top_tri_c = a.top_trigrams[0][1] if a.top_trigrams else 0
            print(f"  {a.run_name:35s} {a.mean_entropy:>7.3f} {a.repetition_ratio:>9.3f}  "
                  f"'{top_tri}' ({top_tri_c})")

        print(f"\n  Detailed top n-grams for collapsed runs:\n")
        for a in collapsed:
            print(f"  --- {a.run_name} (H={a.mean_entropy:.3f}) ---")
            print(f"    Bigrams:  {', '.join(f'{g}({c})' for g,c in a.top_bigrams[:5])}")
            print(f"    Trigrams: {', '.join(f'{g}({c})' for g,c in a.top_trigrams[:5])}")
            print(f"    4-grams:  {', '.join(f'{g}({c})' for g,c in a.top_4grams[:5])}")

    # ── Repetition Onset ──
    print(f"\n{'='*70}")
    print("  REPETITION ONSET")
    print(f"{'='*70}\n")

    repetitions.sort(key=lambda r: (r.L, r.T))
    print(f"  {'Name':35s} {'Onset':>8s} {'Frac':>6s} {'Final':>6s}  Profile (5k-step windows)")
    print("  " + "-" * 90)
    for r in repetitions:
        onset_str = f"{r.onset_step:>8d}" if r.onset_step is not None else "   never"
        frac_str = f"{r.onset_fraction:>6.2f}" if r.onset_step is not None else "     -"
        # Mini sparkline of profile
        spark = ""
        for val in r.profile:
            if val < 0.1:
                spark += "▁"
            elif val < 0.2:
                spark += "▂"
            elif val < 0.3:
                spark += "▃"
            elif val < 0.4:
                spark += "▄"
            elif val < 0.5:
                spark += "▅"
            elif val < 0.7:
                spark += "▆"
            elif val < 0.9:
                spark += "▇"
            else:
                spark += "█"
        print(f"  {r.run_name:35s} {onset_str} {frac_str} {r.final_rep_ratio:>6.3f}  {spark}")

    # ── Heaps' Law ──
    print(f"\n{'='*70}")
    print("  HEAPS' LAW: V(n) = K·n^β")
    print(f"{'='*70}\n")

    heaps.sort(key=lambda h: (h.L, h.T))
    print(f"  {'Name':35s} {'β':>6s} {'K':>8s} {'R²':>6s} {'V@10k':>7s} {'V@100k':>7s} {'Sat':>5s}")
    print("  " + "-" * 85)
    for h in heaps:
        print(f"  {h.run_name:35s} {h.beta:>6.3f} {h.K:>8.1f} {h.r_squared:>6.3f} "
              f"{h.vocab_at_10k:>7d} {h.vocab_at_100k:>7d} {h.saturation_ratio:>5.2f}")

    # Summary: β by T (averaged across L)
    print(f"\n  β by Temperature (averaged across L):\n")
    beta_by_t: dict[float, list[float]] = {}
    for h in heaps:
        beta_by_t.setdefault(h.T, []).append(h.beta)
    for t_val in sorted(beta_by_t):
        betas = beta_by_t[t_val]
        mean_b = sum(betas) / len(betas)
        bar = "█" * int(mean_b * 20)
        print(f"    T={t_val:.2f}: β={mean_b:.3f}  {bar}")

    # ── Semantic Coherence ──
    print(f"\n{'='*70}")
    print("  SEMANTIC COHERENCE (bigram Jaccard between adjacent 500-word windows)")
    print(f"{'='*70}\n")

    coherence.sort(key=lambda c: (c.L, c.T))
    print(f"  {'Name':35s} {'Mean':>6s} {'Std':>6s} {'Min':>6s}  Profile")
    print("  " + "-" * 80)
    for c in coherence:
        # Subsample coherence curve for sparkline (take every Nth)
        curve = c.coherence_curve
        if len(curve) > 40:
            step = len(curve) // 40
            curve = curve[::step]
        spark = ""
        for val in curve:
            if val < 0.02:
                spark += "▁"
            elif val < 0.05:
                spark += "▂"
            elif val < 0.10:
                spark += "▃"
            elif val < 0.20:
                spark += "▄"
            elif val < 0.35:
                spark += "▅"
            elif val < 0.50:
                spark += "▆"
            elif val < 0.70:
                spark += "▇"
            else:
                spark += "█"
        print(f"  {c.run_name:35s} {c.mean_coherence:>6.3f} {c.std_coherence:>6.3f} "
              f"{c.min_coherence:>6.3f}  {spark}")

    # Summary: coherence by T
    print(f"\n  Mean coherence by Temperature:\n")
    coh_by_t: dict[float, list[float]] = {}
    for c in coherence:
        coh_by_t.setdefault(c.T, []).append(c.mean_coherence)
    for t_val in sorted(coh_by_t):
        vals = coh_by_t[t_val]
        mean_c = sum(vals) / len(vals)
        bar = "█" * int(mean_c * 40)
        print(f"    T={t_val:.2f}: coherence={mean_c:.3f}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic analysis across runs")
    parser.add_argument("--theme", default="temperature",
                        help="Theme word/phrase to search for (default: temperature)")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Specific parquet files (default: all L*.parquet)")
    parser.add_argument("--context-radius", type=int, default=80,
                        help="Characters of context around each hit (default: 80)")
    parser.add_argument("--entropy-window", type=int, default=20,
                        help="Token window for local entropy stats (default: 20)")
    parser.add_argument("--csv", default=None,
                        help="Export vocab stats to CSV")
    parser.add_argument("--seed", type=int, default=None,
                        help="Filter to specific seed (default: all)")
    args = parser.parse_args()

    # Discover runs
    if args.runs:
        files = []
        for pattern in args.runs:
            files.extend(glob.glob(pattern))
    else:
        files = sorted(
            glob.glob("data/runs/L*.parquet")
            + glob.glob("data/runs/anneal_*.parquet")
            + glob.glob("data/runs/sched_*.parquet")
        )

    if not files:
        log.error("No parquet files found")
        sys.exit(1)

    log.info(f"Loading {len(files)} runs...")

    # Load all runs
    all_runs: dict[str, RunInfo] = {}
    for f in sorted(files):
        run = load_run(f)
        if run is None:
            continue
        if args.seed is not None and run.seed != args.seed:
            continue
        all_runs[run.name] = run

    log.info(f"Loaded {len(all_runs)} runs")

    # Find theme hits
    log.info(f"Searching for '{args.theme}'...")
    all_hits: list[ThemeHit] = []
    for run in all_runs.values():
        hits = find_theme_hits(run, args.theme, args.context_radius, args.entropy_window)
        all_hits.extend(hits)

    log.info(f"Found {len(all_hits)} hits across {len(set(h.run_name for h in all_hits))} runs")

    # Neighbor analysis
    neighbors = neighbor_profile(all_hits, all_runs)

    # Vocab stats
    log.info("Computing vocabulary stats...")
    vocab = [vocab_stats(run) for run in all_runs.values()]

    # Attractor catalog
    log.info("Building attractor catalog...")
    attractors = attractor_catalog(all_runs)

    # Repetition onset
    log.info("Detecting repetition onset...")
    repetitions = [detect_repetition_onset(run) for run in all_runs.values()]

    # Heaps' law
    log.info("Fitting Heaps' law...")
    heaps = [fit_heaps_law(run) for run in all_runs.values()]

    # Semantic coherence
    log.info("Measuring semantic coherence...")
    coherence_results = [measure_coherence(run) for run in all_runs.values()]

    # Report
    print_report(
        args.theme, all_runs, all_hits, vocab, neighbors,
        attractors, repetitions, heaps, coherence_results,
    )

    # CSV export
    if args.csv:
        rows = []
        heaps_by_name = {h.run_name: h for h in heaps}
        coh_by_name = {c.run_name: c for c in coherence_results}
        rep_by_name = {r.run_name: r for r in repetitions}
        attr_by_name = {a.run_name: a for a in attractors}

        for v in vocab:
            row = {k: val for k, val in v.items() if k != "top10"}
            name = v["name"]
            if name in heaps_by_name:
                h = heaps_by_name[name]
                row.update({"heaps_beta": h.beta, "heaps_K": h.K,
                            "heaps_r2": h.r_squared, "saturation_ratio": h.saturation_ratio})
            if name in coh_by_name:
                c = coh_by_name[name]
                row.update({"coherence_mean": c.mean_coherence,
                            "coherence_std": c.std_coherence,
                            "coherence_min": c.min_coherence})
            if name in rep_by_name:
                r = rep_by_name[name]
                row.update({"rep_onset_step": r.onset_step,
                            "rep_onset_frac": r.onset_fraction,
                            "rep_final_ratio": r.final_rep_ratio})
            if name in attr_by_name:
                a = attr_by_name[name]
                row.update({"is_collapsed": a.is_collapsed,
                            "rep_ratio": a.repetition_ratio,
                            "mean_entropy": a.mean_entropy})
            rows.append(row)

        pd.DataFrame(rows).to_csv(args.csv, index=False)
        log.info(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
