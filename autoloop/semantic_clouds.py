"""Theme cloud discovery, co-occurrence, and basin mapping.

Auto-discovers high-density content words across runs, finds co-occurring
theme pairs, and maps each run's dominant semantic basin.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from .semantic import RunInfo

log = logging.getLogger(__name__)

# No curated stopwords — let spikiness and distribution do the filtering.
# Interesting words ("state", "power", "order", "set", "in") were previously
# hidden by a hand-curated list. Now we filter only mechanical noise:
# punctuation, single chars, subword fragments.
STOPWORDS = frozenset()


@dataclass
class ThemeCloud:
    """A theme word's density profile across runs."""
    word: str
    total_hits: int
    n_runs_present: int
    max_density: float
    mean_density: float
    top_run: str
    top_count: int
    spikiness: float


@dataclass
class ThemePair:
    """Co-occurrence of two themes in the same runs."""
    word_a: str
    word_b: str
    shared_runs: list[str]
    jaccard: float


@dataclass
class RunBasin:
    """A run's dominant semantic basin (top themes)."""
    run_name: str
    L: int
    T: float
    top_themes: list[tuple[str, float]]  # (word, density)
    basin_label: str  # human-readable summary


def _clean_words(tokens: list[str]) -> list[str]:
    """Extract content words from token list, filtering mechanical noise."""
    text = "".join(tokens)
    words = text.lower().split()
    cleaned = []
    for w in words:
        w = re.sub(r"[^a-z]", "", w)
        if len(w) >= 2 and w not in STOPWORDS:
            cleaned.append(w)
    return cleaned


def discover_themes(
    runs: dict[str, RunInfo], min_hits: int = 100, min_runs: int = 3, top_n: int = 60
) -> tuple[list[ThemeCloud], dict[str, Counter], dict[str, int]]:
    """Find content words with highest density across runs."""
    global_counts: Counter = Counter()
    per_run: dict[str, Counter] = {}
    run_sizes: dict[str, int] = {}

    for name, run in runs.items():
        words = _clean_words(run.tokens)
        c = Counter(words)
        per_run[name] = c
        run_sizes[name] = len(words)
        global_counts.update(c)

    results = []
    for word, total in global_counts.items():
        if total < min_hits:
            continue
        densities = []
        for name in runs:
            n = run_sizes.get(name, 0)
            if n == 0:
                continue
            count = per_run[name].get(word, 0)
            densities.append((name, count, count / n))

        n_present = sum(1 for d in densities if d[1] > 0)
        if n_present < min_runs:
            continue

        densities.sort(key=lambda x: -x[2])
        max_d = densities[0][2]
        mean_d = sum(d[2] for d in densities) / len(densities)
        spikiness = max_d / mean_d if mean_d > 0 else 0

        results.append(ThemeCloud(
            word=word, total_hits=total, n_runs_present=n_present,
            max_density=max_d, mean_density=mean_d,
            top_run=densities[0][0], top_count=densities[0][1],
            spikiness=spikiness,
        ))

    results.sort(key=lambda w: -w.total_hits)
    return results[:top_n], per_run, run_sizes


def find_co_occurrences(
    themes: list[ThemeCloud],
    per_run: dict[str, Counter],
    run_sizes: dict[str, int],
    density_factor: float = 2.0,
) -> list[ThemePair]:
    """Find theme pairs that co-occur in the same runs above threshold."""
    # For each theme, find runs where density > factor × mean
    dense_runs: dict[str, set[str]] = {}
    for tc in themes:
        threshold = tc.mean_density * density_factor
        dense = set()
        for name, n in run_sizes.items():
            if n == 0:
                continue
            if per_run[name].get(tc.word, 0) / n > threshold:
                dense.add(name)
        dense_runs[tc.word] = dense

    pairs = []
    words = [tc.word for tc in themes]
    for i, w1 in enumerate(words):
        for w2 in words[i + 1:]:
            overlap = dense_runs[w1] & dense_runs[w2]
            if not overlap:
                continue
            union = dense_runs[w1] | dense_runs[w2]
            jaccard = len(overlap) / len(union) if union else 0
            pairs.append(ThemePair(
                word_a=w1, word_b=w2,
                shared_runs=sorted(overlap), jaccard=jaccard,
            ))

    pairs.sort(key=lambda p: -p.jaccard)
    return pairs


def map_run_basins(
    runs: dict[str, RunInfo],
    themes: list[ThemeCloud],
    per_run: dict[str, Counter],
    run_sizes: dict[str, int],
    top_k: int = 5,
) -> list[RunBasin]:
    """For each run, identify its dominant semantic basin."""
    theme_words = {tc.word for tc in themes}
    results = []
    for name, run in runs.items():
        n = run_sizes.get(name, 0)
        if n == 0:
            continue
        densities = []
        for word in theme_words:
            count = per_run[name].get(word, 0)
            if count > 0:
                densities.append((word, count / n))
        densities.sort(key=lambda x: -x[1])
        top = densities[:top_k]
        label = " + ".join(w for w, _ in top[:3]) if top else "(empty)"
        results.append(RunBasin(
            run_name=name, L=run.L, T=run.T,
            top_themes=top, basin_label=label,
        ))
    results.sort(key=lambda r: (r.L, r.T))
    return results


def print_clouds_report(
    themes: list[ThemeCloud],
    pairs: list[ThemePair],
    basins: list[RunBasin],
) -> None:
    """Print the multi-theme cloud analysis."""
    print(f"\n{'=' * 90}")
    print(f"  SEMANTIC THEME MAP — {len(basins)} runs, {len(themes)} themes discovered")
    print(f"{'=' * 90}\n")

    print(f"  {'Word':18s} {'Total':>7s} {'Runs':>5s} {'MaxDens':>8s} {'MeanDens':>9s}"
          f" {'Spike':>6s} {'TopRun':>30s}")
    print("  " + "-" * 90)
    for tc in themes:
        print(f"  {tc.word:18s} {tc.total_hits:>7d} {tc.n_runs_present:>5d}"
              f" {tc.max_density:>8.4f} {tc.mean_density:>9.5f}"
              f" {tc.spikiness:>6.1f} {tc.top_run:>30s}")

    print(f"\n{'=' * 90}")
    print(f"  THEME CO-OCCURRENCE (Jaccard > 0, runs at >2× mean density)")
    print(f"{'=' * 90}\n")

    print(f"  {'Theme A':15s} {'Theme B':15s} {'Shared':>7s} {'Jaccard':>8s}  Shared runs")
    print("  " + "-" * 90)
    for p in pairs[:40]:
        run_list = ", ".join(p.shared_runs[:3])
        if len(p.shared_runs) > 3:
            run_list += f" +{len(p.shared_runs) - 3}"
        print(f"  {p.word_a:15s} {p.word_b:15s} {len(p.shared_runs):>7d}"
              f" {p.jaccard:>8.3f}  {run_list}")

    print(f"\n{'=' * 90}")
    print(f"  RUN BASINS — dominant themes per run")
    print(f"{'=' * 90}\n")

    print(f"  {'Run':35s} {'L':>4s} {'T':>5s}  Basin")
    print("  " + "-" * 85)
    for b in basins:
        theme_str = ", ".join(f"{w}({d:.3f})" for w, d in b.top_themes[:5])
        print(f"  {b.run_name:35s} {b.L:>4d} {b.T:>5.2f}  {theme_str}")


def run_clouds(all_runs: dict[str, RunInfo], csv_path: str | None = None) -> None:
    """Run the theme cloud discovery + co-occurrence + basin mapping."""
    log.info("Discovering themes...")
    themes, per_run, run_sizes = discover_themes(all_runs)
    log.info(f"Found {len(themes)} themes")

    log.info("Finding co-occurrences...")
    pairs = find_co_occurrences(themes, per_run, run_sizes)
    log.info(f"Found {len(pairs)} co-occurring pairs")

    log.info("Mapping run basins...")
    basins = map_run_basins(all_runs, themes, per_run, run_sizes)

    print_clouds_report(themes, pairs, basins)

    if csv_path:
        rows = []
        for b in basins:
            row = {"run": b.run_name, "L": b.L, "T": b.T, "basin": b.basin_label}
            for word, density in b.top_themes:
                row[f"theme_{word}"] = density
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log.info(f"Wrote {csv_path}")
