"""Semantic analysis report printing and CLI entry points.

Contains print_report (full single-theme analysis), run_themes (compact
multi-theme density), and run_full_analysis orchestration.
"""

import logging
from collections import Counter

import pandas as pd

from .analyze.semantic import HeapsLaw, CoherenceProfile, RepetitionOnset
from .semantic import (
    RunInfo,
    ThemeHit,
    AttractorInfo,
    RunRepetitionOnset,
    RunHeapsLaw,
    RunCoherenceProfile,
    find_theme_hits,
    vocab_stats,
    neighbor_profile,
    neighbor_morphology,
    attractor_catalog,
    detect_repetition_onset,
    fit_heaps_law,
    measure_coherence,
)

log = logging.getLogger(__name__)


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


def run_themes(
    all_runs: dict[str, RunInfo],
    themes: list[str],
    context_radius: int,
    entropy_window: int,
) -> None:
    """Run theme search for one or more specific themes (compact report)."""
    for theme in themes:
        log.info(f"Searching for '{theme}'...")
        all_hits: list[ThemeHit] = []
        for run in all_runs.values():
            hits = find_theme_hits(run, theme, context_radius, entropy_window)
            all_hits.extend(hits)

        log.info(f"  {len(all_hits)} hits across"
                 f" {len(set(h.run_name for h in all_hits))} runs")

        neighbors = neighbor_profile(all_hits, all_runs)

        # Compact per-theme report
        print(f"\n{'=' * 70}")
        print(f"  '{theme}' — {len(all_hits)} hits across"
              f" {len(set(h.run_name for h in all_hits))}/{len(all_runs)} runs")
        print(f"{'=' * 70}")

        # Top runs by density
        by_run: dict[str, list[ThemeHit]] = {}
        for h in all_hits:
            by_run.setdefault(h.run_name, []).append(h)

        densities = []
        for rname, rhits in by_run.items():
            run = all_runs[rname]
            total_chars = len(run.text)
            rate = len(rhits) / total_chars * 100000 if total_chars > 0 else 0
            densities.append((rname, len(rhits), rate))
        densities.sort(key=lambda x: -x[2])

        print(f"\n  Top runs:")
        for rname, count, rate in densities[:8]:
            bar = "#" * min(int(rate / 5), 40)
            print(f"    {rname:35s} {count:4d} ({rate:6.1f}/100k) {bar}")

        # Neighbor cloud
        for t_val in sorted(neighbors):
            top = neighbors[t_val].most_common(15)
            top_str = ", ".join(f"{w}({c})" for w, c in top)
            print(f"\n  Neighbors (T={t_val:.2f}): {top_str}")

        # Morphological crossover
        morph = neighbor_morphology(all_hits, all_runs, theme)
        if morph:
            print(f"\n  Morphological crossover ('{theme}'):")
            print(f"    {'T':>5s}  {'morph':>6s}  {'subword':>7s}  {'n':>5s}")
            for t_val in sorted(morph):
                mr, sr, n = morph[t_val]
                bar_m = "#" * int(mr * 40)
                bar_s = ":" * int(sr * 40)
                print(f"    {t_val:>5.2f}  {mr:>6.3f}  {sr:>7.3f}  {n:>5d}  {bar_m}{bar_s}")


def run_full_analysis(
    all_runs: dict[str, RunInfo],
    theme: str,
    context_radius: int,
    entropy_window: int,
    csv_path: str | None = None,
) -> None:
    """Run the original full single-theme analysis."""
    # Find theme hits
    log.info(f"Searching for '{theme}'...")
    all_hits: list[ThemeHit] = []
    for run in all_runs.values():
        hits = find_theme_hits(run, theme, context_radius, entropy_window)
        all_hits.extend(hits)

    log.info(f"Found {len(all_hits)} hits across {len(set(h.run_name for h in all_hits))} runs")

    neighbors = neighbor_profile(all_hits, all_runs)

    log.info("Computing vocabulary stats...")
    vocab_list = [vocab_stats(run) for run in all_runs.values()]

    log.info("Building attractor catalog...")
    attractors = attractor_catalog(all_runs)

    log.info("Detecting repetition onset...")
    repetitions = [detect_repetition_onset(run) for run in all_runs.values()]

    log.info("Fitting Heaps' law...")
    heaps = [fit_heaps_law(run) for run in all_runs.values()]

    log.info("Measuring semantic coherence...")
    coherence_results = [measure_coherence(run) for run in all_runs.values()]

    print_report(
        theme, all_runs, all_hits, vocab_list, neighbors,
        attractors, repetitions, heaps, coherence_results,
    )

    if csv_path:
        rows = []
        heaps_by_name = {h.run_name: h for h in heaps}
        coh_by_name = {c.run_name: c for c in coherence_results}
        rep_by_name = {r.run_name: r for r in repetitions}
        attr_by_name = {a.run_name: a for a in attractors}

        for v in vocab_list:
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

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        log.info(f"Wrote {csv_path}")
