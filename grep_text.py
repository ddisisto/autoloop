#!/usr/bin/env python3
"""Grep decoded text in parquet run files.

Usage:
    python grep_text.py "Star Wars" --runs data/runs/ctrld_S42_8_1.00.parquet
    python grep_text.py "Star Wars" --runs data/runs/L0064_T*.parquet
    python grep_text.py "young|old" --regex --runs data/runs/*.parquet
    python grep_text.py "the end" --context 40 --runs data/runs/ctrld_S42_8_1.00.parquet
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd


def grep_run(
    path: Path,
    pattern: re.Pattern,
    context_tokens: int = 20,
    max_matches: int = 0,
) -> list[dict]:
    """Search decoded text in a parquet file. Returns match dicts."""
    cols = ["step", "decoded_text", "temperature"]
    # context_length only exists in controller/schedule runs
    try:
        df = pd.read_parquet(path, columns=cols + ["context_length"])
    except Exception:
        df = pd.read_parquet(path, columns=cols)
        df["context_length"] = 0
    texts = df["decoded_text"].tolist()
    steps = df["step"].tolist()

    # Build full text with token boundary positions
    full = ""
    boundaries: list[int] = []  # char offset where each token starts
    for t in texts:
        boundaries.append(len(full))
        full += t

    matches = []
    for m in pattern.finditer(full):
        start, end = m.start(), m.end()

        # Find token index of match start
        # Binary search: largest boundary <= start
        lo, hi = 0, len(boundaries) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if boundaries[mid] <= start:
                lo = mid
            else:
                hi = mid - 1
        tok_idx = lo

        # Context window
        ctx_start = max(0, tok_idx - context_tokens)
        ctx_end = min(len(texts), tok_idx + context_tokens + 1)

        # Build context string, marking match
        pre = "".join(texts[ctx_start:tok_idx])
        # Find how many tokens the match spans
        tok_end = tok_idx
        while tok_end < len(texts) - 1 and boundaries[tok_end + 1] < end:
            tok_end += 1
        match_text = "".join(texts[tok_idx:tok_end + 1])
        post = "".join(texts[tok_end + 1:ctx_end])

        step = steps[tok_idx]
        row = df.iloc[tok_idx]

        matches.append({
            "step": step,
            "L": int(row["context_length"]),
            "T": float(row["temperature"]),
            "pre": pre,
            "match": match_text,
            "post": post,
        })

        if max_matches and len(matches) >= max_matches:
            break

    return matches


def format_match(m: dict, run_name: str) -> str:
    """Format a single match for display."""
    header = f"  step {m['step']:>8d}  L={m['L']:>3d} T={m['T']:.3f}"
    # Highlight match with ANSI bold red
    text = f"{m['pre']}\033[1;31m{m['match']}\033[0m{m['post']}"
    # Collapse runs of whitespace for readability, keep single newlines
    text = re.sub(r"\n{2,}", "\n", text)
    return f"{header}  ...{text}..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Grep decoded text in parquet run files")
    parser.add_argument("pattern", help="Search pattern (literal unless --regex)")
    parser.add_argument("--runs", nargs="+", required=True, help="Parquet files or globs")
    parser.add_argument("--regex", action="store_true", help="Treat pattern as regex")
    parser.add_argument("--ignore-case", "-i", action="store_true", help="Case-insensitive")
    parser.add_argument("--context", "-C", type=int, default=20, help="Context tokens (default 20)")
    parser.add_argument("--max", "-m", type=int, default=0, help="Max matches per file (0=all)")
    parser.add_argument("--count", "-c", action="store_true", help="Only show match counts")
    args = parser.parse_args()

    # Expand globs
    paths: list[Path] = []
    for r in args.runs:
        expanded = sorted(glob.glob(r))
        if not expanded:
            print(f"Warning: no files match {r}", file=sys.stderr)
        paths.extend(Path(p) for p in expanded if p.endswith(".parquet"))

    if not paths:
        print("No parquet files found.", file=sys.stderr)
        sys.exit(1)

    # Compile pattern
    flags = re.IGNORECASE if args.ignore_case else 0
    if args.regex:
        pat = re.compile(args.pattern, flags)
    else:
        pat = re.compile(re.escape(args.pattern), flags)

    total = 0
    for path in paths:
        run_name = path.stem
        matches = grep_run(path, pat, args.context, args.max)
        total += len(matches)

        if args.count:
            if matches:
                print(f"{run_name}: {len(matches)}")
            continue

        if matches:
            print(f"\n\033[1m{run_name}\033[0m  ({len(matches)} matches)")
            for m in matches:
                print(format_match(m, run_name))

    if args.count:
        print(f"\nTotal: {total}")
    elif total == 0:
        print("No matches found.")


if __name__ == "__main__":
    main()
