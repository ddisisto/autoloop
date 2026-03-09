"""Rebuild decoded_text in existing parquet files from raw token_ids.

Fixes U+FFFD replacement characters produced by single-token decoding of
byte-level BPE tokens.  Uses the tokenizer to batch-decode groups of
consecutive affected tokens, recovering proper multi-byte UTF-8 characters.

Usage:
    python rebuild_decoded_text.py                  # preview changes (dry run)
    python rebuild_decoded_text.py --apply           # overwrite parquets in place
    python rebuild_decoded_text.py data/runs/L0064_T0.50_S42.parquet --apply  # single file
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from transformers.utils.logging import disable_progress_bar

from utils import fix_decoded_texts

log = logging.getLogger(__name__)

MODEL_DIR = Path("data/model/SmolLM-135M")
RUNS_DIR = Path("data/runs")


def rebuild_file(path: Path, tokenizer: AutoTokenizer, apply: bool) -> int:
    """Rebuild decoded_text for one parquet file.  Returns count of fixed tokens."""
    df = pd.read_parquet(path)
    if "token_id" not in df.columns or "decoded_text" not in df.columns:
        log.warning("Skipping %s: missing required columns", path.name)
        return 0

    token_ids = df["token_id"].tolist()
    old_texts = df["decoded_text"].tolist()

    n_bad = sum(1 for t in old_texts if "\ufffd" in t)
    if n_bad == 0:
        log.info("%-30s  no replacements found", path.name)
        return 0

    fixed = fix_decoded_texts(tokenizer, token_ids, old_texts)
    n_still_bad = sum(1 for t in fixed if "\ufffd" in t)
    n_fixed = n_bad - n_still_bad

    if apply:
        df["decoded_text"] = fixed
        df.to_parquet(path, index=False)
        log.info("%-30s  fixed %d/%d tokens (wrote)", path.name, n_fixed, n_bad)
    else:
        log.info("%-30s  would fix %d/%d tokens (dry run)", path.name, n_fixed, n_bad)

    return n_fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild decoded_text from token_ids")
    parser.add_argument("files", nargs="*", help="Parquet files (default: all in data/runs/)")
    parser.add_argument("--apply", action="store_true", help="Overwrite parquets in place")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(RUNS_DIR.glob("*.parquet"))

    if not paths:
        log.info("No parquet files found.")
        sys.exit(0)

    log.info("Loading tokenizer from %s", MODEL_DIR)
    disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    total_fixed = 0
    for p in paths:
        total_fixed += rebuild_file(p, tokenizer, args.apply)

    action = "Fixed" if args.apply else "Would fix"
    log.info("\n%s %d tokens across %d files.", action, total_fixed, len(paths))
    if not args.apply and total_fixed > 0:
        log.info("Run with --apply to write changes.")


if __name__ == "__main__":
    main()
