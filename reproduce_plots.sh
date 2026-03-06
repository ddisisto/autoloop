#!/usr/bin/env bash
# reproduce.sh — Regenerate all standard plots from available run data.
#
# Produces plots for every useful slice of the pilot sweep:
#   - Per context length (all temperatures, seed 42)
#   - Per temperature (all context lengths, seed 42)
#   - All seed-42 runs together
#
# Usage:
#   bash data/figures/reproduce.sh        # from repo root
#   bash reproduce.sh                     # from data/figures/
#
# Requires: python plot.py (see plot.py --help for dependencies)

set -euo pipefail

# Change to repo root (two levels up from this script)
cd "$(dirname "$0")/../.."

run_plot() {
    local label="$1"
    local glob="$2"
    # shellcheck disable=SC2086
    if compgen -G $glob > /dev/null 2>&1; then
        echo "==> Plotting: $label"
        # shellcheck disable=SC2086
        python plot.py --runs $glob
    else
        echo "==> Skipping: $label (no files match $glob)"
    fi
}

# --- Per context length, all temperatures (seed 42) ---
run_plot "L=64, all T, seed 42"   "data/runs/L0064_T*_S42.parquet"
run_plot "L=256, all T, seed 42"  "data/runs/L0256_T*_S42.parquet"
run_plot "L=1024, all T, seed 42" "data/runs/L1024_T*_S42.parquet"

# --- Per temperature, all context lengths (seed 42) ---
run_plot "T=0.50, all L, seed 42" "data/runs/L*_T0.50_S42.parquet"
run_plot "T=1.00, all L, seed 42" "data/runs/L*_T1.00_S42.parquet"
run_plot "T=1.50, all L, seed 42" "data/runs/L*_T1.50_S42.parquet"

# --- All seed-42 runs together ---
run_plot "All runs, seed 42"      "data/runs/L*_T*_S42.parquet"

echo "==> Done."
