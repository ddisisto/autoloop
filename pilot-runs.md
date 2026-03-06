# Pilot Grid Runs (Phase 0)

## Command Template

```
python generate.py --context-length {L} --temperature {T} --seed {seed} --num-tokens 100000 --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda
```

## Runs

Priority order: sweep temperatures first, then context lengths, then additional seeds.

| L    | T    | Seed | Status  | Notes |
|------|------|------|---------|-------|
| 64   | 0.5  | 42   | done    | 100k |
| 64   | 1.0  | 42   | done    | 100k |
| 64   | 1.5  | 42   | done    | 100k |
| 256  | 0.5  | 42   | done    | 100k |
| 256  | 1.0  | 42   | done    | 100k |
| 256  | 1.5  | 42   | running | 75k/100k |
| 1024 | 0.5  | 42   |         | deprioritized — see observations 2026-03-07 |
| 1024 | 1.0  | 42   |         | deprioritized |
| 1024 | 1.5  | 42   |         | deprioritized |
| 128  | 0.5  | 42   |         | new: L-densification |
| 128  | 1.0  | 42   |         | new: L-densification |
| 128  | 1.5  | 42   |         | new: L-densification |
| 192  | 0.5  | 42   |         | new: L-densification |
| 192  | 1.0  | 42   |         | new: L-densification |
| 192  | 1.5  | 42   |         | new: L-densification |
| 64   | 0.5  | 123  |         | replicate seeds — lower priority |
| 64   | 1.0  | 123  |         |       |
| 64   | 1.5  | 123  |         |       |
| 128  | 0.5  | 123  |         |       |
| 128  | 1.0  | 123  |         |       |
| 128  | 1.5  | 123  |         |       |
| 192  | 0.5  | 123  |         |       |
| 192  | 1.0  | 123  |         |       |
| 192  | 1.5  | 123  |         |       |
| 256  | 0.5  | 123  |         |       |
| 256  | 1.0  | 123  |         |       |
| 256  | 1.5  | 123  |         |       |
| 64   | 0.5  | 7    |         |       |
| 64   | 1.0  | 7    |         |       |
| 64   | 1.5  | 7    |         |       |
| 128  | 0.5  | 7    |         |       |
| 128  | 1.0  | 7    |         |       |
| 128  | 1.5  | 7    |         |       |
| 192  | 0.5  | 7    |         |       |
| 192  | 1.0  | 7    |         |       |
| 192  | 1.5  | 7    |         |       |
| 256  | 0.5  | 7    |         |       |
| 256  | 1.0  | 7    |         |       |
| 256  | 1.5  | 7    |         |       |
