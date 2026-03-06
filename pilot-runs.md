# Pilot Grid Runs (Phase 0)

## Command Template

```
python generate.py --context-length {L} --temperature {T} --seed {seed} --num-tokens 100000 --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda
```

## Runs

Priority order: sweep temperatures first, then context lengths, then additional seeds.

| L    | T    | Seed | Status  | Notes |
|------|------|------|---------|-------|
| 64   | 0.5  | 42   | running |       |
| 64   | 1.0  | 42   |         |       |
| 64   | 1.5  | 42   |         |       |
| 256  | 0.5  | 42   |         |       |
| 256  | 1.0  | 42   |         |       |
| 256  | 1.5  | 42   |         |       |
| 1024 | 0.5  | 42   |         |       |
| 1024 | 1.0  | 42   |         |       |
| 1024 | 1.5  | 42   |         |       |
| 64   | 0.5  | 123  |         |       |
| 64   | 1.0  | 123  |         |       |
| 64   | 1.5  | 123  |         |       |
| 256  | 0.5  | 123  |         |       |
| 256  | 1.0  | 123  |         |       |
| 256  | 1.5  | 123  |         |       |
| 1024 | 0.5  | 123  |         |       |
| 1024 | 1.0  | 123  |         |       |
| 1024 | 1.5  | 123  |         |       |
| 64   | 0.5  | 7    |         |       |
| 64   | 1.0  | 7    |         |       |
| 64   | 1.5  | 7    |         |       |
| 256  | 0.5  | 7    |         |       |
| 256  | 1.0  | 7    |         |       |
| 256  | 1.5  | 7    |         |       |
| 1024 | 0.5  | 7    |         |       |
| 1024 | 1.0  | 7    |         |       |
| 1024 | 1.5  | 7    |         |       |
