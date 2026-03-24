# Compressibility-Aware Training via Hoyer Sparsity Regularization

## Core Idea

All submissions are zlib-compressed after int8/int6 quantization. The compression ratio depends
entirely on the *entropy* of the quantized byte stream — not the raw weight magnitudes.

Per-row quantization normalizes each row by its own maximum before converting to int8. This means
absolute weight scale is irrelevant; only the **shape** of each row's distribution matters.

| Row distribution | Hoyer L1/L2 | int8 entropy | zlib ratio |
|-----------------|-------------|--------------|------------|
| Gaussian (Muon) | ~0.80 | ~7.0 bits | ~1.0x (barely compresses) |
| 70% sparse      | ~0.45 | ~3.5 bits | ~1.5x |
| 90% sparse      | ~0.28 | ~2.1 bits | ~2.1x |

**The Muon problem**: Newton-Schulz orthogonalization drives weight matrices toward uniform
singular values, which produces near-Gaussian row distributions — maximum entropy, worst-case
compressibility. Weight decay doesn't help because per-row scaling normalizes it out.

**The fix**: Add a Hoyer sparsity regularizer (L1/L2 per row) that explicitly counteracts
Muon's orthogonalizing tendency, applied during warmdown when the LR is annealing.

If we can push `est_compress_ratio` from 1.08x (current baseline) to 1.3x, we unlock
~2 extra layers within the same 16MB budget — the architecture budget analyzer tells you exactly.

## New Features

### `compression_regularizer(model)` — the B innovation
Penalizes the mean Hoyer L1/L2 ratio across all large weight matrices. Minimizing this
pushes rows toward spiky distributions (few dominant values), directly reducing int8 entropy.
Applied to the last micro-step gradient so it integrates cleanly with the existing optimizer.

### `measure_compression_metrics(model)` — live feedback
Logged every `COMPRESSION_LOG_EVERY` steps (default 1000):
```
step:15000 hoyer_mean:0.6832 est_compress_ratio:1.31x
budget_analysis: compress=1.31x int8 max_layers=11 max_params=20.1M
```

### `budget_analysis()` — the C methodology
Given the *measured* compression ratio, computes the optimal architecture for the next run.
Printed at init and at end of training. This structures architecture search around compressed
bytes rather than raw parameter counts.

## Running

### Setup (RunPod — official template has all deps pre-installed)
```bash
cd /workspace
git clone https://github.com/thedavidmurray/parameter-golf.git
cd parameter-golf
git checkout claude/parameter-golf-exploration-qLSR8
python3 data/cached_challenge_fineweb.py --variant sp1024
```

### Quick test (1×H100, ~5 min, verify hoyer signal)
```bash
bash records/track_10min_16mb/2026-03-24_CompressibilityReg_HoyerSparsity/run_sweep.sh smoke
```

### Full isolation sweep (1×H100, compare comp_reg_weight values)
```bash
bash records/track_10min_16mb/2026-03-24_CompressibilityReg_HoyerSparsity/run_sweep.sh sweep
```

### Full leaderboard run (8×H100, 10 min wall clock)
```bash
bash records/track_10min_16mb/2026-03-24_CompressibilityReg_HoyerSparsity/run_sweep.sh full8
```

## What to look for

| Signal | Interpretation |
|--------|----------------|
| `hoyer_mean` at step 0 | Should be ~0.79-0.80 (near-Gaussian init) |
| `hoyer_mean` drops during warmdown | Regularizer is working |
| `hoyer_mean` < 0.65 at end | Strong signal; compression is improved |
| `est_compress_ratio` > 1.2x | Budget freed for extra layer next run |
| val_bpb with reg ≈ val_bpb without reg | Regularizer not hurting capacity (good) |
| val_bpb with reg > val_bpb without reg | reg_weight too high, reduce it |

## Hyperparameters

| Env var | Default | Notes |
|---------|---------|-------|
| `COMP_REG_WEIGHT` | `1e-3` | Coefficient on Hoyer loss. Try 5e-4, 1e-3, 2e-3 |
| `COMP_REG_WARMDOWN_ONLY` | `1` | 1 = only during warmdown (safe). 0 = always (stronger) |
| `COMPRESSION_LOG_EVERY` | `1000` | Steps between compression metric logs |

## Next step if it works

Stack onto SOTA features from `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`:
- int6+lzma quantization (more params in budget)
- XSA attention (extra context)
- BigramHash embeddings
- Parallel Muon

With int6+lzma and improved compression ratio, `budget_analysis` should unlock 13+ layers.
