# tsfm-anomaly-bench

**A rigorous head-to-head benchmark of time-series foundation models against strong classical baselines on multivariate anomaly detection.**

Foundation models like Chronos, TimesFM, and Moment have generated real excitement, but published comparisons often pit them against weak baselines or shift between datasets. This repo asks a narrower question: *on a standard multivariate anomaly detection benchmark (SMD), do foundation models actually outperform a well-tuned classical baseline?*

## Status

🚧 In progress. Results table updates as experiments complete.

## Headline results — SMD (Server Machine Dataset)

Aggregate over 28 machines × 3 seeds.

| Method | Family | Best F1 | PA Best F1 | AUC-PR | Notes |
|---|---|---|---|---|---|
| Isolation Forest | Classical | 0.326 ± 0.202 | 0.842 ± 0.153 | 0.277 ± 0.212 | Per-timestep, no temporal context |
| LSTM Autoencoder | Classical-DL | – | – | – | Pending |
| Chronos-T5-base | Foundation | – | – | – | Pending |
| TimesFM | Foundation | – | – | – | Pending |
| Moment-base | Foundation | – | – | – | Pending |

**Notes on metrics.** Best F1 and PA Best F1 both use per-test-set threshold tuning. The gap between them (0.33 vs 0.84 for IF) reflects the inflation effect of point-adjustment, which the literature widely reports despite known weaknesses ([Garg et al., 2021](https://arxiv.org/abs/2109.05257)). AUC-PR is threshold-free and the most rigorous of the three. The per-machine F1 spread (0.07 to 0.85) shows SMD is highly heterogeneous; aggregate numbers hide a lot.

## Why SMD

- 28 server machines × 38 metrics. Multivariate, real-world flavor.
- Strong literature baselines (OmniAnomaly, USAD, MTAD-GAT, TranAD) make comparisons concrete.
- Recent foundation-model papers have reported on it, so reproductions can be sanity-checked.

## Methodology

- Per-machine train/val/test splits per the SMD paper.
- Three random seeds per run, mean ± std reported.
- Hyperparameters tuned on validation only; test set untouched until final eval.
- All runs logged to Weights & Biases. Reproduction commands in `scripts/`.

## Reproduce

```bash
git clone https://github.com/necdetduruk/tsfm-anomaly-bench
cd tsfm-anomaly-bench
uv sync
bash scripts/download_smd.sh
uv run python -m src.experiments.run_iforest --machine all --seeds 0 1 2
```

W&B project: *link pending*

## Companion writeups

- Part 1: Setting up the benchmark — *coming soon*
- Part 2: Foundation models vs. classical baselines — *coming soon*

## About

Built by [Necdet Duruk](https://www.linkedin.com/in/necdetduruk/), Senior ML Engineer with 6 years in production time-series ML at scale. [Blog]

## License

MIT