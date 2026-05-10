# tsfm-anomaly-bench

**A rigorous head-to-head benchmark of time-series foundation models against strong classical baselines on multivariate anomaly detection.**

Foundation models like Chronos, TimesFM, and Moment have generated real excitement, but published comparisons often pit them against weak baselines or shift between datasets. This repo asks a narrower question: *on a standard multivariate anomaly detection benchmark (SMD), do foundation models actually outperform a well-tuned classical baseline?*

## Status

🚧 In progress. Results table updates as experiments complete.

## Headline results — SMD (Server Machine Dataset)

| Method | Family | F1 | Precision | Recall | Notes |
|---|---|---|---|---|---|
| Isolation Forest (tuned) | Classical | TBD | TBD | TBD | Per-machine tuning |
| LSTM Autoencoder | Classical-DL | – | – | – | Pending |
| OmniAnomaly | Classical-DL | – | – | – | Reproduced from paper |
| Chronos-T5-base | Foundation | – | – | – | Pending |
| TimesFM | Foundation | – | – | – | Pending |
| Moment-base | Foundation | – | – | – | Pending |

Metrics use point-adjusted F1 over a fixed evaluation window. See `src/evaluation/` for definitions.

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
git clone https://github.com/<you>/tsfm-anomaly-bench
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

Built by Necdet Duruk, Senior ML Engineer with 6 years in production time-series ML at scale. [https://www.linkedin.com/in/necdetduruk/] · [Blog]

## License

MIT
