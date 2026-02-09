# Minimal Barrier-Crossing Classifier (BTCUSDT 1m)

Offline binary classification on Binance 1-minute klines, predicting whether price crosses an upward log-return barrier within a fixed horizon.

This repository implements `docs/MINIMAL_PROJECT_SPEC_v2.md` (Minimal Barrier-Crossing Classifier Specification v4.0) as the single source of truth, including strict causality invariants (no lookahead bias), validation checkpoints, and required output artifacts.

## Quick Start (Appendix C)

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Create directory structure
mkdir -p data/raw_data/zips data/model_dataset/plots data/model_dataset/analytics

# 3. Run notebooks in order
cd notebooks
jupyter notebook

# Execute sequentially:
# - 01_data_download.ipynb   (downloads ~2GB, takes 10-30 min)
# - 02_feature_building.ipynb (compute features, ~5-15 min)
# - 03_model_training.ipynb  (train and evaluate, ~10-30 min)
```

## Outputs

Running the notebooks produces:

- Raw data: `data/raw_data/klines_1m.parquet`, `data/raw_data/download_manifest.json`, `data/raw_data/validation_report.json`
- Model dataset: `data/model_dataset/dataset.parquet`, `data/model_dataset/dataset_metadata.json`, `data/model_dataset/feature_list.json`
- Model + evaluation: `data/model_dataset/catboost_model.cbm`, `data/model_dataset/analytics/metrics.json`, `data/model_dataset/analytics/calibration_by_regime.json`, `data/model_dataset/analytics/threshold_analysis.csv`, `data/model_dataset/analytics/best_hyperparameters.json`
- Required plots (Section 12.1): `data/model_dataset/plots/roc_curve.png`, `data/model_dataset/plots/pr_curve.png`, `data/model_dataset/plots/calibration_curve.png`, `data/model_dataset/plots/calibration_by_regime.png`, `data/model_dataset/plots/feature_analysis.png`, `data/model_dataset/plots/prediction_distribution.png`, `data/model_dataset/plots/threshold_analysis.png`
