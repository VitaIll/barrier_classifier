# Changelog

## 2026-01-01
- Add optional HPO train truncation (drop oldest fraction) and explicit per-trial fold count in Optuna walk-forward CV, plus a tqdm fallback progress bar.
- Set CatBoost `border_count=128`, `thread_count=-1`, and `allow_writing_files=False` to reduce HPO runtime (notably on OneDrive-backed paths).
- Add Optuna NSGA-II multi-objective search (logloss, PR-AUC) with ordered CatBoost Pools and per-trial seed variation.
- Add Optuna Pareto/importance visualizations and learning curve plotting outputs.
- Add prediction-based feature importance plot and Pool-based evaluation outputs with best params/iteration saved.
- Add Optuna/plotly/kaleido dependencies and shared hyperparameter constants in utils.
- Add post-save target visualizations (return distribution and time series markers) in the feature building notebook.
