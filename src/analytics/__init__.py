"""Offline-model analytics: bootstrap-certified metrics, curves, degradation, edge,
SHAP cohorts, virtual-ensemble uncertainty, production audits.

Bootstrap policy
----------------

The headline-metric bundle (``metrics``), curves (``curves``), edge analytics
(``edge``), degradation diagnostics (``degradation``), and the SHAP-cohort
bootstrap (``cohorts.bootstrap_shap_diff``) all accept an optional
``block_size`` keyword.

- **Boundary cadence** (legacy non-overlapping labels): leave ``block_size``
  at its default (``None``) — labels are independent by construction so
  class-stratified iid bootstrap gives correct CIs.
- **1-min cadence** (overlapping M-bar barrier labels): pass ``block_size``
  ≈ ``M`` (the label horizon) so the bootstrap switches to a moving-block
  resampler. Adjacent labels share M−1 of their M future bars, so iid
  bootstrap underestimates CI width by roughly ``sqrt(M)``; block bootstrap
  preserves the within-block correlation structure.

Use ``analytics.sampling`` for purged train/val/test splits and per-row
label-uniqueness weights when training on overlapping 1-min labels.
"""
