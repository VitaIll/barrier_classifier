"""Offline-model analytics: bootstrap-certified metrics, curves, degradation, edge,
SHAP cohorts, virtual-ensemble uncertainty, production audits.

Boundary samples are non-overlapping by construction (M-step cadence equals M-bar
horizon -> zero label concurrency), so iid bootstrap is the correct primitive
here; block bootstrap is not needed for headline metrics.
"""
