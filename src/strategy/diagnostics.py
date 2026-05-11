"""Pre-flight diagnostics that gate which StrategySpecs are eligible.

Each diagnostic is a small pure function returning a ``DiagnosticResult``
(passed / value / threshold / message / details). The calibration notebook
runs them on the val split, builds a ``{name: passed}`` dict, and passes it
to ``filter_specs_by_diagnostics``. A spec whose ``requires`` mentions a
diagnostic that didn't pass is silently dropped.

Framing: these are the same diagnostics you'd run on any selective-
classification system. ``ve_diag`` is the Geifman–El-Yaniv risk-coverage
test (does the rejection signal predict mistakes?). ``vol_gate_diag`` is
a conditional-precision check (does the gating covariate buy us anything?).
``cluster_persistence_diag`` tests whether high-confidence predictions
arrive in temporal clusters — relevant for any sequential-decision setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class DiagnosticResult:
    name: str
    passed: bool
    value: float
    threshold: float
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "value": float(self.value),
            "threshold": float(self.threshold),
            "message": self.message,
            "details": dict(self.details),
        }


# ---------------------------------------------------------------------------
# 1. VE / knowledge-uncertainty diagnostic (Geifman–El-Yaniv style)
# ---------------------------------------------------------------------------


def ve_diagnostic(
    y: np.ndarray,
    mean_p: np.ndarray,
    knowledge_unc: np.ndarray,
    *,
    score_top_quantile: float = 0.20,
    pass_uplift: float = 0.04,
    min_per_subset: int = 50,
) -> DiagnosticResult:
    """Within the high-``p`` subset (top ``score_top_quantile``), does dropping
    the high-MI half improve precision?

    Decision-relevant framing (matches strategy use): we only act when ``p``
    is high; the question is whether MI tells us *which* high-``p`` predictions
    to trust. Compute precision at top ``score_top_quantile`` of ``p``, then
    split that subset by median MI and report ``precision_low_mi -
    precision_high_mi``. Pass when the uplift exceeds ``pass_uplift``.

    Robust to small samples and constant MI by short-circuiting before the
    quantile split.
    """
    y = np.asarray(y).astype(int)
    mean_p = np.asarray(mean_p, dtype=float)
    knowledge_unc = np.asarray(knowledge_unc, dtype=float)
    if not (len(y) == len(mean_p) == len(knowledge_unc)):
        raise ValueError("y, mean_p, knowledge_unc must have the same length")

    valid = np.isfinite(mean_p) & np.isfinite(knowledge_unc)
    if valid.sum() < 2 * min_per_subset:
        return DiagnosticResult(
            name="ve_diag",
            passed=False,
            value=float("nan"),
            threshold=pass_uplift,
            message=(
                f"insufficient data: {int(valid.sum())} valid rows, need >= "
                f"{2 * min_per_subset}"
            ),
            details={"n_valid": int(valid.sum())},
        )
    y_v, p_v, u_v = y[valid], mean_p[valid], knowledge_unc[valid]

    if u_v.std() < 1e-12:
        return DiagnosticResult(
            name="ve_diag",
            passed=False,
            value=float("nan"),
            threshold=pass_uplift,
            message="knowledge_uncertainty is (nearly) constant — no sortable variation",
            details={"unc_std": float(u_v.std())},
        )

    p_cut = float(np.quantile(p_v, 1.0 - score_top_quantile))
    sel = p_v >= p_cut
    n_sel = int(sel.sum())
    if n_sel < 2 * min_per_subset:
        return DiagnosticResult(
            name="ve_diag",
            passed=False,
            value=float("nan"),
            threshold=pass_uplift,
            message=(
                f"too few high-p rows ({n_sel} < {2 * min_per_subset}) — relax "
                f"score_top_quantile or get more data"
            ),
            details={"n_selected": n_sel},
        )

    u_in_sel = u_v[sel]
    u_med = float(np.median(u_in_sel))
    low_mi = sel & (u_v <= u_med)
    high_mi = sel & (u_v > u_med)

    n_low, n_high = int(low_mi.sum()), int(high_mi.sum())
    if n_low < min_per_subset or n_high < min_per_subset:
        return DiagnosticResult(
            name="ve_diag",
            passed=False,
            value=float("nan"),
            threshold=pass_uplift,
            message=(
                f"MI median split is too uneven (low={n_low}, high={n_high}); "
                f"both subsets need >= {min_per_subset}"
            ),
            details={"n_low_mi": n_low, "n_high_mi": n_high},
        )

    prec_low = float(y_v[low_mi].mean())
    prec_high = float(y_v[high_mi].mean())
    uplift = prec_low - prec_high
    passed = bool(uplift >= pass_uplift)
    return DiagnosticResult(
        name="ve_diag",
        passed=passed,
        value=float(uplift),
        threshold=float(pass_uplift),
        message=(
            f"top-{score_top_quantile:.0%} of p split by MI median: "
            f"precision_low_mi={prec_low:.3f} ({n_low}), "
            f"precision_high_mi={prec_high:.3f} ({n_high}), "
            f"uplift={uplift:+.3f} ({'PASS' if passed else 'FAIL'} @ {pass_uplift:+.2f})"
        ),
        details={
            "precision_low_mi": prec_low,
            "precision_high_mi": prec_high,
            "n_low_mi": n_low,
            "n_high_mi": n_high,
            "p_cut": p_cut,
            "u_median": u_med,
        },
    )


# ---------------------------------------------------------------------------
# 2. Vol-gate diagnostic (conditional precision)
# ---------------------------------------------------------------------------


def vol_gate_diagnostic(
    y: np.ndarray,
    p: np.ndarray,
    regime: np.ndarray,
    *,
    score_top_quantile: float = 0.10,
    regime_top_quantile: float = 0.30,
    pass_uplift: float = 0.05,
    min_trades: int = 50,
) -> DiagnosticResult:
    """Is precision higher in the top regime tercile vs all regimes pooled?

    Compares precision @ top-``score_top_quantile`` of ``p``:
    - all rows (no gate)
    - rows with ``regime`` in its top ``regime_top_quantile`` (vol gate ON)

    Pass if ``precision_gated - precision_all >= pass_uplift``.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    regime = np.asarray(regime, dtype=float)

    n = len(y)
    if not (len(p) == len(regime) == n):
        raise ValueError("y, p, regime must have the same length")

    score_cut = float(np.quantile(p, 1.0 - score_top_quantile))
    regime_cut = float(np.quantile(regime, 1.0 - regime_top_quantile))

    sel_all = p >= score_cut
    sel_gate = sel_all & (regime >= regime_cut)

    if int(sel_all.sum()) < min_trades:
        return DiagnosticResult(
            name="vol_gate",
            passed=False,
            value=float("nan"),
            threshold=pass_uplift,
            message=f"too few candidates ({int(sel_all.sum())} < {min_trades})",
            details={"n_all": int(sel_all.sum()), "n_gated": int(sel_gate.sum())},
        )

    prec_all = float(y[sel_all].mean())
    prec_gate = float(y[sel_gate].mean()) if sel_gate.any() else float("nan")
    uplift = prec_gate - prec_all
    passed = bool(np.isfinite(uplift) and uplift >= pass_uplift)
    return DiagnosticResult(
        name="vol_gate",
        passed=passed,
        value=float(uplift),
        threshold=float(pass_uplift),
        message=(
            f"precision @ top-{score_top_quantile:.0%} of p: all={prec_all:.3f}, "
            f"gated={prec_gate:.3f}, uplift={uplift:+.3f} "
            f"({'PASS' if passed else 'FAIL'} @ threshold={pass_uplift:+.2f})"
        ),
        details={
            "precision_all": prec_all,
            "precision_gated": prec_gate,
            "n_all": int(sel_all.sum()),
            "n_gated": int(sel_gate.sum()),
            "score_cut": score_cut,
            "regime_cut": regime_cut,
        },
    )


# ---------------------------------------------------------------------------
# 3. Cluster persistence diagnostic
# ---------------------------------------------------------------------------


def cluster_persistence_diagnostic(
    p: np.ndarray,
    *,
    threshold_quantile: float = 0.70,
    pass_lift: float = 1.5,
    min_above_threshold: int = 50,
) -> DiagnosticResult:
    """Is ``P(p_{k+1} > τ | p_k > τ)`` materially higher than the marginal?

    A lift > ``pass_lift`` (default 1.5×) means high-`p` predictions arrive
    in clusters — justifying any strategy element that conditions on the
    sequence rather than the single boundary (stacking, bulk-close).
    """
    p = np.asarray(p, dtype=float)
    if len(p) < 100:
        return DiagnosticResult(
            name="cluster_persistence",
            passed=False,
            value=float("nan"),
            threshold=pass_lift,
            message=f"too few rows ({len(p)} < 100)",
            details={"n": int(len(p))},
        )
    tau = float(np.quantile(p, threshold_quantile))
    above = p > tau
    n_above = int(above.sum())
    if n_above < min_above_threshold:
        return DiagnosticResult(
            name="cluster_persistence",
            passed=False,
            value=float("nan"),
            threshold=pass_lift,
            message=(
                f"too few rows above threshold ({n_above} < {min_above_threshold})"
            ),
            details={"n_above": n_above, "tau": tau},
        )

    # P(above_{k+1} | above_k) using lag-1 conditioning
    above_kplus1 = np.r_[above[1:], False]  # shift -1
    n_pair_above = int((above & above_kplus1).sum())
    n_above_excl_last = int(above[:-1].sum()) if len(above) > 1 else 0
    if n_above_excl_last == 0:
        return DiagnosticResult(
            name="cluster_persistence",
            passed=False,
            value=float("nan"),
            threshold=pass_lift,
            message="no above-threshold rows except possibly the last",
            details={"n_above": n_above},
        )
    p_marginal = float(above.mean())
    p_conditional = float(n_pair_above / n_above_excl_last)
    lift = p_conditional / p_marginal if p_marginal > 0 else float("nan")
    passed = bool(np.isfinite(lift) and lift >= pass_lift)
    return DiagnosticResult(
        name="cluster_persistence",
        passed=passed,
        value=lift,
        threshold=pass_lift,
        message=(
            f"P(above_{{k+1}} | above_k) = {p_conditional:.3f} vs marginal {p_marginal:.3f} → "
            f"lift {lift:.2f}× ({'PASS' if passed else 'FAIL'} @ threshold {pass_lift:.1f}×)"
        ),
        details={
            "p_conditional": p_conditional,
            "p_marginal": p_marginal,
            "tau": tau,
            "n_above": n_above,
        },
    )


# ---------------------------------------------------------------------------
# 4. Within-regime signal diagnostic (does p discriminate inside a regime?)
# ---------------------------------------------------------------------------


def within_regime_signal_diagnostic(
    y: np.ndarray,
    p: np.ndarray,
    regime: np.ndarray,
    *,
    n_terciles: int = 3,
    pass_threshold: float = 0.55,
    min_per_tercile: int = 100,
) -> DiagnosticResult:
    """Within each regime tercile, does ``p`` have AUC above ``pass_threshold``?

    The headline AUC double-counts the cross-regime base-rate effect; this
    diagnostic isolates the *intra*-regime signal. Pass if median tercile
    AUC >= threshold."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    regime = np.asarray(regime, dtype=float)
    if not (len(p) == len(regime) == len(y)):
        raise ValueError("y, p, regime must have the same length")

    terciles = pd.qcut(regime, n_terciles, labels=False, duplicates="drop")
    aucs: dict[int, float] = {}
    for t in sorted(set(int(x) for x in terciles if not pd.isna(x))):
        mask = (terciles == t)
        if mask.sum() < min_per_tercile:
            continue
        y_t = y[np.asarray(mask)]
        p_t = p[np.asarray(mask)]
        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            continue
        aucs[t] = float(roc_auc_score(y_t, p_t))

    if not aucs:
        return DiagnosticResult(
            name="within_regime_signal",
            passed=False,
            value=float("nan"),
            threshold=pass_threshold,
            message="no regime tercile met min sample / both-class requirements",
            details={},
        )
    median_auc = float(np.median(list(aucs.values())))
    passed = bool(median_auc >= pass_threshold)
    return DiagnosticResult(
        name="within_regime_signal",
        passed=passed,
        value=median_auc,
        threshold=pass_threshold,
        message=(
            f"median within-regime AUC = {median_auc:.3f} "
            f"({'PASS' if passed else 'FAIL'} @ threshold {pass_threshold:.2f})"
        ),
        details={"aucs_per_tercile": aucs},
    )


# ---------------------------------------------------------------------------
# Bundle helper
# ---------------------------------------------------------------------------


def run_all_diagnostics(
    y: np.ndarray,
    p: np.ndarray,
    regime: np.ndarray,
    *,
    mean_p_ve: Optional[np.ndarray] = None,
    knowledge_unc: Optional[np.ndarray] = None,
) -> dict[str, DiagnosticResult]:
    """Run every diagnostic supported by the data; return a dict by name.

    ``ve_diag`` is skipped (and recorded as a non-pass) if VE quantities
    aren't supplied.
    """
    out: dict[str, DiagnosticResult] = {}
    out["vol_gate"] = vol_gate_diagnostic(y, p, regime)
    out["cluster_persistence"] = cluster_persistence_diagnostic(p)
    out["within_regime_signal"] = within_regime_signal_diagnostic(y, p, regime)
    if mean_p_ve is not None and knowledge_unc is not None:
        out["ve_diag"] = ve_diagnostic(y, mean_p_ve, knowledge_unc)
    else:
        out["ve_diag"] = DiagnosticResult(
            name="ve_diag",
            passed=False,
            value=float("nan"),
            threshold=-0.20,
            message="VE quantities (mean_p_ve, knowledge_unc) not provided",
            details={},
        )
    return out


def passed_flags(results: dict[str, DiagnosticResult]) -> dict[str, bool]:
    """Convenience: extract ``{name: passed}`` for ``filter_specs_by_diagnostics``."""
    return {name: bool(r.passed) for name, r in results.items()}
