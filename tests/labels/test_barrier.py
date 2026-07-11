"""Barrier-label domain: spec, vectorized kernel (vs frozen legacy oracle),
block wrapper, and the frame-alignment guard on the legacy adapter.

The oracle below is the pre-refactor per-row loop from
``src/features/boundary.py`` (2026-05 revision), copied VERBATIM in its
numerical semantics. The kernel must reproduce it bit-for-bit — including
NaN propagation, non-positive-price behavior, and first-crossing ties —
across cadences and pathologies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.core.contracts import assert_unmutated
from src.core.errors import ConfigError, ContractError
from src.labels.barrier import (
    LABEL_SCHEMA,
    BarrierLabeler,
    BarrierSpec,
    barrier_label_arrays,
    label_frame,
)

pytestmark = pytest.mark.labels


# ---------------------------------------------------------------------------
# Frozen oracle — the legacy loop, semantics copied verbatim.
# ---------------------------------------------------------------------------


def legacy_loop_labels(
    close: np.ndarray,
    upper: np.ndarray,
    K: int,
    M: int,
    phi: float,
    bar_stride: int,
    low: np.ndarray | None = None,
):
    n_total = len(close)
    add_dn = low is not None
    y = np.full(K, np.nan)
    m_k = np.full(K, np.nan)
    tau_k = np.full(K, np.nan)
    m_dn = np.full(K, np.nan) if add_dn else None
    tau_dn = np.full(K, np.nan) if add_dn else None

    for k in range(K):
        n_k = k * bar_stride
        if n_k + M >= n_total:
            continue
        base = close[n_k]
        future_up = upper[n_k + 1 : n_k + M + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            future_up_ret = np.log(future_up / base)
        m_val = float(np.max(future_up_ret))
        if not np.isfinite(m_val):
            y[k] = 0.0
            continue
        m_k[k] = m_val
        hit = m_val >= phi
        y[k] = 1.0 if hit else 0.0
        if hit:
            tau_k[k] = float(np.argmax(future_up_ret >= phi) + 1)
        if add_dn:
            future_dn = low[n_k + 1 : n_k + M + 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                future_dn_ret = np.log(future_dn / base)
            m_dn_val = float(-np.min(future_dn_ret))
            if not np.isfinite(m_dn_val):
                continue
            m_dn[k] = m_dn_val
            dn_hit = future_dn_ret <= -phi
            if dn_hit.any():
                tau_dn[k] = float(np.argmax(dn_hit) + 1)
    return y, m_k, tau_k, m_dn, tau_dn


def make_prices(n: int, seed: int, *, pathologies: bool = True):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.0009, n)
    close = 50_000 * np.exp(np.cumsum(r))
    high = close * np.exp(np.abs(rng.normal(0, 0.0005, n)))
    low = close * np.exp(-np.abs(rng.normal(0, 0.0005, n)))
    if pathologies and n >= 40:
        bad = rng.choice(n, size=8, replace=False)
        high[bad[0]] = np.nan          # NaN in the barrier series
        high[bad[1]] = 0.0             # zero -> log(-inf) path
        high[bad[2]] = -5.0            # negative -> NaN path
        low[bad[3]] = np.nan
        low[bad[4]] = 0.0
        close[bad[5]] = np.nan         # NaN base
        close[bad[6]] = 0.0            # zero base -> inf ratio
        close[bad[7]] = -1.0           # negative base
    return close, high, low


def assert_bit_equal(a: np.ndarray, b: np.ndarray, name: str) -> None:
    assert np.array_equal(a, b, equal_nan=True), (
        f"{name}: kernel diverges from the legacy oracle "
        f"({int((~(np.isclose(a, b, equal_nan=True))).sum())} cells)"
    )


# ---------------------------------------------------------------------------
# BarrierSpec
# ---------------------------------------------------------------------------


class TestBarrierSpec:
    def test_validation(self):
        with pytest.raises(ConfigError):
            BarrierSpec(horizon=0, upper=0.0025)
        with pytest.raises(ConfigError):
            BarrierSpec(horizon=20, upper=float("nan"))
        with pytest.raises(ConfigError):
            BarrierSpec(horizon=20, upper=-0.001)
        with pytest.raises(ConfigError):
            BarrierSpec(horizon=20, upper=0.0025, source="open")  # type: ignore[arg-type]
        with pytest.raises(ConfigError):
            BarrierSpec(horizon=20, upper=0.0025, stride=0)

    def test_maturity_shift(self):
        assert BarrierSpec(horizon=20, upper=0.0025, stride=1).maturity_shift == 20
        assert BarrierSpec(horizon=20, upper=0.0025, stride=20).maturity_shift == 1
        assert BarrierSpec(horizon=20, upper=0.0025, stride=7).maturity_shift == 2
        # Shift never collapses below one row.
        assert BarrierSpec(horizon=3, upper=0.0025, stride=20).maturity_shift == 1

    def test_label_intervals_matches_analytics_sampling(self):
        # One source of truth: the spec's intervals must equal the
        # convention already used by uniqueness weights / purged splits.
        from src.analytics.sampling import label_intervals

        spec = BarrierSpec(horizon=20, upper=0.0025, stride=1)
        np.testing.assert_array_equal(
            spec.label_intervals(1000), label_intervals(1000, 20, bar_stride=1)
        )
        spec_b = BarrierSpec(horizon=20, upper=0.0025, stride=20)
        np.testing.assert_array_equal(
            spec_b.label_intervals(50), label_intervals(50, 20, bar_stride=20)
        )

    def test_spec_is_hashable_value_object(self):
        a = BarrierSpec(horizon=20, upper=0.0025)
        b = BarrierSpec(horizon=20, upper=0.0025)
        assert a == b and hash(a) == hash(b)

    def test_spec_owns_labeling(self):
        # spec.label(bars) is the owner-attached form of label_frame.
        spec = BarrierSpec(horizon=20, upper=0.0025)
        bars = _bar_frame(200)
        via_method = spec.label(bars)
        via_function = label_frame(bars, spec)
        assert via_method.equals(via_function)

    def test_spec_owns_uniqueness_weights(self):
        from src.analytics.sampling import compute_uniqueness_weights

        spec = BarrierSpec(horizon=20, upper=0.0025, stride=1)
        got = spec.uniqueness_weights(500, normalize=False)
        exp = compute_uniqueness_weights(500, 20, bar_stride=1, normalize=False)
        np.testing.assert_array_equal(got, exp)


# ---------------------------------------------------------------------------
# Kernel parity vs the frozen oracle
# ---------------------------------------------------------------------------


class TestKernelParity:
    @pytest.mark.parametrize("stride", [1, 5, 20])
    @pytest.mark.parametrize("source", ["high", "close"])
    def test_bit_exact_parity_with_pathologies(self, stride, source):
        M, phi = 20, 0.0025
        close, high, low = make_prices(4_000, seed=stride * 7 + len(source))
        upper = high if source == "high" else close
        K = -(-len(close) // stride)

        y_o, m_o, t_o, mdn_o, tdn_o = legacy_loop_labels(
            close, upper, K, M, phi, stride, low
        )
        out = barrier_label_arrays(
            close, upper, horizon=M, phi=phi, stride=stride, n_out=K, low=low
        )
        assert_bit_equal(out.y, y_o, "y")
        assert_bit_equal(out.m_k, m_o, "m_k")
        assert_bit_equal(out.tau_k, t_o, "tau_k")
        assert_bit_equal(out.m_dn, mdn_o, "m_dn")
        assert_bit_equal(out.tau_dn, tdn_o, "tau_dn")

    def test_parity_without_downside(self):
        close, high, _ = make_prices(2_000, seed=3)
        y_o, m_o, t_o, _, _ = legacy_loop_labels(close, high, 2_000, 20, 0.0025, 1)
        out = barrier_label_arrays(close, high, horizon=20, phi=0.0025, stride=1)
        assert out.m_dn is None and out.tau_dn is None
        assert_bit_equal(out.y, y_o, "y")
        assert_bit_equal(out.m_k, m_o, "m_k")
        assert_bit_equal(out.tau_k, t_o, "tau_k")

    def test_parity_across_chunk_boundaries(self, monkeypatch):
        # Force multiple chunks to prove chunking is invisible in results.
        import src.labels.barrier as mod

        close, high, low = make_prices(1_500, seed=11)
        full = barrier_label_arrays(
            close, high, horizon=20, phi=0.0025, stride=1, low=low
        )
        monkeypatch.setattr(mod, "_CHUNK_CELLS", 512)  # ~25 rows per chunk
        chunked = barrier_label_arrays(
            close, high, horizon=20, phi=0.0025, stride=1, low=low
        )
        for name in ("y", "m_k", "tau_k", "m_dn", "tau_dn"):
            assert_bit_equal(getattr(chunked, name), getattr(full, name), name)

    @pytest.mark.parametrize("n", [0, 5, 20, 21, 22])
    def test_short_frames(self, n):
        M, phi = 20, 0.0025
        close, high, low = make_prices(max(n, 1), seed=n)[0:3]
        close, high, low = close[:n], high[:n], low[:n]
        K = -(-n // 1) if n else 0
        y_o, m_o, t_o, mdn_o, tdn_o = legacy_loop_labels(
            close, high, K, M, phi, 1, low
        )
        out = barrier_label_arrays(
            close, high, horizon=M, phi=phi, stride=1, low=low
        )
        assert_bit_equal(out.y, y_o, "y")
        assert_bit_equal(out.m_k, m_o, "m_k")
        assert_bit_equal(out.tau_k, t_o, "tau_k")

    def test_n_out_overrides_row_count_like_legacy(self):
        close, high, _ = make_prices(200, seed=1, pathologies=False)
        # K beyond the data: extra rows unlabeled (NaN) — same as legacy.
        out = barrier_label_arrays(
            close, high, horizon=20, phi=0.0025, stride=1, n_out=250
        )
        assert len(out.y) == 250 and np.isnan(out.y[200:]).all()
        # K below: truncated output.
        out2 = barrier_label_arrays(
            close, high, horizon=20, phi=0.0025, stride=1, n_out=100
        )
        assert len(out2.y) == 100

    def test_first_crossing_tie_semantics(self):
        # Barrier crossed at bars 3 and 7 — tau must be the FIRST (3).
        close = np.full(30, 100.0)
        high = np.full(30, 100.0)
        phi = 0.01
        high[3] = 100.0 * np.exp(0.02)
        high[7] = 100.0 * np.exp(0.05)
        out = barrier_label_arrays(close, high, horizon=10, phi=phi, stride=1)
        assert out.y[0] == 1.0 and out.tau_k[0] == 3.0

    def test_misaligned_inputs_rejected(self):
        with pytest.raises(ContractError, match="aligned"):
            barrier_label_arrays(
                np.ones(10), np.ones(9), horizon=3, phi=0.01
            )
        with pytest.raises(ContractError, match="low"):
            barrier_label_arrays(
                np.ones(10), np.ones(10), horizon=3, phi=0.01, low=np.ones(4)
            )

    def test_bad_params_rejected(self):
        with pytest.raises(ConfigError):
            barrier_label_arrays(np.ones(10), np.ones(10), horizon=0, phi=0.01)
        with pytest.raises(ConfigError):
            barrier_label_arrays(np.ones(10), np.ones(10), horizon=3, phi=np.nan)
        with pytest.raises(ConfigError):
            barrier_label_arrays(
                np.ones(10), np.ones(10), horizon=3, phi=0.01, stride=-1
            )


# ---------------------------------------------------------------------------
# Causality property
# ---------------------------------------------------------------------------


class TestCausality:
    def test_perturbing_beyond_horizon_never_changes_a_label(self):
        close, high, _ = make_prices(500, seed=5, pathologies=False)
        M, phi, k = 20, 0.0025, 100
        base = barrier_label_arrays(close, high, horizon=M, phi=phi, stride=1)
        # Mutate everything strictly after the window [k+1, k+M].
        high2 = high.copy()
        close2 = close.copy()
        high2[k + M + 1 :] *= 3.0
        close2[k + M + 1 :] *= 3.0
        out = barrier_label_arrays(close2, high2, horizon=M, phi=phi, stride=1)
        assert out.y[k] == base.y[k]
        assert np.array_equal(
            out.m_k[: k + 1], base.m_k[: k + 1], equal_nan=True
        )

    def test_perturbing_inside_horizon_changes_the_label(self):
        close = np.full(100, 100.0)
        high = np.full(100, 100.0)
        M, phi, k = 20, 0.01, 10
        base = barrier_label_arrays(close, high, horizon=M, phi=phi, stride=1)
        assert base.y[k] == 0.0
        high2 = high.copy()
        high2[k + 5] = 100.0 * np.exp(0.02)  # inside (k, k+M]
        out = barrier_label_arrays(close, high2, horizon=M, phi=phi, stride=1)
        assert out.y[k] == 1.0 and out.tau_k[k] == 5.0


# ---------------------------------------------------------------------------
# label_frame + BarrierLabeler block
# ---------------------------------------------------------------------------


def _bar_frame(n: int = 300, seed: int = 2) -> pl.DataFrame:
    close, high, low = make_prices(n, seed=seed, pathologies=False)
    ts = pd.date_range("2025-01-01", periods=n, freq="1min").to_numpy()
    return pl.DataFrame(
        {"ts": ts, "close": close, "high": high, "low": low}
    )


class TestLabelFrame:
    def test_columns_dtypes_and_null_semantics(self):
        spec = BarrierSpec(horizon=20, upper=0.0025, source="high")
        out = label_frame(_bar_frame(), spec)
        assert out.columns == ["y", "m_k", "tau_k", "phi"]
        assert out.schema["y"] == pl.Float64
        # Tail rows (open horizon) must be null, not NaN.
        assert out["y"][-1] is None
        assert int(out["y"].is_nan().sum() or 0) == 0
        LABEL_SCHEMA.validate(out, level="data")

    def test_downside_columns_when_requested(self):
        spec = BarrierSpec(horizon=20, upper=0.0025, downside=True)
        out = label_frame(_bar_frame(), spec)
        assert {"m_dn", "tau_dn"} <= set(out.columns)

    def test_missing_source_column_is_contract_error(self):
        spec = BarrierSpec(horizon=20, upper=0.0025, source="high")
        with pytest.raises(ContractError, match="high"):
            label_frame(_bar_frame().drop("high"), spec)


class TestBarrierLabelerBlock:
    def test_stride_must_be_one(self):
        with pytest.raises(ConfigError, match="stride"):
            BarrierLabeler(BarrierSpec(horizon=20, upper=0.0025, stride=20))

    def test_requires_derived_from_spec(self):
        b_high = BarrierLabeler(BarrierSpec(horizon=20, upper=0.0025, source="high"))
        assert set(b_high.requires) == {"close", "high"}
        b_close = BarrierLabeler(BarrierSpec(horizon=20, upper=0.0025, source="close"))
        assert set(b_close.requires) == {"close"}
        b_dn = BarrierLabeler(
            BarrierSpec(horizon=20, upper=0.0025, downside=True)
        )
        assert set(b_dn.requires) == {"close", "high", "low"}
        assert set(b_dn.provides) == {"y", "m_k", "tau_k", "phi", "m_dn", "tau_dn"}

    def test_apply_attaches_labels_row_aligned_and_does_not_mutate(self):
        frame = _bar_frame()
        block = BarrierLabeler(BarrierSpec(horizon=20, upper=0.0025))
        with assert_unmutated(frame, context="BarrierLabeler.apply"):
            out = block.apply(frame)
        assert out.height == frame.height
        assert {"y", "m_k", "tau_k", "phi"} <= set(out.columns)
        # Cross-check one row against the kernel on raw arrays.
        close = frame["close"].to_numpy()
        high = frame["high"].to_numpy()
        ref = barrier_label_arrays(close, high, horizon=20, phi=0.0025, stride=1)
        got = out["y"].fill_null(np.nan).to_numpy()
        assert_bit_equal(got, ref.y, "block y")


# ---------------------------------------------------------------------------
# Legacy adapter: construct_labels_pl alignment guard
# ---------------------------------------------------------------------------


class TestConstructLabelsAlignmentGuard:
    def _frames(self, n: int = 200):
        raw = _bar_frame(n)
        boundaries = raw.with_columns(pl.int_range(pl.len()).alias("k"))
        return boundaries, raw

    def test_aligned_frames_pass(self):
        from src.features.boundary import construct_labels_pl

        boundaries, raw = self._frames()
        out = construct_labels_pl(
            boundaries, raw, 20, 0.0002, 0.0023, bar_stride=1, barrier_source="high"
        )
        assert {"y", "m_k", "tau_k", "phi"} <= set(out.columns)

    def test_misaligned_frames_raise_contract_error(self):
        from src.features.boundary import construct_labels_pl

        boundaries, raw = self._frames()
        # Slice raw but not boundaries — the classic silent-corruption bug.
        with pytest.raises(ContractError, match="not aligned"):
            construct_labels_pl(
                boundaries,
                raw.tail(150),
                20,
                0.0002,
                0.0023,
                bar_stride=1,
                barrier_source="high",
            )

    def test_frames_without_ts_skip_the_guard(self):
        from src.features.boundary import construct_labels_pl

        boundaries, raw = self._frames()
        construct_labels_pl(
            boundaries.drop("ts"),
            raw.tail(150).drop("ts"),
            20,
            0.0002,
            0.0023,
            bar_stride=1,
            barrier_source="high",
        )  # no raise: positional semantics preserved for schemaless frames
