"""RawBars — the 1-minute OHLCV series as a domain object.

The bar series owns its integrity behaviors, the way a matrix owns its
rank: ``bars.validate()`` (the kline sanity report), ``bars.fill_gaps()``
(deterministic flat-bar grid repair, spec §4.5). Both were previously a
free ``utils`` function plus ~45 inline notebook lines with ZERO test
coverage despite being causality-relevant (a fabricated bar that looks
real can train a model — review finding N11).

``fill_gaps`` therefore reports exactly WHICH timestamps were fabricated,
so downstream training exclusions (the engine stores a ``synthetic`` flag
for the same reason) have a research-side source of truth.

All methods are pure: they never mutate the wrapped frame and return new
objects/reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.core.errors import ContractError
from src.core.log import get_logger

_log = get_logger("market")

OHLC_COLS: tuple[str, ...] = ("open", "high", "low", "close")
VOLUME_COLS: tuple[str, ...] = (
    "volume", "quote_volume", "taker_buy_base", "taker_buy_quote",
)


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of :meth:`RawBars.validate` — issues, not exceptions."""

    n_rows: int
    date_range: tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]
    issues: tuple[str, ...]
    gap_locations: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class GapFillReport:
    """What :meth:`RawBars.fill_gaps` fabricated."""

    n_filled: int
    synthetic_ts: pd.DatetimeIndex

    @property
    def any_filled(self) -> bool:
        return self.n_filled > 0


@dataclass(frozen=True)
class RawBars:
    """A tz-aware (UTC) 1-minute OHLCV frame with owner-attached integrity.

    Construction validates the STRUCTURE (index tz, required columns);
    data-level problems are the job of :meth:`validate`, which reports
    rather than raises — a research dataset with three bad ticks is a
    dataset with three documented issues, not an exception.
    """

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        idx = self.frame.index
        if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
            raise ContractError(
                "RawBars requires a tz-aware DatetimeIndex (UTC); got "
                f"{type(idx).__name__} tz={getattr(idx, 'tz', None)!r}"
            )
        if str(idx.tz) != "UTC":
            raise ContractError(f"RawBars index must be UTC; got tz={idx.tz!r}")
        missing = [c for c in OHLC_COLS if c not in self.frame.columns]
        if missing:
            raise ContractError(f"RawBars frame is missing OHLC column(s) {missing}")

    # -- introspection ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index

    def expected_grid(self) -> pd.DatetimeIndex:
        """The complete 1-min grid spanning this series' first..last bar."""
        return pd.date_range(
            self.index.min(), self.index.max(), freq="1min", tz="UTC"
        )

    # -- validation (port of utils.validate_klines; report, don't raise) ------

    def validate(self) -> ValidationReport:
        """Kline sanity report: OHLC coherence, volumes, grid, monotonicity."""
        df = self.frame
        issues: list[str] = []
        gap_locations: dict = {}

        invalid_high = df["high"] < df[["open", "close"]].max(axis=1)
        invalid_low = df["low"] > df[["open", "close"]].min(axis=1)
        invalid_range = df["high"] < df["low"]
        non_positive = (df[list(OHLC_COLS)] <= 0).any(axis=1)
        if invalid_high.any():
            issues.append(f"High < max(O,C): {int(invalid_high.sum())} bars")
        if invalid_low.any():
            issues.append(f"Low > min(O,C): {int(invalid_low.sum())} bars")
        if invalid_range.any():
            issues.append(f"High < Low: {int(invalid_range.sum())} bars")
        if non_positive.any():
            issues.append(f"Non-positive OHLC: {int(non_positive.sum())} bars")

        if "volume" in df.columns:
            if (df["volume"] < 0).any():
                issues.append(f"Negative volume: {int((df['volume'] < 0).sum())} bars")
            if "taker_buy_base" in df.columns:
                bad = df["taker_buy_base"] > df["volume"]
                if bad.any():
                    issues.append(f"Taker > Volume: {int(bad.sum())} bars")

        dups = df.index.duplicated()
        if dups.any():
            issues.append(f"Duplicate timestamps: {int(dups.sum())}")

        time_diffs = df.index.to_series().diff().dropna()
        gaps = time_diffs[time_diffs != pd.Timedelta(minutes=1)]
        if len(gaps) > 0:
            issues.append(f"Gaps detected: {int(len(gaps))}")
            gap_locations = {
                k.isoformat(): str(v) for k, v in gaps.head(10).items()
            }

        if not df.index.is_monotonic_increasing:
            issues.append("Timestamps not monotonic increasing")

        return ValidationReport(
            n_rows=int(len(df)),
            date_range=(
                (df.index.min(), df.index.max()) if len(df) else (None, None)
            ),
            issues=tuple(issues),
            gap_locations=gap_locations,
        )

    # -- gap repair (port of the nb01 grid-enforcement cell; spec §4.5) -------

    def fill_gaps(
        self, expected_index: Optional[pd.DatetimeIndex] = None
    ) -> tuple["RawBars", GapFillReport]:
        """Deterministic flat-bar repair onto the complete 1-min grid.

        Missing bars become FLAT synthetic bars: OHLC = previous close
        (forward-filled), volumes/quote/taker = 0, num_trades = 0 — the
        same rule the live engine's grid guard applies, so research and
        serving fabricate identical bars. The report carries the
        fabricated timestamps; training-row exclusions key off them.

        Raises :class:`ContractError` when the FIRST expected bar is
        missing — there is no previous close to fabricate from, and
        inventing one would poison every window that touches it.
        """
        if expected_index is None:
            expected_index = self.expected_grid()
        df = self.frame.reindex(expected_index)
        missing = df["close"].isna()
        n_missing = int(missing.sum())
        if n_missing == 0:
            return RawBars(df), GapFillReport(
                n_filled=0, synthetic_ts=pd.DatetimeIndex([], tz="UTC")
            )
        close_ffill = df["close"].ffill()
        if close_ffill.isna().any():
            raise ContractError(
                "cannot fill missing bars: the grid starts before the first "
                "real bar (leading NaNs after ffill) — no previous close to "
                "fabricate from"
            )
        df = df.copy()
        for col in OHLC_COLS:
            df.loc[missing, col] = close_ffill[missing]
        for col in VOLUME_COLS:
            if col in df.columns:
                df.loc[missing, col] = 0.0
        if "num_trades" in df.columns:
            df.loc[missing, "num_trades"] = 0
        synthetic_ts = df.index[missing]
        _log.info(
            "RawBars.fill_gaps: fabricated %d flat bar(s) "
            "(%.4f%% of the grid); first at %s",
            n_missing,
            100.0 * n_missing / max(len(df), 1),
            synthetic_ts[0],
        )
        return RawBars(df), GapFillReport(
            n_filled=n_missing, synthetic_ts=synthetic_ts
        )
