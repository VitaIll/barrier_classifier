"""Feature engine: run registered Feature classes against a bar dataframe.

Tiered execution: features declare a ``tier`` (int or str) on which they
depend. The engine groups specs by tier and runs each tier in its own
``with_columns`` call so later-tier expressions can reference columns
emitted by earlier tiers — polars cannot resolve forward references
within a single ``with_columns`` call.

Steps:
1. Coerce float NaN -> polars null (parity with pandas missing-value semantics).
2. Build (spec, window, column_name) plan from the registry.
3. Group plan by tier, ordered: ints ascending first, then strings.
4. For each tier, evaluate its expressions in one with_columns().
5. Trim warmup (leading nulls) and tail (trailing nulls) per declared
   per-feature contracts (max across all specs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import polars as pl

from src.features.base import Feature, get_registry
from src.features.observability import FeatureReport


# Order for string tiers — declared once, used by the planner. Ints sort
# naturally; strings follow this list (anything not listed sorts last).
_STRING_TIER_ORDER = ("boundary", "post_label", "barrier", "block")


@dataclass(frozen=True)
class EngineResult:
    data: pl.DataFrame
    warmup_trimmed: int
    tail_trimmed: int
    per_feature_report: dict[str, FeatureReport] = field(default_factory=dict)


class FeatureEngine:
    def __init__(
        self,
        *,
        tiers: Iterable[int | str] = (1,),
        families: Iterable[str] | None = None,
    ) -> None:
        self.tiers = tuple(tiers)
        self.families = tuple(families) if families is not None else None
        self.specs: list[Feature] = [
            cls() for cls in get_registry(tiers=self.tiers, families=self.families)
        ]
        # Dedupe check: two specs in the same tier emitting the same column
        # name would have one silently overwrite the other inside the
        # tier's ``with_columns`` call. Fail loudly at engine construction
        # instead of producing a wrong column.
        seen: dict[tuple[int | str, str], type[Feature]] = {}
        for spec in self.specs:
            for _w, name in spec.expanded():
                key = (spec.tier, name)
                prev = seen.get(key)
                if prev is not None and prev is not type(spec):
                    raise ValueError(
                        f"FeatureEngine column-name collision in tier "
                        f"{spec.tier!r}: column {name!r} is emitted by both "
                        f"{prev.__name__} and {type(spec).__name__}"
                    )
                seen[key] = type(spec)

    def plan(self) -> list[tuple[Feature, int | None, str]]:
        return [(spec, w, name) for spec in self.specs for w, name in spec.expanded()]

    def transform(
        self,
        bars: pl.DataFrame,
        *,
        validate: bool = False,
        oracle: pl.DataFrame | None = None,
        trim: bool = True,
    ) -> EngineResult:
        """Run all expressions tier-by-tier and return the wide frame.

        ``trim=True`` (default) slices off the leading warmup and trailing
        tail rows declared by the registered features. Set ``trim=False``
        when the caller needs the full row range — e.g. an end-to-end
        pipeline that samples decision boundaries on the un-trimmed
        frame so boundary indices match the original timestamp index.
        """
        bars = bars.with_columns(pl.col(pl.Float64).fill_nan(None))

        plan = self.plan()
        if not plan:
            return EngineResult(data=bars, warmup_trimmed=0, tail_trimmed=0)

        plans_by_tier: dict[int | str, list[tuple[Feature, int | None, str]]] = {}
        for triple in plan:
            plans_by_tier.setdefault(triple[0].tier, []).append(triple)

        out = bars
        for tier in _ordered_tiers(plans_by_tier.keys()):
            tier_plan = plans_by_tier[tier]
            exprs = [spec.compute(w).alias(name) for spec, w, name in tier_plan]
            out = out.with_columns(exprs)

        warmup, tail = self._declared_null_pattern(plan)
        if trim:
            n = len(out)
            trimmed = out.slice(warmup, n - warmup - tail) if (warmup or tail) else out
            warmup_actual = warmup
            tail_actual = tail
        else:
            trimmed = out
            warmup_actual = 0
            tail_actual = 0

        return EngineResult(
            data=trimmed,
            warmup_trimmed=warmup_actual,
            tail_trimmed=tail_actual,
            per_feature_report={},
        )

    @staticmethod
    def _declared_null_pattern(
        plan: list[tuple[Feature, int | None, str]],
    ) -> tuple[int, int]:
        warmup = max((spec.warmup_for(w) for spec, w, _ in plan), default=0)
        tail = max((spec.null_tail_bars for spec, _, _ in plan), default=0)
        return warmup, tail


def _ordered_tiers(tiers: Iterable[int | str]) -> list[int | str]:
    """Sort tiers so that dependencies resolve correctly.

    Integer tiers come first (ascending). String tiers follow in
    ``_STRING_TIER_ORDER``; unrecognized strings go last.
    """
    int_tiers = sorted(t for t in tiers if isinstance(t, int))

    def _str_key(t: str) -> int:
        try:
            return _STRING_TIER_ORDER.index(t)
        except ValueError:
            return len(_STRING_TIER_ORDER)

    str_tiers = sorted(
        (t for t in tiers if isinstance(t, str)),
        key=_str_key,
    )
    return int_tiers + str_tiers
