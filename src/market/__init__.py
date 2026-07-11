"""Market-data domain: raw bar series that own their validation and repair.

The behaviors that lived as free ``utils`` functions and notebook cells
(kline validation, 1-min grid enforcement, flat-bar gap repair) attach to
the object that naturally owns them — the bar series itself.
"""

from src.market.bars import GapFillReport, RawBars, ValidationReport

__all__ = ["GapFillReport", "RawBars", "ValidationReport"]
