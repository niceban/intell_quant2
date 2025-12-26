from __future__ import annotations

from typing import Any, Dict

from ..config import STATUS_FINAL


class StatefulIndicator:
    """Base class for stateful, status-aware rolling indicators."""

    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.last_value: Any = None

    def update(self, **kwargs):  # pragma: no cover - interface only
        raise NotImplementedError

    @staticmethod
    def should_commit(status: int) -> bool:
        return status == STATUS_FINAL


def stream_apply(df, indicator: StatefulIndicator, value_fn):
    """Apply a stateful indicator row-by-row.

    `value_fn(row)` should map a row to the numeric input (e.g., close price).
    """
    values = []
    for _, row in df.iterrows():
        val = indicator.update(price=value_fn(row), row=row)
        values.append(val)
    out = df.copy()
    out[indicator.__class__.__name__] = values
    return out

