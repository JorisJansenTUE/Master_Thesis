"""Cumulative opportunity accessibility from shortest job costs."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CumulativeAccessibility:
    """Compute employment accessible at one or more travel-cost thresholds.

    The job counts are indexed by unique STATENT cell IDs, not connectors.
    An unreachable or unsnapped origin is NaN, not 0 accessible jobs.
    """

    thresholds_min: tuple[float, ...] = (20.0,)

    def __post_init__(self) -> None:
        values = tuple(sorted(set(float(t) for t in self.thresholds_min)))
        if not values or any(not isfinite(t) or t <= 0 for t in values):
            raise ValueError("Accessibility thresholds must be positive and finite")
        self.thresholds_min = values

    @property
    def max_cost_s(self) -> float:
        return max(self.thresholds_min) * 60.0

    @staticmethod
    def column_name(threshold_min: float) -> str:
        value = f"{threshold_min:g}".replace(".", "p")
        return f"jobs_{value}min"

    def evaluate(self, costs: np.ndarray | None, jobs: np.ndarray,
                 *, extra_time_s: float = 0.0) -> dict[str, float]:
        if costs is None:
            return {self.column_name(t): float("nan") for t in self.thresholds_min}
        if not isfinite(extra_time_s) or extra_time_s < 0:
            raise ValueError("Extra journey time must be finite and nonnegative")
        if len(costs) != len(jobs):
            raise ValueError("Costs/jobs length mismatch")
        return {
            self.column_name(t): float(jobs[costs + extra_time_s <= t * 60.0].sum())
            for t in self.thresholds_min
        }

    def run(self, router, origin_ids: Iterable[str], *, mode: str,
            extra_time_s: float = 0.0) -> pd.DataFrame:
        rows: list[dict] = []
        for hex_id in origin_ids:
            hex_id = str(hex_id)
            costs = router.shortest_job_costs(hex_id, self.max_cost_s)
            row = {"hex_id": hex_id, "mode": mode, "attached": costs is not None}
            row.update(self.evaluate(costs, router.job_counts, extra_time_s=extra_time_s))
            rows.append(row)
        return pd.DataFrame(rows)
