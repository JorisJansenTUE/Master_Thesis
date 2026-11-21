"""Accessibility-only sufficiency indicators from cumulative reachable jobs.

The low/high reference follows van der Veen et al. (2020): 50% and 100% of
*baseline peak-period car accessibility*. Use the SAME car reference for every
mode and intervention scenario; do not recalculate from each scenario.

This module intentionally does NOT claim to implement the paper's full AFI:
that indicator also applies a potential-mobility threshold and population-
weighted aggregation, neither of which is available from our hexagon data.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from .cumulative import CumulativeAccessibility


@dataclass(frozen=True)
class SufficiencyReference:
    """Fixed, baseline car reference levels, one for each travel-time cutoff."""

    mean_car_jobs: dict[str, float]
    low_fraction: float = 0.5
    high_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.low_fraction < self.high_fraction:
            raise ValueError("Require 0 < low_fraction < high_fraction")
        if not self.mean_car_jobs:
            raise ValueError("No baseline car reference levels")
        for column, value in self.mean_car_jobs.items():
            if not str(column).startswith("jobs_") or not isfinite(value) or value <= 0:
                raise ValueError(f"Invalid baseline car mean for {column!r}: {value}")

    @classmethod
    def from_baseline_car(cls, car_results: pd.DataFrame,
                          thresholds_min: tuple[float, ...]) -> "SufficiencyReference":
        """Unweighted mean across baseline car-attached origin hexagons.

        There are no residential population weights in the current inputs.
        Unattached origins are excluded, not reinterpreted as zero jobs.
        """
        if "attached" not in car_results:
            raise ValueError("Cumulative car results need an 'attached' column")
        attached = car_results["attached"].astype(bool)
        if not attached.any():
            raise ValueError("No baseline car-attached origins to define the reference")
        means: dict[str, float] = {}
        for threshold in thresholds_min:
            col = CumulativeAccessibility.column_name(threshold)
            if col not in car_results:
                raise ValueError(f"Missing baseline car column: {col}")
            values = pd.to_numeric(car_results.loc[attached, col], errors="raise")
            if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                raise ValueError(f"Non-finite baseline car accessibility in {col}")
            means[col] = float(values.mean())
        return cls(means)

    def threshold_jobs(self, column: str, level: str) -> float:
        if column not in self.mean_car_jobs:
            raise KeyError(f"No baseline reference for {column!r}")
        if level not in ("low", "high"):
            raise ValueError("level must be 'low' or 'high'")
        fraction = self.low_fraction if level == "low" else self.high_fraction
        return fraction * self.mean_car_jobs[column]

    def save(self, filename: str | Path) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "reference": "baseline_car_attached_origin_mean",
            "weighting": "unweighted_hexagons",
            "mean_car_jobs": self.mean_car_jobs,
            "low_fraction": self.low_fraction,
            "high_fraction": self.high_fraction,
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, filename: str | Path) -> "SufficiencyReference":
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        if payload.get("reference") != "baseline_car_attached_origin_mean":
            raise ValueError("Unrecognised sufficiency reference file")
        if payload.get("weighting") != "unweighted_hexagons":
            raise ValueError("Sufficiency reference weighting differs from this implementation")
        return cls({str(k): float(v) for k, v in payload["mean_car_jobs"].items()},
                   float(payload["low_fraction"]), float(payload["high_fraction"]))


def assess_sufficiency(accessibility: pd.DataFrame,
                       reference: SufficiencyReference,
                       thresholds_min: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-hexagon sufficiency and squared deficiency; summarise by mode.

    Sufficient = accessible jobs >= reference threshold.
    Accessibility-only squared shortfall = max(0, 1 - A/T)**2, in [0,1].
    Unattached origins stay missing (NaN), never converted to zero.

    Reports are equally weighted over *attached hexagons*. This is not the
    population-weighted / PMI-gated Accessibility Fairness Index of the paper.
    """
    required = {"hex_id", "mode", "attached"}
    if missing := required.difference(accessibility.columns):
        raise ValueError(f"Missing accessibility columns: {sorted(missing)}")
    data = accessibility.copy()
    if data.duplicated(["hex_id", "mode"]).any():
        raise ValueError("Duplicate (hex_id, mode) in accessibility results")
    attached = data["attached"].astype(bool).to_numpy()
    modes = data["mode"].unique()

    for cutoff in thresholds_min:
        col = CumulativeAccessibility.column_name(cutoff)
        if col not in data:
            raise ValueError(f"Missing accessibility column: {col}")
        values = pd.to_numeric(data[col], errors="raise").to_numpy(dtype=float)
        if np.any(~np.isfinite(values[attached])) or np.any(values[attached] < 0):
            raise ValueError(f"Attached origins have invalid accessibility in {col}")
        if np.any(~np.isnan(values[~attached])):
            raise ValueError(f"Unattached origins must have NaN accessibility in {col}")
        postfix = col.removeprefix("jobs_")
        for level in ("low", "high"):
            threshold = reference.threshold_jobs(col, level)
            enough = np.full(len(data), np.nan)
            shortfall = np.full(len(data), np.nan)
            enough[attached] = (values[attached] >= threshold).astype(float)
            shortfall[attached] = np.maximum(0., 1. - values[attached] / threshold) ** 2
            data[f"sufficient_{level}_{postfix}"] = enough
            data[f"shortfall_{level}_{postfix}"] = shortfall

    summary_rows: list[dict] = []
    for mode in modes:
        subset = data.loc[data["mode"] == mode]
        valid = subset["attached"].astype(bool)
        n = int(valid.sum())
        for cutoff in thresholds_min:
            col = CumulativeAccessibility.column_name(cutoff)
            postfix = col.removeprefix("jobs_")
            row = {
                "mode": str(mode), "time_min": float(cutoff),
                "n_origins": len(subset), "n_attached": n,
                "n_unattached": len(subset)-n,
                "mean_accessible_jobs": float(subset.loc[valid, col].mean()) if n else np.nan,
                "baseline_car_mean_jobs": reference.mean_car_jobs[col],
            }
            for level in ("low", "high"):
                enough = subset.loc[valid, f"sufficient_{level}_{postfix}"]
                shortfall = subset.loc[valid, f"shortfall_{level}_{postfix}"]
                row[f"threshold_{level}_jobs"] = reference.threshold_jobs(col, level)
                row[f"n_sufficient_{level}"] = int(enough.sum())
                row[f"pct_sufficient_{level}"] = 100.0 * enough.mean() if n else np.nan
                row[f"mean_shortfall_{level}"] = float(shortfall.mean()) if n else np.nan
            summary_rows.append(row)
    return data, pd.DataFrame(summary_rows)
