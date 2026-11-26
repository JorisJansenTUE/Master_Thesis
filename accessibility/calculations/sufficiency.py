"""Baseline-referenced sufficient accessibility, with optional STATPOP weighting.

Low/high thresholds: 50% / 100% of mean baseline peak-car accessibility,
following van der Veen et al. (2020, with Martens). In population mode the
reference is the *resident-weighted* baseline car mean across all populated
origins (including zero accessible jobs at unattached origins), as in Martens,
Singer & Cohen-Zada (2022). Per-mode population sufficiency and squared
shortfall are weighted by the same residents. Unattached populated origins
are insufficient, not dropped from population denominators.

This is an accessibility-only implementation of the AFI shortfall. It does NOT
implement the original paper's separate Potential Mobility Index (PMI) gate
or group-specific income/mode-availability population estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from .cumulative import CumulativeAccessibility
from .population_metrics import resident_weighted_mean

WEIGHTING_UNWEIGHTED = "unweighted_hexagons"
WEIGHTING_POPULATION = "statpop_residents"


@dataclass(frozen=True)
class SufficiencyReference:
    mean_car_jobs: dict[str, float]
    low_fraction: float = 0.5
    high_fraction: float = 1.0
    car_parking_penalty_s: float = 300.0
    weighting: str = WEIGHTING_UNWEIGHTED
    population_sha256: str | None = None

    def __post_init__(self) -> None:
        if not 0 < self.low_fraction < self.high_fraction:
            raise ValueError("Require 0 < low_fraction < high_fraction")
        if not isfinite(self.car_parking_penalty_s) or self.car_parking_penalty_s < 0:
            raise ValueError("Car parking penalty must be finite and nonnegative")
        if not self.mean_car_jobs:
            raise ValueError("No baseline car reference levels")
        for column, value in self.mean_car_jobs.items():
            if not str(column).startswith("jobs_") or not isfinite(value) or value <= 0:
                raise ValueError(f"Invalid baseline car mean for {column!r}: {value}")
        if self.weighting not in (WEIGHTING_UNWEIGHTED, WEIGHTING_POPULATION):
            raise ValueError(f"Unknown sufficiency weighting: {self.weighting}")
        if self.weighting == WEIGHTING_POPULATION and not self.population_sha256:
            raise ValueError("Population reference must record STATPOP file sha256")

    @property
    def population_weighted(self) -> bool:
        return self.weighting == WEIGHTING_POPULATION

    @classmethod
    def from_baseline_car(cls, car_results: pd.DataFrame,
                          thresholds_min: tuple[float, ...], *,
                          car_parking_penalty_s: float = 300.0,
                          population_weighted: bool = False,
                          population_sha256: str | None = None) -> "SufficiencyReference":
        """Compute baseline car benchmark for the selected interpretation.

        Legacy (unweighted): average of baseline car-attached hexagons.
        Population-weighted (Martens et al., 2022): mean across ALL residents,
        including zero car accessibility for detached origins. Never weight
        opportunity counts by the number of people at destination or origin.
        """
        if "attached" not in car_results:
            raise ValueError("Cumulative car results need an 'attached' column")
        attached = car_results["attached"].astype(bool)
        if not attached.any():
            raise ValueError("No baseline car-attached origins")
        if population_weighted and "population" not in car_results:
            raise ValueError("Missing STATPOP population column for weighted baseline reference")
        weights = (pd.to_numeric(car_results["population"], errors="raise")
                   if population_weighted else None)
        if population_weighted and (weights.isna().any() or not np.isfinite(weights.to_numpy()).all()
                                    or (weights < 0).any()):
            raise ValueError("Invalid baseline resident population")
        means = {}
        for threshold in thresholds_min:
            col = CumulativeAccessibility.column_name(threshold)
            if col not in car_results:
                raise ValueError(f"Missing baseline car column: {col}")
            raw_values = pd.to_numeric(car_results[col], errors="raise")
            if (raw_values[attached].isna().any() or
                    not np.isfinite(raw_values[attached].to_numpy()).all() or
                    (raw_values[attached] < 0).any()):
                raise ValueError(f"Invalid baseline car accessibility in {col}")
            if population_weighted:
                # Residents outside the baseline car network have zero accessible jobs.
                values = raw_values.where(attached, 0.0).fillna(0.0)
                means[col] = resident_weighted_mean(values, weights)
            else:
                means[col] = float(raw_values[attached].mean())
        return cls(means, car_parking_penalty_s=car_parking_penalty_s,
                   weighting=WEIGHTING_POPULATION if population_weighted else WEIGHTING_UNWEIGHTED,
                   population_sha256=population_sha256 if population_weighted else None)

    def threshold_jobs(self, column: str, level: str) -> float:
        if column not in self.mean_car_jobs:
            raise KeyError(f"No baseline reference for {column!r}")
        if level not in ("low", "high"):
            raise ValueError("level must be 'low' or 'high'")
        return (self.low_fraction if level == "low" else self.high_fraction) * self.mean_car_jobs[column]

    def save(self, filename: str | Path) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 3 if self.population_weighted else 2,
            "reference": ("baseline_population_mean_car_accessibility" if self.population_weighted
                          else "baseline_car_attached_origin_mean"),
            "weighting": self.weighting,
            "mean_car_jobs": self.mean_car_jobs,
            "car_parking_penalty_s": self.car_parking_penalty_s,
            "low_fraction": self.low_fraction,
            "high_fraction": self.high_fraction,
        }
        if self.population_weighted:
            payload["population_sha256"] = self.population_sha256
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, filename: str | Path) -> "SufficiencyReference":
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        weighting = payload.get("weighting")
        version = payload.get("version")
        expected_refs = {
            (2, WEIGHTING_UNWEIGHTED): "baseline_car_attached_origin_mean",
            (3, WEIGHTING_POPULATION): "baseline_population_mean_car_accessibility",
        }
        if (version, weighting) not in expected_refs or payload.get("reference") != expected_refs[(version, weighting)]:
            raise ValueError("Incompatible sufficiency reference version/weighting/benchmark")
        return cls({str(k): float(v) for k, v in payload["mean_car_jobs"].items()},
                   float(payload["low_fraction"]), float(payload["high_fraction"]),
                   float(payload["car_parking_penalty_s"]), weighting,
                   payload.get("population_sha256"))


def assess_sufficiency(accessibility: pd.DataFrame,
                       reference: SufficiencyReference,
                       thresholds_min: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-hex indicators and summary, retaining legacy area statistics.

    If the population column is present, also report person-weighted means,
    shares, and additive local contributions using ALL residents as denominator.
    Population weighting is not the same as weighting destination jobs.
    """
    required = {"hex_id", "mode", "attached"}
    if missing := required - set(accessibility.columns):
        raise ValueError(f"Missing accessibility columns: {sorted(missing)}")
    data = accessibility.copy()
    if data.duplicated(["hex_id", "mode"]).any():
        raise ValueError("Duplicate (hex_id, mode) in accessibility results")
    attached = data["attached"].astype(bool).to_numpy()
    has_pop = "population" in data
    if reference.population_weighted and not has_pop:
        raise ValueError("Population reference requires population per hexagon")
    if has_pop:
        population = pd.to_numeric(data["population"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(population).all() or (population < 0).any():
            raise ValueError("Population must be finite and nonnegative")
        for mode, subset in data.groupby("mode"):
            if not (subset.population.sum() > 0):
                raise ValueError(f"No residents for {mode}")
    else:
        population = None

    for cutoff in thresholds_min:
        col = CumulativeAccessibility.column_name(cutoff)
        if col not in data:
            raise ValueError(f"Missing accessibility column: {col}")
        values = pd.to_numeric(data[col], errors="raise").to_numpy(dtype=float)
        if np.any(~np.isfinite(values[attached])) or np.any(values[attached] < 0):
            raise ValueError(f"Invalid attached accessibility in {col}")
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
            if has_pop:
                # Mode-unattached origins are fully insufficient for residents.
                all_enough = np.where(attached, enough, 0.0)
                all_shortfall = np.where(attached, shortfall, 1.0)
                data[f"resident_sufficient_{level}_{postfix}"] = population * all_enough
                data[f"resident_shortfall_{level}_{postfix}"] = population * all_shortfall

    if has_pop:
        # AFI-like spatial contribution (accessibility-only, no PMI gate):
        # summing these across hexagons for a given mode gives the overall
        # resident-weighted shortfall for that mode in [0,1].
        for mode, mode_rows in data.groupby("mode"):
            total_people = float(mode_rows["population"].sum())
            for cutoff in thresholds_min:
                postfix = CumulativeAccessibility.column_name(cutoff).removeprefix("jobs_")
                for level in ("low", "high"):
                    source = f"resident_shortfall_{level}_{postfix}"
                    data.loc[mode_rows.index, f"afi_contribution_{level}_{postfix}"] = (
                        mode_rows[source].to_numpy(dtype=float) / total_people
                    )

    summary_rows = []
    for mode, subset in data.groupby("mode", sort=False):
        valid = subset["attached"].astype(bool)
        n = int(valid.sum())
        w = subset["population"].to_numpy(dtype=float) if has_pop else None
        w_total = float(w.sum()) if has_pop else 0.
        w_attached = float(w[valid.to_numpy()].sum()) if has_pop else 0.
        for cutoff in thresholds_min:
            col = CumulativeAccessibility.column_name(cutoff)
            postfix = col.removeprefix("jobs_")
            row = {
                "mode": str(mode), "time_min": float(cutoff),
                "n_origins": len(subset), "n_attached": n,
                "n_unattached": len(subset) - n,
                "mean_accessible_jobs": float(subset.loc[valid, col].mean()) if n else np.nan,
                "baseline_car_mean_jobs": reference.mean_car_jobs[col],
                "shortfall_formula": "squared_normalized_v2",
                "car_parking_penalty_s": reference.car_parking_penalty_s,
                "reference_weighting": reference.weighting,
            }
            if has_pop:
                all_values = subset[col].fillna(0.).to_numpy(dtype=float)
                row.update({
                    "population_total": w_total,
                    "population_attached": w_attached,
                    "population_unattached": w_total - w_attached,
                    "mean_accessible_jobs_population": resident_weighted_mean(all_values, w),
                    "population_attachment_pct": 100 * w_attached / w_total,
                })
            for level in ("low", "high"):
                enough = subset.loc[valid, f"sufficient_{level}_{postfix}"]
                shortfall = subset.loc[valid, f"shortfall_{level}_{postfix}"]
                row[f"threshold_{level}_jobs"] = reference.threshold_jobs(col, level)
                row[f"n_sufficient_{level}"] = int(enough.sum())
                row[f"pct_sufficient_{level}"] = 100. * enough.mean() if n else np.nan
                row[f"mean_shortfall_{level}"] = float(shortfall.mean()) if n else np.nan
                if has_pop:
                    pop_suff = float(subset[f"resident_sufficient_{level}_{postfix}"].sum())
                    pop_shortfall = float(subset[f"resident_shortfall_{level}_{postfix}"].sum())
                    row[f"population_sufficient_{level}"] = pop_suff
                    row[f"pct_population_sufficient_{level}"] = 100. * pop_suff / w_total
                    row[f"mean_shortfall_population_{level}"] = pop_shortfall / w_total
                    row[f"population_shortfall_sum_{level}"] = pop_shortfall
            summary_rows.append(row)
    return data, pd.DataFrame(summary_rows)
