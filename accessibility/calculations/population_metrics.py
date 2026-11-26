"""Shared resident population input and weighted accessibility summaries.

Population describes origins (STATPOP residents); opportunities remain STATENT jobs.
Never multiply job counts by resident population in the routing calculation.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


def load_population(spatial_dir: str | Path, *, filename: str = "population_by_hexagon.csv") -> pd.DataFrame:
    """Return exactly one nonnegative resident count for every origin hexagon.

    Compare against `origins.csv` to detect mismatched spatial versions.
    Population-less origins have weight zero, not one. Population must not be
    silently inferred from the number of hexagons or from STATENT jobs.
    """
    spatial = Path(spatial_dir)
    source = spatial / filename
    if not source.is_file():
        raise FileNotFoundError(f"STATPOP origin population is missing: {source}; "
                                "first run python -m accessibility.inputs.grid.population")
    p = pd.read_csv(source, dtype={"hex_id": str})
    if not {"hex_id", "population"}.issubset(p.columns):
        raise ValueError(f"{source}: expected hex_id,population")
    if p.hex_id.isna().any() or p.hex_id.duplicated().any():
        raise ValueError("Missing or duplicate STATPOP origin IDs")
    p["population"] = pd.to_numeric(p["population"], errors="raise")
    if p.population.isna().any() or not np.isfinite(p.population.to_numpy()).all() or (p.population < 0).any():
        raise ValueError("STATPOP population must be finite and >= 0")
    origin_file = spatial / "origins.csv"
    if not origin_file.is_file():
        raise FileNotFoundError(f"Missing spatial origin grid: {origin_file}")
    origins = pd.read_csv(origin_file, usecols=["hex_id"], dtype={"hex_id": str})
    if origins.hex_id.duplicated().any() or set(origins.hex_id) != set(p.hex_id):
        raise ValueError("STATPOP origin grid does not match origins.csv exactly")
    if not (p.population > 0).any():
        raise ValueError("No populated hexagons in the chosen origin grid")
    return p[["hex_id", "population"]].copy()


def attach_population(accessibility: pd.DataFrame, spatial_dir: str | Path) -> pd.DataFrame:
    p = load_population(spatial_dir)
    if "population" in accessibility:
        accessibility = accessibility.drop(columns="population")
    result = accessibility.merge(p, on="hex_id", how="left", validate="many_to_one")
    if result.population.isna().any():
        raise ValueError("Accessibility results contain origin IDs absent from STATPOP")
    return result


def weight_sha256(spatial_dir: str | Path) -> str:
    source = Path(spatial_dir) / "population_by_hexagon.csv"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def resident_weighted_mean(values: pd.Series | np.ndarray,
                           weights: pd.Series | np.ndarray) -> float:
    a = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if a.shape != w.shape or a.ndim != 1:
        raise ValueError("Population and values must be aligned 1-D arrays")
    if not np.isfinite(a).all() or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid values or zero-sum resident weights")
    return float(np.dot(a, w) / w.sum())
