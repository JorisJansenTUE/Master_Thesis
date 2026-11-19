"""Bicycle perceived-travel-time rules matching the supplied eQasim Java code.

The network loader evaluates this once per bicycle link; shortest-path search
then reads the precomputed ``routing_cost_s`` edge attribute.

The project's BikeDisutilities.getLinkType(link) reads the MATSim custom
attribute ``osm:way:highway``, falling back to "unknown" when absent.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

REFERENCE_VTT = 1.0
INDEPENDENT_PATH_VTT = 1.0
CYCLESTREET_VTT = 1.272
MIXED_30_VTT = 1.434
MIXED_50_VTT = 1.629
DESTINATION_VTT = 1.188

THIRTY_KMH_M_S = 30.0 / 3.6


def bike_vtt(
    attributes: Mapping[str, str], freespeed_m_s: float
) -> tuple[float, str, str]:
    """Return (VTT multiplier, category, detected link type).

    Priority exactly matches the supplied Java code:
    independent path > bicycle street > destination road > mixed-speed road.
    The network *car* freespeed is used only for the 30 km/h category split.
    """
    if math.isnan(freespeed_m_s) or freespeed_m_s <= 0:
        raise ValueError(f"Invalid MATSim link freespeed: {freespeed_m_s!r}")

    highway = attributes.get("osm:way:highway", "unknown")

    if highway in ("cycleway", "bike_suited"):
        return INDEPENDENT_PATH_VTT / REFERENCE_VTT, "independent_path", highway
    if attributes.get("cyclestreet", "").strip().lower() == "true":
        return CYCLESTREET_VTT / REFERENCE_VTT, "cyclestreet", highway
    if attributes.get("destination_road", "").strip().lower() == "true":
        return DESTINATION_VTT / REFERENCE_VTT, "destination_road", highway
    if freespeed_m_s <= THIRTY_KMH_M_S:
        return MIXED_30_VTT / REFERENCE_VTT, "mixed_30", highway
    return MIXED_50_VTT / REFERENCE_VTT, "mixed_50", highway
