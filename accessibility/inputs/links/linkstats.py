"""Small, dependency-free reader for MATSim linkstats travel times.

The reader owns file loading; downstream network/routing code only needs
``travel_times_s``. It deliberately does not infer allowed transport modes,
repair invalid data, or average across hours/iterations.
"""

from __future__ import annotations

import csv
import gzip
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Statistic = Literal["min", "avg", "max"]


@dataclass(frozen=True, slots=True)
class InvalidTravelTime:
    raw_value: str
    reason: str


@dataclass(slots=True)
class LinkTravelTimes:
    """One time-period snapshot of MATSim linkstats.

    All mapping keys are original MATSim link IDs as strings.
    Only *positive, finite* travel times appear in ``travel_times_s``.
    Invalid values are retained in ``invalid_travel_times`` for inspection.

    Linkstats contains neither reliable car permissions nor mode-specific
    cycling/PT travel times. Mode filtering happens with the MATSim network
    in the next development step.
    """

    source_path: Path
    travel_time_column: str
    hour_start: int
    statistic: Statistic
    averaged_iterations: int | None
    row_count: int
    travel_times_s: dict[str, float]
    free_flow_times_s: dict[str, float]
    hourly_volumes: dict[str, float]
    invalid_travel_times: dict[str, InvalidTravelTime]

    @classmethod
    def from_linkstats(
        cls,
        path: str | Path,
        *,
        hour_start: int = 8,
        statistic: Statistic = "avg",
        averaged_iterations: int | None = None,
    ) -> LinkTravelTimes:
        """Load one hourly travel-time column from .txt or .txt.gz linkstats.

        Args:
            path: MATSim linkstats path (plain tab-separated or gzip).
            hour_start: Start hour, from 0 through 23 (e.g. 8 = 08:00-09:00).
            statistic: MATSim linkstats min/avg/max column suffix.
            averaged_iterations: Optional provenance metadata supplied by user.
                The linkstats file does not reliably identify this count.

        Raises:
            ValueError: Missing column, malformed link IDs or duplicate IDs.
            FileNotFoundError: The supplied path does not exist.
        """
        if not isinstance(hour_start, int) or isinstance(hour_start, bool) or not 0 <= hour_start <= 23:
            raise ValueError("hour_start must be an integer between 0 and 23")
        if statistic not in ("min", "avg", "max"):
            raise ValueError("statistic must be 'min', 'avg', or 'max'")
        if averaged_iterations is not None and averaged_iterations < 1:
            raise ValueError("averaged_iterations must be positive if provided")

        source = Path(path).expanduser()
        column = f"TRAVELTIME{hour_start}-{hour_start + 1}{statistic}"
        volume_column = f"HRS{hour_start}-{hour_start + 1}{statistic}"
        opener = gzip.open if source.suffix.lower() == ".gz" else open
        travel_times: dict[str, float] = {}
        free_flow_times: dict[str, float] = {}
        volumes: dict[str, float] = {}
        invalid: dict[str, InvalidTravelTime] = {}
        seen: set[str] = set()
        row_count = 0

        with opener(source, mode="rt", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            available = set(reader.fieldnames or [])
            required = {"LINK", "LENGTH", "FREESPEED", column}
            missing = required - available
            if missing:
                raise ValueError(
                    f"Missing linkstats columns: {', '.join(sorted(missing))}. "
                    f"Selected travel time column: {column}"
                )

            for row_count, row in enumerate(reader, start=1):
                link_id = (row.get("LINK") or "").strip()
                if not link_id:
                    raise ValueError(f"Missing LINK at data row {row_count}")
                if link_id in seen:
                    raise ValueError(f"Duplicate LINK '{link_id}' at data row {row_count}")
                seen.add(link_id)

                raw_tt = (row.get(column) or "").strip()
                try:
                    tt = float(raw_tt)
                except (ValueError, TypeError):
                    invalid[link_id] = InvalidTravelTime(raw_tt, "missing_or_nonnumeric")
                else:
                    if not math.isfinite(tt):
                        invalid[link_id] = InvalidTravelTime(raw_tt, "nonfinite")
                    elif tt <= 0:
                        invalid[link_id] = InvalidTravelTime(raw_tt, "nonpositive")
                    else:
                        travel_times[link_id] = tt

                # Free-flow time is an explicit routing input for *_spec links.
                # For ordinary links, it is diagnostic only and must not be
                # used to silently replace an invalid observed travel time.
                length = _finite_float(row.get("LENGTH"))
                speed = _finite_float(row.get("FREESPEED"))
                if length is not None and length > 0 and speed is not None and speed > 0:
                    free_flow_times[link_id] = length / speed

                if volume_column in available:
                    volume = _finite_float(row.get(volume_column))
                    if volume is not None and volume >= 0:
                        volumes[link_id] = volume

        return cls(
            source_path=source,
            travel_time_column=column,
            hour_start=hour_start,
            statistic=statistic,
            averaged_iterations=averaged_iterations,
            row_count=row_count,
            travel_times_s=travel_times,
            free_flow_times_s=free_flow_times,
            hourly_volumes=volumes,
            invalid_travel_times=invalid,
        )

    def travel_time_s(self, link_id: str) -> float | None:
        """Return a valid travel time, or None for missing/invalid link IDs."""
        return self.travel_times_s.get(str(link_id))

    @property
    def valid_count(self) -> int:
        return len(self.travel_times_s)


def _finite_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None
