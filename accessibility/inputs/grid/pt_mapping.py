"""Static PT access/egress mapping from hexagons and STATENT jobs to stops.

This is deliberately separate from the road-link snapping and the PT routing
network. It uses only scheduled stop coordinates (LV95) and route membership.

Each input point may connect to multiple *served* stop facilities. Walking is
allowed only as origin->stop or stop->destination; this module NEVER adds
stop-to-stop walking edges or changes the schedule-defined transfer graph.

The mapping is scenario-independent as long as the stop locations and route
service topology remain unchanged. Cache its CSVs once for the optimisation loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


from accessibility.inputs.links.schedule import TransitSchedule


CONNECTOR_COLUMNS = (
    "point_id", "stop_id", "distance_m", "walk_distance_m", "walk_time_s",
    "route_count",
)


@dataclass(slots=True)
class PTStopAttachments:
    connectors: pd.DataFrame
    total_points: int
    attached_points: int
    unattached_ids: tuple[str, ...]
    nearest_distances_m: dict[str, float]

    @property
    def unattached_count(self) -> int:
        return self.total_points - self.attached_points

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.connectors.to_csv(target, index=False)


class PTStopMapper:
    """Fast radius-based spatial index for served MATSim transit stop facilities.

    Uses a single SciPy KD-tree instead of scanning all stops per point.
    All stop facilities inside the radius are retained (not just the nearest),
    allowing later routing to choose the best PT route/boarding stop.

    ``max_distance_m`` is a straight-line search radius. Walking travel time
    is estimated as (distance * walking_detour_factor) / walking_speed_m_s.
    This is a deliberate first-order approximation, not pedestrian routing.
    """

    def __init__(
        self,
        schedule: TransitSchedule,
        *,
        walking_speed_m_s: float = 1.3, #same as Matsim default
        walking_detour_factor: float = 1.3,
    ) -> None:
        if not math.isfinite(walking_speed_m_s) or walking_speed_m_s <= 0:
            raise ValueError("walking_speed_m_s must be finite and positive")
        if not math.isfinite(walking_detour_factor) or walking_detour_factor < 1:
            raise ValueError("walking_detour_factor must be finite and >= 1")

        routes_by_stop: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for key, route in schedule.routes.items():
            for stop in route.stops:
                routes_by_stop[stop.stop_id].add(key)
        if not routes_by_stop:
            raise ValueError("Transit schedule has no served stop facilities")

        # Sorted stable order gives reproducible mappings and CSV outputs.
        self.stop_ids = tuple(sorted(routes_by_stop))
        missing = [sid for sid in self.stop_ids if sid not in schedule.stops]
        if missing:
            raise ValueError(f"Served transit stops missing from schedule: {missing[:5]}")
        coordinates = np.array(
            [(schedule.stops[sid].x, schedule.stops[sid].y) for sid in self.stop_ids],
            dtype=np.float64,
        )
        if not np.isfinite(coordinates).all():
            raise ValueError("Invalid transit stop coordinates")
        self.coordinates = coordinates
        self.route_counts = np.array(
            [len(routes_by_stop[sid]) for sid in self.stop_ids], dtype=np.int32,
        )
        self.walking_speed_m_s = float(walking_speed_m_s)
        self.walking_detour_factor = float(walking_detour_factor)
        self.tree = cKDTree(coordinates)
        self.schedule = schedule

    @property
    def num_stops(self) -> int:
        return len(self.stop_ids)

    def attach(
        self,
        points: pd.DataFrame,
        *,
        id_column: str,
        x_column: str,
        y_column: str,
        max_distance_m: float,
    ) -> PTStopAttachments:
        """Create walking connectors from each point to *every* nearby PT stop.

        The result is many-to-many (point -> stop). A job remains one unique
        job, regardless of how many stops can reach it; future aggregation
        must take the minimum arrival time for each job before counting jobs.
        """
        if not math.isfinite(max_distance_m) or max_distance_m <= 0:
            raise ValueError("max_distance_m must be finite and positive")
        required = {id_column, x_column, y_column}
        if missing := required.difference(points.columns):
            raise ValueError(f"Missing point columns: {sorted(missing)}")
        ids = points[id_column].astype(str).to_list()
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate point IDs in {id_column}")
        xy = points[[x_column, y_column]].to_numpy(dtype=np.float64)
        if not np.isfinite(xy).all():
            raise ValueError("Point coordinates must be finite LV95 values")
        if len(ids) == 0:
            return PTStopAttachments(
                pd.DataFrame(columns=CONNECTOR_COLUMNS), 0, 0, (), {},
            )

        nearest, _ = self.tree.query(xy, k=1)
        candidates = self.tree.query_ball_point(xy, r=max_distance_m)
        records: list[tuple[str, str, float, float, float, int]] = []
        unmatched: list[str] = []
        for idx, nearby in enumerate(candidates):
            if not nearby:
                unmatched.append(ids[idx])
                continue
            stop_indices = np.asarray(nearby, dtype=np.intp)
            delta = self.coordinates[stop_indices] - xy[idx]
            lengths = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            order = np.argsort(lengths, kind="stable")
            for pos in order:
                stop_idx = int(stop_indices[pos])
                crow_flight = float(lengths[pos])
                walk = crow_flight * self.walking_detour_factor
                records.append((
                    ids[idx], self.stop_ids[stop_idx], crow_flight, walk,
                    walk / self.walking_speed_m_s,
                    int(self.route_counts[stop_idx]),
                ))

        return PTStopAttachments(
            connectors=pd.DataFrame.from_records(records, columns=CONNECTOR_COLUMNS),
            total_points=len(ids),
            attached_points=len(ids) - len(unmatched),
            unattached_ids=tuple(unmatched),
            nearest_distances_m=dict(zip(ids, nearest.astype(float), strict=True)),
        )
