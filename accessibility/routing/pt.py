"""PT route-state routing with walking at the two ends of a journey.

Uses the existing PTNetwork route-state DiGraph and the cached PT stop
connectors. Transfers are possible only through edges already in PTNetwork
(i.e. explicit schedule minimalTransferTimes relations). No new stop-to-stop
walks, departure waiting, or transfer rules are introduced here.

A PT journey must traverse >=1 `kind="ride"` edge. This prevents treating
walking from an origin to a stop and back to a job as a PT trip.
"""
from __future__ import annotations

from collections import defaultdict
import warnings
from math import inf, isfinite

import networkx as nx
import numpy as np
import pandas as pd

from ._shortest_paths import bounded_shortest_paths
from .road import _prepare_jobs

_REQUIRED = {"point_id", "stop_id", "walk_time_s"}


class PTRouter:
    """One bounded PT shortest-path search per origin, with walking ends.

    ``pt_network`` is the already-built PTNetwork (not the schedule itself).
    Static walking connectors are reusable while stop IDs and coordinates stay
    unchanged, even when stop linkRefIds change to nearby _spec links.
    """

    def __init__(
        self,
        pt_network,
        origin_connectors: pd.DataFrame,
        job_connectors: pd.DataFrame,
        jobs: pd.DataFrame,
    ) -> None:
        graph = pt_network.graph
        if not isinstance(graph, nx.DiGraph) or graph.is_multigraph():
            raise TypeError("PTRouter requires a PTNetwork with a directed state graph")
        self.job_ids, self.job_counts, index = _prepare_jobs(jobs)
        by_stop = pt_network.states_by_stop
        unserved: set[str] = set()

        def prepare(frame: pd.DataFrame, label: str) -> dict[str, tuple[tuple[str, float], ...]]:
            missing = _REQUIRED.difference(frame.columns)
            if missing:
                raise ValueError(f"{label} missing columns: {sorted(missing)}")
            by_point: dict[str, dict[str, float]] = defaultdict(dict)
            for point_id, stop_id, raw in frame[
                ["point_id", "stop_id", "walk_time_s"]
            ].itertuples(index=False, name=None):
                point_id, stop_id = str(point_id), str(stop_id)
                walking_s = float(raw)
                if not isfinite(walking_s) or walking_s < 0:
                    raise ValueError(f"{label}: invalid walking time for {point_id}")
                if stop_id not in by_stop:
                    # A stop can become unserved in a scenario. Never invent PT
                    # access by retaining walking links to a no-longer-served stop.
                    unserved.add(stop_id)
                    continue
                by_point[point_id][stop_id] = min(
                    walking_s, by_point[point_id].get(stop_id, inf)
                )
            return {pid: tuple(times.items()) for pid, times in by_point.items()}

        self.origins = prepare(origin_connectors, "Origin")
        destinations = prepare(job_connectors, "Job")
        self.unknown_job_ids = tuple(sorted(set(destinations).difference(index)))
        for jid in self.unknown_job_ids:
            del destinations[jid]
        self.unserved_stop_ids = tuple(sorted(unserved))
        if self.unknown_job_ids:
            warnings.warn(
                f"Skipping {len(self.unknown_job_ids)} PT job IDs absent "
                "from jobs.csv; check that PT mapping and STATENT use the "
                f"same job selection. Examples: {self.unknown_job_ids[:5]}",
                UserWarning, stacklevel=2,
            )
        if self.unserved_stop_ids:
            warnings.warn(
                f"Ignoring {len(self.unserved_stop_ids)} PT stops that are "
                "not served in the scenario schedule. Rerun PT mapping if "
                f"service coverage changed. Examples: {self.unserved_stop_ids[:5]}",
                UserWarning, stacklevel=2,
            )
        by_stop_jobs: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for jid, stop_costs in destinations.items():
            for stop_id, egress_s in stop_costs:
                by_stop_jobs[stop_id].append((index[jid], egress_s))
        self._jobs_at_stop = {
            sid: tuple(sorted(costs, key=lambda item: item[1]))
            for sid, costs in by_stop_jobs.items()
        }
        self._states_by_stop = by_stop
        self._stop_by_state = {state: str(data["stop_id"]) for state, data in graph.nodes(data=True)}

        # Two layers: has ridden PT? false -> true only via a 'ride' edge.
        # Transfer edges are copied within the same layer, so transfers
        # cannot invent a PT journey without an actual transit ride.
        adjacency = defaultdict(list)
        for u, v, data in graph.edges(data=True):
            kind = data.get("kind")
            if kind not in ("ride", "transfer"):
                raise ValueError(f"Unsupported PT edge kind {kind!r}: {u}->{v}")
            raw = data.get("routing_cost_s")
            cost = float(raw) if raw is not None else float("nan")
            if not isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid PT routing cost on {u}->{v}: {raw!r}")
            adjacency[(u, True)].append(((v, True), cost))
            adjacency[(u, False)].append(((v, kind == "ride"), cost))
        self._adjacency = {k: tuple(v) for k, v in adjacency.items()}
        self.attached_origin_count = len(self.origins)
        self.attached_job_count = len(destinations)
        self.attached_jobs_total = float(sum(self.job_counts[index[j]] for j in destinations))

    def shortest_job_costs(self, origin_id: str, max_cost_s: float) -> np.ndarray | None:
        """Return fastest PT+walk cost per job; inf beyond cutoff, None if unattached.

        Walking access/egress is included. No frequency or boarding wait;
        at least one PT ride is required. Time is in seconds.
        """
        if not isfinite(max_cost_s) or max_cost_s <= 0:
            raise ValueError("max_cost_s must be finite and positive")
        connectors = self.origins.get(str(origin_id))
        if not connectors:
            return None
        starts: dict[tuple[tuple[str, str, int], bool], float] = {}
        for stop_id, walking_s in connectors:
            for state in self._states_by_stop[stop_id]:
                key = (state, False)
                starts[key] = min(walking_s, starts.get(key, inf))

        distances = bounded_shortest_paths(self._adjacency, starts, max_cost_s)
        best_stop: dict[str, float] = {}
        for (state, has_ridden), cost in distances.items():
            if not has_ridden:
                continue
            stop_id = self._stop_by_state[state]
            if cost < best_stop.get(stop_id, inf):
                best_stop[stop_id] = cost

        costs = np.full(len(self.job_ids), np.inf, dtype=np.float64)
        for stop_id, arrival_s in best_stop.items():
            remaining_s = max_cost_s - arrival_s
            for j, egress_s in self._jobs_at_stop.get(stop_id, ()):
                if egress_s > remaining_s:
                    break  # Sorted by walking egress cost.
                candidate = arrival_s + egress_s
                if candidate < costs[j]:
                    costs[j] = candidate
        return costs
