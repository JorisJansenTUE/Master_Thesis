"""Directed car and bicycle routing between snapped origins and STATENT jobs.

Uses the user's existing in-memory scenario-specific road connectors. A point fraction is measured
along a directed MATSim link. The entry/exit fractions are accounted for, and
travel on a shared directed link can go directly between two locations.
Off-network snapping distance is NOT added to travel time.

One Dijkstra search per origin; all STATENT destination IDs occur once in the
returned array even if they have multiple directed link connectors.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import inf, isfinite

import networkx as nx
import numpy as np
import pandas as pd

from ._shortest_paths import bounded_shortest_paths

_REQUIRED = {"point_id", "link_id", "from_node", "to_node", "fraction"}


@dataclass(frozen=True, slots=True)
class _Connector:
    link_id: str
    u: str
    v: str
    fraction: float
    cost_s: float


class RoadRouter:
    """Shortest perceived (bike) or experienced (car) cost to each job.

    ``routing_cost_s`` comes from the already-loaded ModeNetwork graph. The
    object is read-only after construction and may be reused across origins.
    Build a new router for each scenario graph/link-time snapshot.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        origin_connectors: pd.DataFrame,
        job_connectors: pd.DataFrame,
        jobs: pd.DataFrame,
        *,
        cost_attribute: str = "routing_cost_s",
    ) -> None:
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("RoadRouter requires a NetworkX MultiDiGraph")
        if not cost_attribute:
            raise ValueError("cost_attribute must be nonempty")
        self.job_ids, self.job_counts, index = _prepare_jobs(jobs)
        self.cost_attribute = cost_attribute

        # Collapse parallel directed links to their least-cost node-to-node
        # traversal, but retain individual link costs for fractional access.
        adjacency: dict[str, dict[str, float]] = defaultdict(dict)
        edge_costs: dict[tuple[str, str, str], float] = {}
        for u, v, key, data in graph.edges(keys=True, data=True):
            u, v, key = str(u), str(v), str(key)
            raw = data.get(cost_attribute)
            cost = float(raw) if raw is not None else float("nan")
            if not isfinite(cost) or cost < 0:
                raise ValueError(f"Invalid {cost_attribute} on link {key!r}: {raw!r}")
            edge_costs[(u, v, key)] = cost
            adjacency[u][v] = min(cost, adjacency[u].get(v, inf))
        self._adjacency = {u: tuple(neighbors.items()) for u, neighbors in adjacency.items()}

        def prepare(frame: pd.DataFrame, label: str) -> dict[str, tuple[_Connector, ...]]:
            missing = _REQUIRED.difference(frame.columns)
            if missing:
                raise ValueError(f"{label} missing columns: {sorted(missing)}")
            by_point: dict[str, list[_Connector]] = defaultdict(list)
            for point_id, lid, u, v, raw_frac in frame[
                ["point_id", "link_id", "from_node", "to_node", "fraction"]
            ].itertuples(index=False, name=None):
                point_id, lid, u, v = map(str, (point_id, lid, u, v))
                fraction = float(raw_frac)
                if not isfinite(fraction) or not 0 <= fraction <= 1:
                    raise ValueError(f"{label}: invalid fraction at {point_id}: {fraction}")
                key = (u, v, lid)
                if key not in edge_costs:
                    raise ValueError(
                        f"{label} connector for {point_id!r} references unavailable "
                        f"directed link {lid!r} ({u}->{v}). Regenerate snapping "
                        "for the scenario network if its permissions changed."
                    )
                by_point[point_id].append(_Connector(lid, u, v, fraction, edge_costs[key]))
            return {point_id: tuple(values) for point_id, values in by_point.items()}

        self.origins = prepare(origin_connectors, "Origin")
        destinations = prepare(job_connectors, "Job")
        if extra := set(destinations).difference(index):
            raise ValueError(f"Job connectors contain unknown job IDs: {sorted(extra)[:5]}")

        self._jobs_at_node: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._jobs_on_link: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for jid, connectors in destinations.items():
            j = index[jid]
            for c in connectors:
                self._jobs_at_node[c.u].append((j, c.fraction * c.cost_s))
                self._jobs_on_link[c.link_id].append((j, c.fraction))
                # If the point lies exactly at the downstream node, routes
                # arriving there need not traverse the same link again.
                if c.fraction >= 1.0 - 1e-12:
                    self._jobs_at_node[c.v].append((j, 0.0))

        self.attached_origin_count = len(self.origins)
        self.attached_job_count = len(destinations)
        self.attached_jobs_total = float(sum(self.job_counts[index[j]] for j in destinations))

    def shortest_job_costs(self, origin_id: str, max_cost_s: float) -> np.ndarray | None:
        """Return one minimum cost per job (inf if not reachable within cutoff).

        None means the origin has no snapped road connector. Costs are seconds;
        for bicycle they are *perceived* seconds (infrastructure VTT).
        """
        if not isfinite(max_cost_s) or max_cost_s <= 0:
            raise ValueError("max_cost_s must be finite and positive")
        connectors = self.origins.get(str(origin_id))
        if not connectors:
            return None
        costs = np.full(len(self.job_ids), np.inf, dtype=np.float64)
        starts: dict[str, float] = {}
        for c in connectors:
            remaining = (1.0 - c.fraction) * c.cost_s
            starts[c.v] = min(starts.get(c.v, inf), remaining)
            if c.fraction <= 1e-12:
                starts[c.u] = 0.0
            # Partial link from the origin to downstream jobs on the SAME
            # directed link; node-only Dijkstra would miss these trips.
            for j, destination_frac in self._jobs_on_link.get(c.link_id, ()):
                if destination_frac + 1e-12 >= c.fraction:
                    direct_s = max(0.0, destination_frac-c.fraction) * c.cost_s
                    if direct_s <= max_cost_s:
                        costs[j] = min(costs[j], direct_s)

        distance = bounded_shortest_paths(self._adjacency, starts, max_cost_s)
        for node, to_node_s in distance.items():
            for j, partial_cost in self._jobs_at_node.get(node, ()):
                candidate = to_node_s + partial_cost
                if candidate < costs[j] and candidate <= max_cost_s:
                    costs[j] = candidate
        return costs


def _prepare_jobs(jobs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if "job_id" not in jobs or "jobs" not in jobs:
        raise ValueError("jobs needs job_id and jobs columns")
    ids = jobs["job_id"].astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate STATENT job_id")
    counts = pd.to_numeric(jobs["jobs"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Job counts must be finite and nonnegative")
    return ids, counts, {jid: i for i, jid in enumerate(ids)}
