"""Attach origin/job points to a mode-permitted road network.

NetworkX retains all MATSim directed links. We group only geometrically
identical opposite-direction links for the *snapping search*, then restore
one connector for each permitted directed link on the chosen road segment.
This preserves one-way restrictions for the future routing step.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


@dataclass
class PointAttachments:
    connectors: pd.DataFrame
    total_points: int
    attached_points: int
    unsnapped_ids: tuple[str, ...]
    nearest_distances_m: dict[str, float]

    @property
    def unsnapped_count(self) -> int:
        return self.total_points - self.attached_points

    def save(self, path: str) -> None:
        from pathlib import Path
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.connectors.to_csv(output, index=False)


class RoadSnapIndex:
    """Reusable spatial index of directed car OR bike links in a NetworkX graph.

    The graph's node ``x/y`` must be Swiss LV95 coordinates (metres). Every
    edge must have the original MATSim link ID as its MultiDiGraph edge key.
    """

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("Expected a NetworkX MultiDiGraph")
        by_segment: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
        for u, v, key in graph.edges(keys=True):
            if u == v:
                continue  # zero-length loops are not suitable for snapping
            pair = tuple(sorted((str(u), str(v))))
            by_segment.setdefault(pair, []).append((str(u), str(v), str(key)))

        self.segments: list[LineString] = []
        self.links_by_segment: list[list[tuple[str, str, str]]] = []
        self.segment_nodes: list[tuple[str, str]] = []
        for (u, v), links in by_segment.items():
            p = (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"]))
            q = (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"]))
            if p == q:
                continue
            if not all(isfinite(val) for val in (*p, *q)):
                raise ValueError(f"Invalid node coordinates on segment {u} -> {v}")
            self.segments.append(LineString((p, q)))
            self.segment_nodes.append((u, v))
            self.links_by_segment.append(links)
        if not self.segments:
            raise ValueError("No nonzero-length links available for snapping")
        self.tree = STRtree(self.segments)

    def attach(
        self,
        points: pd.DataFrame,
        *,
        id_column: str,
        x_column: str,
        y_column: str,
        max_distance_m: float,
    ) -> PointAttachments:
        """Return directed-link connectors, one row per link candidate.

        ``fraction`` is measured along each *directed* link (0 at its from-node,
        1 at its to-node). Off-network distance is a snapping diagnostic
        only; partial-link traversal is handled later in Step 4.

        The same point may have two connectors for opposing directions. Step 4
        must count a destination's jobs at most once, using its ``point_id``.
        """
        if max_distance_m <= 0:
            raise ValueError("max_distance_m must be positive")
        required = {id_column, x_column, y_column}
        if missing := required.difference(points.columns):
            raise ValueError(f"Missing point columns: {sorted(missing)}")
        if points[id_column].duplicated().any():
            raise ValueError(f"Duplicate point IDs in {id_column}")

        rows: list[dict] = []
        unsnapped: list[str] = []
        nearest_distances_m: dict[str, float] = {}
        for point_id, x, y in points[[id_column, x_column, y_column]].itertuples(index=False, name=None):
            point_id = str(point_id)
            x, y = float(x), float(y)
            if not isfinite(x) or not isfinite(y):
                raise ValueError(f"Invalid coordinates for {point_id}")
            point = Point(x, y)
            segment_idx = int(self.tree.nearest(point))
            segment = self.segments[segment_idx]
            distance = float(segment.distance(point))
            # Preserve distances also for rejected points, for GIS diagnostics.
            nearest_distances_m[point_id] = distance
            if distance > max_distance_m:
                unsnapped.append(point_id)
                continue
            fraction = float(segment.project(point, normalized=True))
            a, b = self.segment_nodes[segment_idx]
            for u, v, link_id in self.links_by_segment[segment_idx]:
                directed_fraction = fraction if (u, v) == (a, b) else 1.0 - fraction
                rows.append({
                    "point_id": point_id,
                    "link_id": link_id,
                    "from_node": u,
                    "to_node": v,
                    "fraction": max(0.0, min(1.0, directed_fraction)),
                    "off_network_distance_m": distance,
                })
        columns = ["point_id", "link_id", "from_node", "to_node", "fraction",
                   "off_network_distance_m"]
        connectors = pd.DataFrame(rows, columns=columns)
        return PointAttachments(
            connectors=connectors,
            total_points=len(points),
            attached_points=len(points) - len(unsnapped),
            unsnapped_ids=tuple(unsnapped),
            nearest_distances_m=nearest_distances_m,
        )
