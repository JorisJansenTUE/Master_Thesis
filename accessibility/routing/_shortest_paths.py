"""Small, bounded multi-source Dijkstra for preweighted directed adjacency lists.

Unlike a temporary NetworkX super-source, this accepts a different entry cost
for each starting node without mutating a shared graph. Both modal routers use
this same implementation; no journey or accessibility policy lives here.
"""
from __future__ import annotations

from heapq import heappop, heappush
from math import inf, isfinite
from typing import Hashable, Mapping, Sequence

Node = Hashable


def bounded_shortest_paths(
    adjacency: Mapping[Node, Sequence[tuple[Node, float]]],
    starts: Mapping[Node, float],
    max_cost_s: float,
) -> dict[Node, float]:
    """Minimum costs from nonnegative source labels up to an inclusive cutoff."""
    if not isfinite(max_cost_s) or max_cost_s <= 0:
        raise ValueError("max_cost_s must be finite and positive")

    distance: dict[Node, float] = {}
    heap: list[tuple[float, int, Node]] = []
    counter = 0
    for node, raw in starts.items():
        cost = float(raw)
        if not isfinite(cost) or cost < 0:
            raise ValueError(f"Invalid entry cost for node {node!r}: {cost}")
        if cost <= max_cost_s and cost < distance.get(node, inf):
            distance[node] = cost
            heappush(heap, (cost, counter, node))
            counter += 1

    while heap:
        cost, _, node = heappop(heap)
        if cost != distance[node]:
            continue
        for target, weight in adjacency.get(node, ()):
            total = cost + weight
            if total <= max_cost_s and total < distance.get(target, inf):
                distance[target] = total
                heappush(heap, (total, counter, target))
                counter += 1
    return distance
