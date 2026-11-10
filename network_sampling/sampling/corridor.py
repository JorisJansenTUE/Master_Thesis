
"""Path-based cycling corridor sampler for physical MATSim street segments.

Assumptions:
- Each physical MultiGraph edge has key=StreetSegment.id or data['segment_id'].
- `index` exposes get(segment_id) -> StreetSegment.
- `network.links.values()` exposes MATSim links with from_node, to_node, allowed_modes.
- `candidates` comes from the existing local intervention eligibility layer.
- Each cycling candidate creates/preserves the cycling connection intended by its type;
  directed modal connectivity must still be checked after application.
"""
from __future__ import annotations

import random
from collections import defaultdict
from math import isfinite

import networkx as nx

from network_sampling.interventions.compatibility import interventions_compatible
from network_sampling.interventions.intervention_types import Intervention, InterventionType
from network_sampling.network.matsim_network import MatsimNetwork
from network_sampling.network.street_segments import StreetSegmentIndex

BIKE_TYPES = {"protected_bike_lane", "bicycle_street", "pt_bike_street"}


def _is_protected_bike_link(link) -> bool:
    """Recognize baseline bike-only infrastructure or an existing bicycle street."""
    return (
        set(link.allowed_modes) == {"bike"}
        or str(getattr(link, "attributes", {}).get("cyclestreet", "")).strip().lower() == "yes"
    )


def _intervention_weight(iv) -> float:
    """Prefer less intrusive options, matching the random sampler's weights."""
    variant = iv.road_space_variant.value if iv.road_space_variant else None
    if variant in (None, "keep_car"):
        return 4.0
    if variant == "remove_one_lane":
        return 2.0
    if variant == "remove_car":
        # Directional targets distinguish one removed car direction from two.
        return 0.25 if len(iv.target_link_ids) > 1 else 1.0
    raise ValueError(f"Unknown road-space variant: {variant}")


def sample_corridors(
    graph: nx.MultiGraph,
    index: StreetSegmentIndex,
    network: MatsimNetwork,
    candidates: list[Intervention],
    budget_m: float,
    *,
    max_corridor_length_m: float | None = NotImplementedError,
    rng: random.Random | None = None,
) -> list[Intervention]:
    """Sample bike-oriented, anchored, simple-path corridor interventions.

    Keep starting corridors while feasible paths fit the remaining budget.
    Each corridor is limited by max_corridor_length_m when provided.
    No MATSim objects are modified. Global mode connectivity is NOT validated.
    """
    if budget_m < 0 or not isfinite(budget_m):
        raise ValueError("budget_m must be finite and nonnegative")
    if max_corridor_length_m is not None and (
        not isfinite(max_corridor_length_m) or max_corridor_length_m <= 0
    ):
        raise ValueError("max_corridor_length_m must be finite and positive")
    if not isinstance(graph, nx.MultiGraph) or graph.is_directed():
        raise TypeError("Expected an undirected nx.MultiGraph")

    rng = rng if rng is not None else random.Random()
    by_segment = defaultdict(list)
    for iv in candidates:
        if iv.intervention_type.value in BIKE_TYPES:
            by_segment[iv.segment_id].append(iv)

    # Filter the existing physical MultiGraph; preserve parallel segment keys.
    available = nx.MultiGraph()
    for u, v, key, data in graph.edges(keys=True, data=True):
        sid = data.get("segment_id")
        if sid is None:
            sid = getattr(data.get("segment"), "id", key)
        if sid not in by_segment or u == v:
            continue
        seg = index.get(sid)
        if not (isfinite(seg.length_m) and seg.length_m > 0):
            raise ValueError(f"Invalid length for physical segment {sid}")
        if {u, v} != {seg.node_a, seg.node_b}:
            raise ValueError(f"Physical graph endpoints differ for {sid}")
        available.add_edge(u, v, key=sid, length_m=seg.length_m)

    if by_segment and available.number_of_edges() == 0:
        raise ValueError("No candidate segment IDs match physical MultiGraph edges")

    # A normal car+bike street is NOT an anchor. Only endpoints of baseline
    # bike-only links or streets with cyclestreet=yes qualify.
    protected_nodes = {
        node
        for link in network.links.values()
        if _is_protected_bike_link(link)
        for node in (link.from_node, link.to_node)
    }
    anchors = protected_nodes & set(available.nodes)
    selected = []
    spent = 0.0

    while spent < budget_m - 1e-9:
        anchors &= set(available.nodes)
        if not anchors or available.number_of_edges() == 0:
            break

        remaining = budget_m - spent
        target_m = min(remaining, max_corridor_length_m or remaining)

        # Multi-source Dijkstra connects to the nearest protected-infrastructure
        # endpoint, including newly selected PBL/bicycle-street endpoints,
        # using only eligible cycling-intervention edges.
        distance, paths = nx.multi_source_dijkstra(
            available, sorted(anchors), weight="length_m"
        )

        # The seed segment is explicitly included even if an alternative
        # shortest path happens to bypass it.
        options = []
        for u, v, sid, data in available.edges(keys=True, data=True):
            near, far = (u, v) if distance.get(u, float("inf")) <= distance.get(v, float("inf")) else (v, u)
            required_length = distance.get(near, float("inf")) + data["length_m"]
            if required_length <= target_m + 1e-9 and far not in paths[near]:
                options.append((sid, near, far, far not in anchors))
        if not options:
            break

        # Prefer remote seeds when available, but support an entirely
        # bike-covered study area as well.
        remote_options = [o for o in options if o[3]]
        sid, near, far, _ = rng.choice(sorted(remote_options or options))

        node_path = paths[near]
        segment_path = []
        for u, v in zip(node_path, node_path[1:]):
            edge_id = min(
                available[u][v],
                key=lambda k: (available[u][v][k]["length_m"], str(k)),
            )
            segment_path.append(edge_id)
        segment_path.append(sid)
        node_path = node_path + [far]
        length_m = sum(index.get(s).length_m for s in segment_path)
        visited = set(node_path)

        # Extend from the remote end. Never revisit a node in this corridor.
        while True:
            end = node_path[-1]
            last_way = index.get(segment_path[-1]).osm_way_id
            steps, weights = [], []
            for next_node, edges in available[end].items():
                if next_node in visited:
                    continue
                for edge_id, data in edges.items():
                    seg = index.get(edge_id)
                    if edge_id in segment_path or length_m + seg.length_m > target_m + 1e-9:
                        continue
                    steps.append((edge_id, next_node))
                    weights.append(3.0 if last_way and seg.osm_way_id == last_way else 1.0)
            if not steps:
                break
            next_sid, next_node = rng.choices(steps, weights=weights, k=1)[0]
            segment_path.append(next_sid)
            node_path.append(next_node)
            visited.add(next_node)
            length_m += index.get(next_sid).length_m

        # Choose a locally eligible intervention for each segment. Prefer
        # maintaining the previous cycling infrastructure type, not mandating it.
        previous_type = None
        for edge_id in segment_path:
            choices = by_segment[edge_id]
            weights = []
            for iv in choices:
                weight = _intervention_weight(iv)
                if iv.intervention_type == previous_type:
                    weight *= 3.0
                weights.append(weight)
            iv = rng.choices(choices, weights=weights, k=1)[0]

            # Every segment appears only once; still call the central rule
            # rather than implementing a separate compatibility definition.
            if any(
                not interventions_compatible(iv, prev, network)
                for prev in selected if prev.segment_id == iv.segment_id
            ):
                raise RuntimeError(f"Unexpected compatibility conflict on {edge_id}")
            selected.append(iv)
            previous_type = iv.intervention_type

        spent += length_m
        # New PBLs and bicycle streets become protected anchors for later
        # corridors. PT-bike streets can connect a corridor but do not by
        # themselves count as new protected infrastructure.
        for iv in selected[-len(segment_path):]:
            if iv.intervention_type.value in {"protected_bike_lane", "bicycle_street"}:
                seg = index.get(iv.segment_id)
                anchors.update((seg.node_a, seg.node_b))
        for edge_id in segment_path:
            seg = index.get(edge_id)
            available.remove_edge(seg.node_a, seg.node_b, key=edge_id)

    return selected
