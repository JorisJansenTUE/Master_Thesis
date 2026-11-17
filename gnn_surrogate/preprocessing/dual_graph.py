"""Build an immutable, directed-street dual graph for GNN preprocessing.

Each ordinary physical StreetSegment has two permanently indexed directions,
including directions unavailable in the original network. Physical PT stop
self-loops have one position. A dual edge i -> j denotes a *potential*
head-to-tail street movement, not a mode-specific or scenario-specific turn.

Input comes only from the original baseline network and the dataset-root
street_segments.csv.gz. Never pass an intervention scenario network here.
The optional baseline traffic file also must come from the baseline scenario.

Returns a node table and an edge index; saving is handled by build_graph.py.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from network_sampling.network.matsim_network import load_matsim_network

MODE_NAMES = ("car", "truck", "bike", "bus", "pt")
FLOW_NAMES = ("cars", "trucks", "bikes", "buses")


def build_dual_graph(
    segments_path: Path,
    *,
    baseline_network_path: Path,
    baseline_flows_path: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Construct one fixed dual graph with infrastructure from the true baseline.

    `segments_path` is the shared dataset-root street_segments.csv.gz.
    `baseline_network_path` is either the original network_with_widths.xml.gz
    or the no-intervention baseline_00000/network.xml.gz generated from it.
    """
    segments = pd.read_csv(
        segments_path,
        dtype={"segment_id": str, "node_a": str, "node_b": str,
               "osm_way_id": str, "baseline_link_ids": str},
    )
    required = {"segment_id", "node_a", "node_b", "length_m",
                "total_width_m", "osm_way_id", "baseline_link_ids"}
    if missing := required.difference(segments.columns):
        raise ValueError(f"Missing street_segments columns: {sorted(missing)}")
    if segments.segment_id.isna().any() or segments.segment_id.duplicated().any():
        raise ValueError("Missing or duplicate physical street segment IDs")
    if segments[["node_a", "node_b", "length_m"]].isna().any().any():
        raise ValueError("Street endpoints and lengths must be present")
    if (segments.length_m <= 0).any():
        raise ValueError("Every physical segment must have positive length")

    baseline = load_matsim_network(baseline_network_path)

    # Stable identity comes ONLY from the immutable physical segment export.
    records = []
    segment_links = {}
    for seg in segments.sort_values("segment_id").itertuples(index=False):
        try:
            lids = json.loads(seg.baseline_link_ids)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid baseline_link_ids for {seg.segment_id}") from exc
        if not isinstance(lids, list) or not all(isinstance(v, str) for v in lids):
            raise ValueError(f"Expected JSON link IDs for {seg.segment_id}")
        segment_links[seg.segment_id] = lids
        directions = [(seg.node_a, seg.node_b, "a_to_b")]
        if seg.node_a != seg.node_b:
            directions.append((seg.node_b, seg.node_a, "b_to_a"))
        for u, v, direction in directions:
            records.append({
                "node_id": len(records),
                "segment_id": seg.segment_id,
                "from_node": u, "to_node": v,
                "direction": direction,
                "is_pt_self_loop": u == v,
                "length_m": float(seg.length_m),
                "total_width_m": seg.total_width_m,
                "osm_way_id": seg.osm_way_id,
            })
    nodes = pd.DataFrame.from_records(records)
    key_to_id = {
        (r.segment_id, r.from_node, r.to_node): int(r.node_id)
        for r in nodes.itertuples(index=False)
    }

    # Fixed *potential* directed topology: accessibility will be applied later.
    outgoing = defaultdict(list)
    for row in nodes.itertuples(index=False):
        outgoing[row.from_node].append(int(row.node_id))
    pairs = sorted({(int(row.node_id), nxt)
                    for row in nodes.itertuples(index=False)
                    for nxt in outgoing[row.to_node]
                    if int(row.node_id) != nxt})
    edge_index = np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T.copy()

    # Provenance and geometry, plus static baseline infrastructure fields.
    for col in ("from_x", "from_y", "to_x", "to_y", "mid_x", "mid_y"):
        nodes[col] = np.nan
    nodes["baseline_active"] = 0
    nodes["baseline_link_count"] = 0
    nodes["baseline_capacity"] = 0.0
    nodes["baseline_freespeed_m_s"] = np.nan
    nodes["baseline_permlanes"] = 0.0
    for mode in MODE_NAMES:
        nodes[f"baseline_allows_{mode}"] = 0
    for mode in FLOW_NAMES:
        nodes[f"baseline_daily_{mode}"] = np.nan
    nodes["baseline_link_ids"] = "[]"
    nodes["baseline_highways"] = None  # Retain original categories; encode later.

    for row in nodes.itertuples(index=False):
        if row.from_node not in baseline.nodes or row.to_node not in baseline.nodes:
            raise ValueError(f"Missing baseline node coordinates for {row.segment_id}")
        a, b = baseline.nodes[row.from_node], baseline.nodes[row.to_node]
        nodes.loc[row.node_id, ["from_x", "from_y", "to_x", "to_y", "mid_x", "mid_y"]] = [
            a.x, a.y, b.x, b.y, (a.x + b.x) / 2, (a.y + b.y) / 2,
        ]

    directed_ids = defaultdict(list)
    seen = set()
    for sid, lids in segment_links.items():
        for lid in lids:
            if lid in seen:
                raise ValueError(f"Original MATSim link {lid} belongs to multiple segments")
            seen.add(lid)
            link = baseline.links.get(lid)
            if link is None:
                raise ValueError(f"Original MATSim link {lid} missing from baseline network")
            key = (sid, str(link.from_node), str(link.to_node))
            if key not in key_to_id:
                raise ValueError(f"Original MATSim link {lid} contradicts segment {sid}")
            directed_ids[key_to_id[key]].append(lid)

    for node_id, lids in directed_ids.items():
        links = [baseline.links[lid] for lid in lids]
        nodes.at[node_id, "baseline_link_ids"] = json.dumps(sorted(lids))
        nodes.at[node_id, "baseline_active"] = 1
        nodes.at[node_id, "baseline_link_count"] = len(lids)
        nodes.at[node_id, "baseline_capacity"] = sum(l.capacity for l in links)
        nodes.at[node_id, "baseline_permlanes"] = sum(l.permlanes for l in links)
        # MATSim links in this pipeline record road class as osm:way:highway.
        nodes.at[node_id, "baseline_highways"] = json.dumps(sorted({
            str(value) for l in links
            if (value := l.attributes.get("highway", l.attributes.get("osm:way:highway")))
            is not None
        }))
        weights = np.asarray([max(l.capacity, 0.0) for l in links])
        speeds = np.asarray([l.freespeed for l in links])
        nodes.at[node_id, "baseline_freespeed_m_s"] = float(
            np.average(speeds, weights=weights) if weights.sum() else speeds.mean()
        )
        for mode in MODE_NAMES:
            nodes.at[node_id, f"baseline_allows_{mode}"] = int(
                any(mode in link.allowed_modes for link in links)
            )

    if baseline_flows_path is not None:
        raw = pd.read_csv(baseline_flows_path, dtype={"link_id": str})
        required_flow = {"link_id", "hour", *FLOW_NAMES}
        if missing := required_flow.difference(raw.columns):
            raise ValueError(f"Missing baseline flow columns: {sorted(missing)}")
        raw = raw.loc[raw.hour.between(0, 23)]
        if raw.duplicated(["link_id", "hour"]).any():
            raise ValueError("Duplicate baseline link-hour records")
        totals = raw.groupby("link_id")[list(FLOW_NAMES)].sum()
        for node_id, lids in directed_ids.items():
            selected = totals.reindex(lids, fill_value=0)
            for mode in FLOW_NAMES:
                nodes.at[node_id, f"baseline_daily_{mode}"] = float(selected[mode].sum())
        for mode in FLOW_NAMES:
            nodes.loc[nodes["baseline_active"] == 0, f"baseline_daily_{mode}"] = 0.0

    return nodes, edge_index
