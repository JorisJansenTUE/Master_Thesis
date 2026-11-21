"""One-hour general-road travel-time-change targets on the fixed directed graph.

Predicts only motor-road ('car') links. Bus-specific *_spec, bicycle and rail
links remain in the graph for structure but have no travel-time target.
"""
from __future__ import annotations

import gzip
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_surrogate.preprocessing.static.input_tensors import _check_node_order

CHANNELS = ("road",)
SLOTS = ("segment_id", "from_node", "to_node")


def _read_network_modes(path: Path) -> dict[str, tuple[set[str], str, str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    links = {}
    with opener(path, "rb") as stream:
        for _, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag != "link":
                continue
            lid = elem.attrib["id"]
            if lid in links:
                raise ValueError(f"Duplicate network link {lid}")
            links[lid] = (set(elem.attrib.get("modes", "").split(",")),
                          elem.attrib["from"], elem.attrib["to"])
            elem.clear()
    if not links:
        raise ValueError(f"No links in {path}")
    return links


def _eligible_links(nodes: pd.DataFrame, scenario: Path, hour: int) -> pd.DataFrame:
    """Return car-accessible MATSim road links aligned to fixed node IDs."""
    _check_node_order(nodes)
    if not 0 <= hour <= 23:
        raise ValueError("hour must be in 0..23")
    mapping = pd.read_csv(scenario / "link_mapping.csv.gz", dtype={
        "link_id": str, "segment_id": str, "from_node": str,
        "to_node": str, "baseline_link_id": str, "link_kind": str,
    })
    if not {"link_id", *SLOTS}.issubset(mapping):
        raise ValueError("Incomplete scenario link mapping")
    if mapping.link_id.isna().any() or mapping.link_id.duplicated().any():
        raise ValueError("Missing or duplicate MATSim link IDs")
    mapping = mapping.merge(nodes[["node_id", *SLOTS]], on=list(SLOTS),
                            how="left", validate="many_to_one")
    if mapping.node_id.isna().any():
        raise ValueError("Scenario mapping contains links outside the fixed graph")
    baseline_ids = {
        lid: int(row.node_id)
        for row in nodes.itertuples(index=False)
        for lid in json.loads(row.baseline_link_ids)
    }
    by_link = mapping.set_index("link_id")["node_id"]
    if any(lid not in by_link or int(by_link[lid]) != node
           for lid, node in baseline_ids.items()):
        raise ValueError("Baseline MATSim links are missing or moved in scenario mapping")

    network = _read_network_modes(scenario / "network.xml.gz")
    selected = []
    for row in mapping.itertuples(index=False):
        link = network.get(row.link_id)
        if link is None:
            raise ValueError(f"Mapped link absent from scenario network: {row.link_id}")
        modes, u, v = link
        if (u, v) != (row.from_node, row.to_node):
            raise ValueError(f"Link direction mismatch: {row.link_id}")
        # Match the road eligibility mask derived from model features.
        if "car" in modes and not row.link_id.endswith("_spec"):
            selected.append((row.link_id, int(row.node_id)))
    result = pd.DataFrame(selected, columns=["link_id", "node_id"])
    if result.empty:
        raise ValueError("No eligible general-road links")
    if result.duplicated("node_id").any():
        raise ValueError("Multiple car links map to one fixed graph node")

    col = f"TRAVELTIME{hour}-{hour+1}avg"
    stats = pd.read_csv(scenario / "targets" / "linkstats.txt.gz", sep="\t",
                        usecols=["LINK", "FREESPEED", col], dtype={"LINK": str})
    if stats.LINK.isna().any() or stats.LINK.duplicated().any():
        raise ValueError("Missing or duplicate LINK in linkstats")
    result = result.merge(stats, left_on="link_id", right_on="LINK",
                          how="left", validate="one_to_one")
    if result.LINK.isna().any():
        raise ValueError("Eligible MATSim links missing from linkstats")
    speed = result.FREESPEED.to_numpy(dtype=np.float64)
    if not (np.isfinite(speed) & (speed > 0)).all():
        raise ValueError("Car-accessible road link has nonfinite or nonpositive speed")
    tt = result[col].to_numpy(dtype=np.float64)
    if not np.isfinite(tt).all() or (tt <= 0).any():
        raise ValueError("Eligible road link has nonfinite or nonpositive travel time")
    return result.rename(columns={col: "tt_s"})


def build_baseline_travel_times(nodes: pd.DataFrame, baseline: Path,
                                hour: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Baseline road travel time and presence, both [N,1], built once."""
    eligible = _eligible_links(nodes, baseline, hour)
    values = np.zeros((len(nodes), 1), dtype=np.float32)
    present = np.zeros((len(nodes), 1), dtype=bool)
    for row in eligible.itertuples(index=False):
        i = int(row.node_id)
        values[i, 0] = float(row.tt_s)
        present[i, 0] = True
    return values, present


def build_delta_targets(nodes: pd.DataFrame, scenario: Path,
                        baseline_tt: np.ndarray, baseline_present: np.ndarray,
                        hour: int = 8) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return signed road travel-time changes [N,1] and eligibility mask [N,1]."""
    n = len(nodes)
    if baseline_tt.shape != (n, 1) or baseline_present.shape != (n, 1):
        raise ValueError("Baseline travel-time tensors must be [N,1]")
    eligible = _eligible_links(nodes, scenario, hour)
    delta = np.zeros((n, 1), dtype=np.float32)
    valid = np.zeros((n, 1), dtype=bool)
    baseline_ids = {
        lid: int(row.node_id)
        for row in nodes.itertuples(index=False)
        for lid in json.loads(row.baseline_link_ids)
    }
    for row in eligible.itertuples(index=False):
        i = int(row.node_id)
        if not baseline_present[i, 0] or baseline_ids.get(row.link_id) != i:
            raise ValueError(f"General road link lacks matching baseline: {row.link_id}")
        delta[i, 0] = float(row.tt_s) - baseline_tt[i, 0]
        valid[i, 0] = True
    if not valid.any():
        raise ValueError("No eligible general-road targets")
    return delta, valid, {"road_targets": int(valid.sum())}
