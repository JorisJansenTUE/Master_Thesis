"""Independent inspection-only audit of interventions and GNN dynamic features.

Reconstruct expected link-level changes from the original network and
interventions.json, then compare with the saved scenario and [N, 9] features.
This intentionally does NOT call the sampler's apply_interventions(): we want
an independent check of its effects. Not used during training/data generation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

from network_sampling.network.matsim_network import load_matsim_network
from .scenario_features import DEFAULT_DYNAMIC_COLUMNS

CAR_MODES = {"car", "car_passenger", "truck", "taxi"}
VARIANTS = {"keep_car", "remove_one_lane", "remove_car"}
KINDS = {"bicycle_street", "protected_bike_lane", "dedicated_bus_lane", "destination_only"}
BICYCLE_SPEED = 25.0 / 3.6  # Actual code value; ignore outdated function docstrings
DESTINATION_SPEED = 20.0 / 3.6
PBL_WIDTH_PER_DIRECTION = 2.0  # PBL_WIDTH_M / 2


@dataclass
class AuditResult:
    interventions: int
    by_type_variant: Counter
    original_links_checked: int
    new_links_checked: int
    expected_feature_counts: dict[str, int]
    bus_routes_checked: int


def _close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, atol=1e-7, rtol=1e-8, equal_nan=True))


def _clone(link):
    return replace(link, allowed_modes=set(link.allowed_modes),
                   attributes=dict(link.attributes))


def _pbl(template, link_id: str, u: str, v: str):
    attrs = dict(template.attributes)
    attrs.update({"osm:way:highway": "cycleway", "protected_bike_lane": "yes",
                  "carriageway_width_m": PBL_WIDTH_PER_DIRECTION})
    return replace(template, id=link_id, from_node=u, to_node=v,
                   allowed_modes={"bike"}, permlanes=1.0,
                   capacity=template.capacity / template.permlanes,
                   attributes=attrs)


def _unique_id(links, preferred: str) -> str:
    if preferred not in links:
        return preferred
    i = 2
    while f"{preferred}_{i}" in links:
        i += 1
    return f"{preferred}_{i}"


def _schedule_bus_links(schedule_path: Path) -> tuple[set[str], set[str]]:
    """Return scheduled bus route links and stop-facility link references."""
    parser = etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False)
    with (gzip.open(schedule_path, "rb") if schedule_path.suffix == ".gz"
          else schedule_path.open("rb")) as f:
        root = etree.parse(f, parser).getroot()
    route_links: set[str] = set()
    for route in root.iterfind(".//transitRoute"):
        if route.findtext("transportMode") == "bus":
            path = route.find("route")
            if path is not None:
                route_links.update(link.attrib["refId"] for link in path.iterfind("link"))
    stops = {stop.attrib["linkRefId"] for stop in
             root.iterfind(".//transitStops/stopFacility") if "linkRefId" in stop.attrib}
    return route_links, stops


def audit_scenario_interventions(
    nodes: pd.DataFrame,
    features: pd.DataFrame,
    *,
    baseline_network_path: Path,
    scenario_dir: Path,
) -> AuditResult:
    """Raise AssertionError with specific link/node context on a discrepancy.

    Checks all original links (including unchanged and PT links), all generated
    links, their mapped directions, model features and dedicated bus references.
    The baseline network is loaded *only for the inspection*, never per sample
    during dynamic feature generation.
    """
    scenario_dir = Path(scenario_dir)
    original = load_matsim_network(baseline_network_path)
    actual = load_matsim_network(scenario_dir / "network.xml.gz")
    entries = json.loads((scenario_dir / "interventions.json").read_text())
    mapping = pd.read_csv(scenario_dir / "link_mapping.csv.gz", dtype=str).fillna("")
    problems: list[str] = []
    def fail(msg: str) -> None:
        problems.append(msg)

    if len(features) != len(nodes) or not np.array_equal(features.node_id, nodes.node_id):
        raise AssertionError("Dynamic features do not match fixed node order")
    if not set(DEFAULT_DYNAMIC_COLUMNS).issubset(features.columns):
        raise AssertionError("Dynamic feature table is missing required columns")
    if mapping.link_id.duplicated().any():
        raise AssertionError("Duplicate link IDs in link_mapping.csv.gz")
    if not isinstance(entries, list):
        raise AssertionError("interventions.json must be a list")

    # Per-segment original links in the original (sorted) StreetSegment order.
    segments: dict[str, dict] = {}
    base_to_node: dict[str, int] = {}
    directional_slot: dict[tuple[str, str, str], int] = {}
    for row in nodes.itertuples(index=False):
        sid = str(row.segment_id)
        segments.setdefault(sid, {"links": set(), "dirs": set()})
        segments[sid]["dirs"].add((str(row.from_node), str(row.to_node)))
        directional_slot[(sid, str(row.from_node), str(row.to_node))] = int(row.node_id)
        for lid in json.loads(row.baseline_link_ids):
            segments[sid]["links"].add(str(lid))
            if lid in base_to_node:
                fail(f"Original MATSim link {lid} belongs to multiple fixed nodes")
            base_to_node[lid] = int(row.node_id)
    for info in segments.values():
        info["links"] = sorted(info["links"])

    # Validate graph's cached baseline references *independently*, so an old or
    # mismatched dual_nodes.csv.gz cannot make the scenario appear consistent.
    for row in nodes.itertuples(index=False):
        links = [original.links[lid] for lid in json.loads(row.baseline_link_ids)]
        car_links = [l for l in links if "car" in l.allowed_modes]
        cap = sum(l.capacity for l in car_links)
        speed = sum(l.capacity * l.freespeed for l in car_links) / cap if cap else 0.0
        if not _close(row.baseline_car_capacity, cap):
            fail(f"node {row.node_id}: cached baseline car capacity != original network")
        if not _close(row.baseline_car_freespeed_m_s, speed):
            fail(f"node {row.node_id}: cached baseline car speed != original network")
        for mode in ("car", "bike", "bus"):
            cached = int(getattr(row, f"baseline_allows_{mode}"))
            true = int(any(mode in l.allowed_modes for l in links))
            if cached != true:
                fail(f"node {row.node_id}: baseline {mode} access != original network")

    # Build an independent expected network from baseline + policy rules.
    # Generated links are added to the expected mapping alongside originals.
    expected = {lid: _clone(link) for lid, link in original.links.items()}
    expected_to_node = dict(base_to_node)
    expected_labels = {col: np.zeros(len(nodes), dtype=np.float32)
                       for col in DEFAULT_DYNAMIC_COLUMNS[:4]}
    by_type_variant = Counter()
    bus_replacements: dict[str, str] = {}
    used_segments = set()

    for index, entry in enumerate(entries):
        sid = str(entry["segment_id"])
        kind = entry["intervention_type"]
        variant = entry.get("road_space_variant")
        targets = [str(lid) for lid in entry.get("target_link_ids", [])]
        if kind not in KINDS or sid not in segments or sid in used_segments:
            fail(f"intervention #{index}: invalid/repeated kind or segment {kind!r}/{sid!r}")
            continue
        used_segments.add(sid)
        by_type_variant[(kind, variant)] += 1
        seg = segments[sid]
        links = seg["links"]
        directions = sorted(seg["dirs"])
        car_links = [lid for lid in links if "car" in original.links[lid].allowed_modes]
        if kind in {"bicycle_street", "destination_only"}:
            if variant is not None or targets:
                fail(f"{sid}: {kind} must not have a road-space variant or target IDs")
        else:
            if variant not in VARIANTS:
                fail(f"{sid}: invalid {kind} road_space_variant: {variant}")
                continue
            if kind == "dedicated_bus_lane" and len(targets) != 1:
                fail(f"{sid}: bus lane must have exactly one target car link")
                continue
            if kind == "protected_bike_lane" and variant == "keep_car" and targets:
                fail(f"{sid}: keep_car PBL should not target individual links")
            if variant != "keep_car" and not targets:
                fail(f"{sid}: {variant} needs at least one target link")
            for lid in targets:
                if lid not in car_links:
                    fail(f"{sid}: target {lid} is not an original car link on segment")
                    continue

        if kind in ("bicycle_street", "protected_bike_lane"):
            for u, v in directions:
                expected_labels[kind][directional_slot[sid, u, v]] = 1
        elif kind == "destination_only":
            for lid in car_links:
                expected_labels[kind][base_to_node[lid]] = 1
        elif targets and targets[0] in base_to_node:
            expected_labels[kind][base_to_node[targets[0]]] = 1

        if kind == "destination_only":
            for lid in car_links:
                link = expected[lid]
                attrs = dict(link.attributes)
                attrs["destination_road"] = "True"
                expected[lid] = replace(link, freespeed=DESTINATION_SPEED, attributes=attrs)
            continue

        if kind == "bicycle_street":
            for lid in car_links:
                link = expected[lid]
                attrs = dict(link.attributes)
                attrs["cyclestreet"] = "True"
                expected[lid] = replace(link, freespeed=BICYCLE_SPEED,
                                        allowed_modes=set(link.allowed_modes) | {"bike"},
                                        attributes=attrs)
            has_bike = {(expected[lid].from_node, expected[lid].to_node)
                        for lid in links if "bike" in expected[lid].allowed_modes}
            for u, v in directions:
                if (u, v) in has_bike:
                    continue
                if not car_links:
                    fail(f"{sid}: bicycle street has no car template")
                    break
                template = expected[car_links[0]]
                attrs = dict(template.attributes)
                attrs["cyclestreet"] = True
                attrs.pop("carriageway_width_m", None)
                newid = _unique_id(expected, f"{sid}__bike__{u}__{v}")
                expected[newid] = replace(template, id=newid, from_node=u, to_node=v,
                                          allowed_modes={"bike"}, freespeed=BICYCLE_SPEED,
                                          permlanes=1.0,
                                          capacity=template.capacity / template.permlanes,
                                          attributes=attrs)
                expected_to_node[newid] = directional_slot[sid, u, v]
                has_bike.add((u, v))
            continue

        # Road-space variant applied only to targeted original car links.
        if variant in {"remove_one_lane", "remove_car"}:
            for lid in targets:
                if lid not in car_links:
                    continue
                link = expected[lid]
                if variant == "remove_one_lane":
                    if link.permlanes <= 1:
                        fail(f"{sid}/{lid}: cannot remove lane from <=1-lane link")
                        continue
                    new_lanes = link.permlanes - 1
                    expected[lid] = replace(link, permlanes=new_lanes,
                                            capacity=link.capacity * new_lanes / link.permlanes)
                else:
                    expected[lid] = replace(link,
                                            allowed_modes=set(link.allowed_modes) - CAR_MODES)

        if kind == "protected_bike_lane":
            reuse = set()
            if variant == "remove_car":
                for lid in targets:
                    if lid not in car_links:
                        continue
                    link = expected[lid]
                    reuse.add((link.from_node, link.to_node))
                    expected[lid] = _pbl(link, lid, link.from_node, link.to_node)
            for u, v in directions:
                if (u, v) in reuse:
                    continue
                template_id = next((lid for lid in links if
                                    (expected[lid].from_node, expected[lid].to_node) == (u, v)),
                                   links[0])
                template = expected[template_id]
                newid = _unique_id(expected, f"{sid}__pbl__{u}__{v}")
                expected[newid] = _pbl(template, newid, u, v)
                expected_to_node[newid] = directional_slot[sid, u, v]

        elif kind == "dedicated_bus_lane" and targets[0] in car_links:
            lid = targets[0]
            orig = original.links[lid]
            general = expected[lid]
            expected[lid] = replace(general,
                                    allowed_modes=set(general.allowed_modes) - {"bus"})
            attrs = dict(orig.attributes)
            attrs.pop("carriageway_width_m", None)
            attrs["dedicated_bus_lane"] = "yes"
            newid = lid + "_spec"
            if newid in expected:
                fail(f"{sid}: generated bus link already exists: {newid}")
            else:
                expected[newid] = replace(orig, id=newid, permlanes=1.0,
                                           capacity=orig.capacity / orig.permlanes,
                                           allowed_modes={"bus", "pt"}, attributes=attrs)
                expected_to_node[newid] = base_to_node[lid]
                bus_replacements[lid] = newid

    # Strict link-by-link comparison; catches both unintended changes and
    # forgotten changes on original links, including other modes and attributes.
    missing_links = set(expected) - set(actual.links)
    extra_links = set(actual.links) - set(expected)
    if missing_links:
        fail(f"Missing expected links: {sorted(missing_links)[:8]}")
    if extra_links:
        fail(f"Unexpected links: {sorted(extra_links)[:8]}")
    for lid in sorted(set(expected) & set(actual.links)):
        e, a = expected[lid], actual.links[lid]
        if (e.from_node, e.to_node) != (a.from_node, a.to_node):
            fail(f"{lid}: from/to differs: expected {e.from_node}->{e.to_node}, got {a.from_node}->{a.to_node}")
        for field in ("length", "capacity", "permlanes", "freespeed"):
            if not _close(getattr(e, field), getattr(a, field)):
                fail(f"{lid}: {field} expected {getattr(e, field):.8g}, got {getattr(a, field):.8g}")
        if set(e.allowed_modes) != set(a.allowed_modes):
            fail(f"{lid}: allowed_modes expected {sorted(e.allowed_modes)}, got {sorted(a.allowed_modes)}")
        if e.attributes != a.attributes:
            differences = [k for k in set(e.attributes) | set(a.attributes)
                           if e.attributes.get(k) != a.attributes.get(k)]
            fail(f"{lid}: custom attributes differ at {sorted(differences)[:6]}")

    # Ensure the mapping includes every generated link in its correct slot.
    mapped = mapping.set_index("link_id", drop=False)
    for lid, nodeid in expected_to_node.items():
        if lid not in mapped.index:
            fail(f"{lid}: absent from link_mapping.csv.gz")
            continue
        m = mapped.loc[lid]
        nd = nodes.iloc[nodeid]
        if (m.segment_id, m.from_node, m.to_node) != (
            str(nd.segment_id), str(nd.from_node), str(nd.to_node)
        ):
            fail(f"{lid}: mapping points at wrong fixed directional slot")
    if set(mapped.index) != set(expected_to_node):
        extra = set(mapped.index) - set(expected_to_node)
        if extra:
            fail(f"Unexpected mapped physical links: {sorted(extra)[:8]}")

    # Reconstruct the expected 9 features from the *policy-derived* network,
    # not the scenario network. This tests the feature extraction itself.
    n = len(nodes)
    base_capacity = np.asarray(nodes.baseline_car_capacity, dtype=np.float64)
    base_speed = np.asarray(nodes.baseline_car_freespeed_m_s, dtype=np.float64)
    base_access = {mode: np.asarray(nodes[f"baseline_allows_{mode}"], dtype=np.float64)
                   for mode in ("car", "bike", "bus")}
    capacity = np.zeros(n, dtype=np.float64)
    speed_sum = np.zeros(n, dtype=np.float64)
    access = {mode: np.zeros(n, dtype=np.float64) for mode in base_access}
    for lid, nodeid in expected_to_node.items():
        if lid not in expected:
            continue
        link = expected[lid]
        for mode in access:
            access[mode][nodeid] = max(access[mode][nodeid], float(mode in link.allowed_modes))
        if "car" in link.allowed_modes:
            capacity[nodeid] += link.capacity
            speed_sum[nodeid] += link.capacity * link.freespeed
    speed = np.divide(speed_sum, capacity, out=np.zeros(n), where=capacity > 0)
    expected_features: dict[str, np.ndarray] = dict(expected_labels)
    expected_features["car_capacity_change_fraction"] = np.divide(
        capacity - base_capacity, base_capacity, out=np.zeros(n), where=base_capacity > 0
    )
    expected_features["car_freespeed_change_fraction"] = np.divide(
        speed - base_speed, base_speed, out=np.zeros(n),
        where=(capacity > 0) & (base_capacity > 0) & (base_speed > 0),
    )
    for mode in access:
        expected_features[f"{mode}_access_change"] = access[mode] - base_access[mode]

    feature_counts = {}
    for name in DEFAULT_DYNAMIC_COLUMNS:
        e = expected_features[name]
        e[np.abs(e) < 1e-12] = 0
        a = features[name].to_numpy(dtype=np.float64)
        differing = np.flatnonzero(~np.isclose(a, e, atol=1e-6, rtol=1e-5))
        feature_counts[name] = int(np.count_nonzero(e))
        if len(differing):
            samples = [(int(i), round(float(e[i]), 5), round(float(a[i]), 5))
                       for i in differing[:5]]
            fail(f"feature {name}: {len(differing)} mismatches (node, expected, actual): {samples}")

    # Ensure scheduled buses actually use dedicated links and not their parents.
    bus_routes_checked = 0
    if bus_replacements:
        schedule = scenario_dir / "transit_schedule.xml.gz"
        if not schedule.exists():
            fail("Bus-lane scenario missing transit_schedule.xml.gz")
        else:
            route_links, stops = _schedule_bus_links(schedule)
            for old, new in bus_replacements.items():
                bus_routes_checked += 1
                if new not in route_links:
                    fail(f"bus lane {new}: not used by a scheduled bus route")
                if old in route_links:
                    fail(f"bus lane {old}: old general link still used by bus route")
                if old in stops:
                    fail(f"bus lane {old}: stop facility still refers to original link")

    if problems:
        text = "\n".join("  - " + p for p in problems[:20])
        extra = f"\n  ... plus {len(problems) - 20} more" if len(problems) > 20 else ""
        raise AssertionError(f"Intervention audit FAIL ({len(problems)} discrepancies):\n{text}{extra}")

    return AuditResult(
        interventions=len(entries), by_type_variant=by_type_variant,
        original_links_checked=len(original.links),
        new_links_checked=len(expected) - len(original.links),
        expected_feature_counts=feature_counts,
        bus_routes_checked=bus_routes_checked,
    )
