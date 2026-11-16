"""Preserve physical street identities for later training-target construction.

No GNN tensors, feature encodings or dual graphs are created here.

A scenario's link_mapping.csv.gz maps the directed MATSim links (including
intervention-created links) back to the original undirected StreetSegment and
actual travel direction. A single street_segments.csv.gz records the exact
baseline StreetSegmentIndex used by the generator.


TO DO: Construct link_mapping directly when applying interventions, rather than re-reading the MATSim network. -> This will be slightly faster and is more robust to MATSim network changes
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import tempfile
from pathlib import Path

from network_sampling.network.street_segments import StreetSegmentIndex

MAPPING_COLUMNS = (
    "link_id", "segment_id", "from_node", "to_node", "baseline_link_id", "link_kind"
)
SEGMENT_COLUMNS = (
    "segment_id", "node_a", "node_b", "length_m", "total_width_m",
    "osm_way_id", "baseline_link_ids"
)


def _atomic_gzip_csv(path: Path, columns, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".csv.gz", dir=path.parent)
    os.close(fd)
    try:
        # Standard gzip CSV; no special GNN indexing assumptions.
        with gzip.open(tmp, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _segment_lookup(index: StreetSegmentIndex):
    segments = {str(seg.id): seg for seg in index.all()}
    original_to_segment: dict[str, str] = {}
    for sid, seg in segments.items():
        for link_id in seg.link_ids:
            lid = str(link_id)
            other = original_to_segment.setdefault(lid, sid)
            if other != sid:
                raise ValueError(f"Baseline link {lid} appears in multiple street segments")
    return segments, original_to_segment


def write_street_segments(destination: Path, index: StreetSegmentIndex) -> int:
    """Export stable physical street IDs and their original MATSim link IDs.

    This is a provenance snapshot, not a GNN input graph. It is created once.
    """
    segments, _ = _segment_lookup(index)
    rows = []
    for sid, seg in sorted(segments.items()):
        rows.append({
            "segment_id": sid,
            "node_a": str(seg.node_a),
            "node_b": str(seg.node_b),
            "length_m": float(seg.length_m),
            "total_width_m": "" if getattr(seg, "total_width_m", None) is None else seg.total_width_m,
            "osm_way_id": "" if getattr(seg, "osm_way_id", None) is None else seg.osm_way_id,
            "baseline_link_ids": json.dumps([str(link_id) for link_id in seg.link_ids]),
        })
    _atomic_gzip_csv(destination, SEGMENT_COLUMNS, rows)
    return len(rows)


def write_link_mapping(
    destination: Path,
    index: StreetSegmentIndex,
    baseline,
    modified,
) -> int:
    """Map every road MATSim link to its baseline physical segment + direction.

    Recognized intervention-created link IDs:
      <original_link_id>_spec       dedicated bus link
      <segment_id>__pbl__...       protected bicycle link
      <segment_id>__bike__...      bicycle-street reverse-direction link

    Unknown *new* links raise an error. Baseline PT/non-road links not in the
    StreetSegmentIndex are intentionally omitted (their raw flows are retained).
    """
    segments, original_to_segment = _segment_lookup(index)
    baseline_ids = {str(lid) for lid in baseline.links}
    rows = []
    seen: set[str] = set()
    for raw_id, link in sorted(modified.links.items(), key=lambda pair: str(pair[0])):
        lid = str(raw_id)
        if lid in seen:
            raise ValueError(f"Duplicate MATSim link: {lid}")
        seen.add(lid)
        parent = ""
        kind = "baseline"
        if lid in original_to_segment:
            sid = original_to_segment[lid]
            parent = lid
        elif lid in baseline_ids:
            # Original PT/rail links excluded from physical street index.
            continue
        elif lid.endswith("_spec") and lid[:-5] in original_to_segment:
            parent = lid[:-5]
            sid = original_to_segment[parent]
            kind = "dedicated_bus_lane"
        else:
            sid = None
            for marker, generated_kind in (
                ("__pbl__", "protected_bike_lane"),
                ("__bike__", "bicycle_street_reverse"),
            ):
                if marker in lid:
                    prefix = lid.split(marker, 1)[0]
                    if prefix in segments:
                        sid = prefix
                        kind = generated_kind
                        break
            if sid is None:
                raise ValueError(
                    f"New MATSim link {lid!r} has an unknown naming convention. "
                    "Check interventions/matsim_app/apply_network.py before training."
                )
        seg = segments[sid]
        u, v = str(link.from_node), str(link.to_node)
        if (u, v) not in {
            (str(seg.node_a), str(seg.node_b)),
            (str(seg.node_b), str(seg.node_a)),
        }:
            raise ValueError(
                f"Link {lid!r}: direction ({u}, {v}) does not match "
                f"segment {sid!r}: ({seg.node_a}, {seg.node_b})"
            )
        rows.append({
            "link_id": lid, "segment_id": sid,
            "from_node": u, "to_node": v,
            "baseline_link_id": parent, "link_kind": kind,
        })
    if not rows:
        raise ValueError("No road links mapped; check the baseline StreetSegmentIndex")
    _atomic_gzip_csv(destination, MAPPING_COLUMNS, rows)
    return len(rows)
