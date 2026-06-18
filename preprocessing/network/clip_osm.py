from __future__ import annotations

from pathlib import Path
from typing import Mapping

import osmium
import logging

from preprocessing.network.common import create_parent_dirs, require_executable, run_command
from preprocessing.network.config import PipelineConfig, PipelinePaths
from preprocessing.network.geometry import write_boundary_geojson

BIKE_ACCESS_VALUES = frozenset({"yes", "designated"})
BIKE_SUITED_HIGHWAY_VALUE = "bike_suited"
ORIGINAL_HIGHWAY_TAG = "original_highway"

def clip_osm_to_shape(config: PipelineConfig, paths: PipelinePaths) -> None:
    require_executable("osmium")
    create_parent_dirs([paths.boundary_geojson, paths.clipped_osm])

    write_boundary_geojson(
        shape_path=config.project_shape,
        output_geojson=paths.boundary_geojson,
        buffer_degrees=config.shape_buffer_degrees,
    )

    run_command(
        [
            "osmium",
            "extract",
            "--polygon",
            paths.boundary_geojson,
            "--strategy",
            "smart",
            "--overwrite",
            "-o",
            paths.clipped_osm,
            config.osm_file,
        ]
    )

def contains_golf_tag(tags: Mapping[str, str]) -> bool:
    """Return whether any tag key or value contains 'golf'."""
    return any(
        "golf" in key.lower() or "golf" in value.lower()
        for key, value in tags.items()
    )


def is_bike_suited_track(tags: Mapping[str, str]) -> bool:
    """
    Select highway=track ways when either:

    - tracktype=grade1; or
    - bicycle=yes/designated.

    Ways containing a golf-related key or value are excluded.
    """
    if tags.get("highway") != "track":
        return False

    if contains_golf_tag(tags):
        return False

    return (
        tags.get("tracktype") == "grade1"
        or tags.get("bicycle") in BIKE_ACCESS_VALUES
    )


def is_bike_suited_path(tags: Mapping[str, str]) -> bool:
    """
    Select highway=path ways explicitly designated for bicycles.

    Golf-related paths are excluded as well.
    """
    return (
        tags.get("highway") == "path"
        and tags.get("bicycle") == "designated"
        and not contains_golf_tag(tags)
    )


def is_bike_suited_way(tags: Mapping[str, str]) -> bool:
    """Return whether an OSM way should be reclassified."""
    return (
        is_bike_suited_track(tags)
        or is_bike_suited_path(tags)
    )


class BikeSuitableWayWriter(osmium.SimpleHandler):
    """Copy an OSM dataset while reclassifying suitable bike ways."""

    def __init__(self, output_path: Path) -> None:
        super().__init__()

        self.writer = osmium.SimpleWriter(str(output_path))

        self.changed_tracks = 0
        self.changed_paths = 0
        self.changed_way_ids: list[int] = []

    def node(self, node: osmium.osm.Node) -> None:
        self.writer.add_node(node)

    def way(self, way: osmium.osm.Way) -> None:
        tags = dict(way.tags)

        if not is_bike_suited_way(tags):
            self.writer.add_way(way)
            return

        original_highway = tags["highway"]

        tags[ORIGINAL_HIGHWAY_TAG] = original_highway
        tags["highway"] = BIKE_SUITED_HIGHWAY_VALUE

        modified_way = way.replace(tags=tags)
        self.writer.add_way(modified_way)

        self.changed_way_ids.append(way.id)

        if original_highway == "track":
            self.changed_tracks += 1
        elif original_highway == "path":
            self.changed_paths += 1

    def relation(self, relation: osmium.osm.Relation) -> None:
        self.writer.add_relation(relation)

    def close(self) -> None:
        self.writer.close()


def classify_bike_suitable_ways(
    config: PipelineConfig,
    paths: PipelinePaths,
) -> None:
    """
    Reclassify selected tracks and paths as highway=bike_suited.

    Input:
        paths.clipped_osm

    Output:
        paths.classified_osm
    """
    del config  # Reserved for future classification settings.

    input_path = Path(paths.clipped_osm)
    output_path = Path(paths.processed_tags_osm)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Clipped OSM input does not exist: {input_path}"
        )

    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "The clipped and classified OSM paths must be different."
        )

    create_parent_dirs([output_path])

    # SimpleWriter refuses to overwrite an existing file.
    if output_path.exists():
        output_path.unlink()

    handler = BikeSuitableWayWriter(output_path)

    try:
        handler.apply_file(
            str(input_path),
            locations=False,
        )
    finally:
        handler.close()

    logging.info(
        "Bike-suitable OSM reclassification completed: "
        "%s tracks, %s paths, %s total ways changed. "
        "Input=%s, output=%s",
        f"{handler.changed_tracks:,}",
        f"{handler.changed_paths:,}",
        f"{len(handler.changed_way_ids):,}",
        input_path,
        output_path,
    )
    if handler.changed_way_ids:
        logging.debug(
            "Example changed way IDs: %s",
            handler.changed_way_ids[:20],
        )