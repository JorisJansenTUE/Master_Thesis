from __future__ import annotations
import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

from preprocessing.network.clip_gtfs import clip_gtfs_to_shape, read_gtfs_table
from preprocessing.network.clip_osm import clip_osm_to_shape
from preprocessing.network.common import ensure_clean_dir, setup_logging
from preprocessing.network.config import PipelineConfig, PipelinePaths, make_paths
from preprocessing.network.run_pt2matsim import (
    create_mapped_network_and_schedule,
    create_unmapped_multimodal_network,
    create_unmapped_transit_schedule,
)
from preprocessing.utils import PT2MATSIM_DIR


def parse_key_value_overrides(items: Optional[list[str]]) -> dict[str, str]:
    if not items:
        return {}

    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()

    return overrides


def validate_inputs(config: PipelineConfig) -> None:
    required_files = [
        config.osm_file,
        config.project_shape,
        config.gtfs_zip,
        config.osm_config_template,
        config.mapper_config_template,
    ]

    missing = [path for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(map(str, missing)))

    pom = config.matsim_project_dir / "pom.xml"
    if not pom.exists():
        raise FileNotFoundError(
            f"Could not find pom.xml in MATSim project directory:\n{config.matsim_project_dir}\n"
            f"Check MATSIM_PROJECT_DIR in preprocessing/utils.py."
        )

def create_output_parent_dirs(paths: PipelinePaths) -> None:
    output_files = [
        paths.boundary_geojson,
        paths.clipped_osm,
        paths.clipped_gtfs_zip_copy,
        paths.unmapped_network,
        paths.unmapped_detailed_link_geometry,
        paths.unmapped_schedule,
        paths.unmapped_vehicles,
        paths.mapped_network,
        paths.mapped_schedule,
        paths.osm_config,
        paths.mapper_config,
    ]

    for path in output_files:
        path.parent.mkdir(parents=True, exist_ok=True)

    paths.clipped_gtfs_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

def prepare_output_dirs(config: PipelineConfig) -> None:
    if config.overwrite:
        for path in [config.interim_scenario_dir, config.scenario_dir]:
            if path.exists():
                shutil.rmtree(path)

    ensure_clean_dir(config.interim_scenario_dir, overwrite=False)
    ensure_clean_dir(config.scenario_dir, overwrite=False)

def sanity_check_gtfs(paths: PipelinePaths) -> None:
    stops = read_gtfs_table(paths.clipped_gtfs_dir, "stops.txt", required=True)
    stop_times = read_gtfs_table(paths.clipped_gtfs_dir, "stop_times.txt", required=True)
    trips = read_gtfs_table(paths.clipped_gtfs_dir, "trips.txt", required=True)
    routes = read_gtfs_table(paths.clipped_gtfs_dir, "routes.txt", required=True)

    assert stops is not None
    assert stop_times is not None
    assert trips is not None
    assert routes is not None

    logging.info("GTFS sanity check:")
    logging.info("  stops:      %s", len(stops))
    logging.info("  stop_times: %s", len(stop_times))
    logging.info("  trips:      %s", len(trips))
    logging.info("  routes:     %s", len(routes))

    missing_stop_ids = set(stop_times["stop_id"]) - set(stops["stop_id"])
    missing_trip_ids = set(stop_times["trip_id"]) - set(trips["trip_id"])

    if missing_stop_ids:
        raise ValueError(f"stop_times references missing stop_id values: {list(missing_stop_ids)[:10]}")

    if missing_trip_ids:
        raise ValueError(f"stop_times references missing trip_id values: {list(missing_trip_ids)[:10]}")

    stops_per_trip = stop_times.groupby("trip_id").size()
    if (stops_per_trip < 2).any():
        bad = stops_per_trip[stops_per_trip < 2].index.tolist()[:10]
        raise ValueError(f"Some retained GTFS trips have fewer than two stops: {bad}")

def sanity_check_outputs(paths: PipelinePaths) -> None:
    expected = [
        paths.clipped_osm,
        paths.clipped_gtfs_dir,
        paths.unmapped_network,
        paths.unmapped_schedule,
        paths.unmapped_vehicles,
        paths.mapped_network,
        paths.mapped_schedule,
    ]

    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing expected output files:\n" + "\n".join(map(str, missing)))

    logging.info("All expected output files exist.")


def run_pipeline(config: PipelineConfig) -> PipelinePaths:
    validate_inputs(config)
    prepare_output_dirs(config)
    
    paths = make_paths(config)
    create_output_parent_dirs(make_paths(config))
    setup_logging(paths.logs_dir)

    logging.info("Starting MATSim/PT2MATSim pipeline for scenario: %s", config.scenario_name)
    logging.info("Interim directory:  %s", config.interim_scenario_dir)
    logging.info("Scenario directory: %s", config.scenario_dir)

    clip_osm_to_shape(config, paths)
    create_unmapped_multimodal_network(config, paths)

    clip_gtfs_to_shape(config, paths)
    sanity_check_gtfs(paths)

    create_unmapped_transit_schedule(config, paths)
    create_mapped_network_and_schedule(config, paths)

    sanity_check_outputs(paths)

    logging.info("Pipeline finished successfully.")
    logging.info("Mapped network:  %s", paths.mapped_network)
    logging.info("Mapped schedule: %s", paths.mapped_schedule)

    return paths



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full OSM + GTFS + PT2MATSim network creation pipeline."
    )

    parser.add_argument("--scenario", required=True, help="Scenario name, e.g. locarno")

    parser.add_argument(
        "--osm",
        required=True,
        type=Path,
        help="Input OSM file. Relative paths are resolved from the current shell directory.",
    )
    parser.add_argument(
        "--shape",
        required=True,
        type=Path,
        help="Project area shapefile.",
    )
    parser.add_argument(
        "--gtfs",
        required=True,
        type=Path,
        help="Nationwide GTFS zip file.",
    )

    parser.add_argument("--target-crs", default="EPSG:2056")
    parser.add_argument("--sample-day", default="dayWithMostTrips")
    parser.add_argument("--additional-line-info", default="schedule")
    parser.add_argument("--shape-buffer-degrees", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--java-runner", default="mvn")

    parser.add_argument(
        "--osm-config-template",
        default=PT2MATSIM_DIR / "osm2multimodal_template.xml",
        type=Path,
    )
    parser.add_argument(
        "--mapper-config-template",
        default=PT2MATSIM_DIR / "transit_mapper_template.xml",
        type=Path,
    )

    parser.add_argument(
        "--osm-override",
        action="append",
        help="Override an OsmConverter config param, e.g. --osm-override maxLinkLength=500.0",
    )
    parser.add_argument(
        "--mapper-override",
        action="append",
        help="Override a mapper config param, e.g. --mapper-override maxLinkCandidateDistance=90.0",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    config = PipelineConfig(
        scenario_name=args.scenario,
        osm_file=args.osm.resolve(),
        project_shape=args.shape.resolve(),
        gtfs_zip=args.gtfs.resolve(),
        target_crs=args.target_crs,
        sample_day=args.sample_day,
        additional_line_info=args.additional_line_info,
        overwrite=args.overwrite,
        shape_buffer_degrees=args.shape_buffer_degrees,
        java_runner=args.java_runner,
        osm_config_template=args.osm_config_template.resolve(),
        mapper_config_template=args.mapper_config_template.resolve(),
        osm_config_overrides=parse_key_value_overrides(args.osm_override),
        mapper_config_overrides=parse_key_value_overrides(args.mapper_override),
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()