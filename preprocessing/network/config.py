from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from preprocessing.utils import (
    MATSIM_PROJECT_DIR,
    PT2MATSIM_DIR,
    INTERIM_DIR,
    SCENARIOS_DIR,
)


@dataclass(frozen=True)
class PipelineConfig:
    scenario_name: str

    osm_file: Path
    project_shape: Path
    gtfs_zip: Path

    target_crs: str = "EPSG:2056"
    sample_day: str = "dayWithMostTrips"
    additional_line_info: Optional[str] = "schedule"

    overwrite: bool = False
    shape_buffer_degrees: float = 0.0

    matsim_project_dir: Path = MATSIM_PROJECT_DIR
    java_runner: str = "mvn.cmd"

    osm_config_template: Path = PT2MATSIM_DIR / "osm2multimodal_template.xml"
    mapper_config_template: Path = PT2MATSIM_DIR / "transit_mapper_template.xml"

    osm2multimodal_main: str = "org.matsim.pt2matsim.run.Osm2MultimodalNetwork"
    gtfs2schedule_main: str = "org.matsim.pt2matsim.run.Gtfs2TransitSchedule"
    mapper_main: str = "org.matsim.pt2matsim.run.PublicTransitMapper"

    osm_config_overrides: dict[str, str] = field(default_factory=dict)
    mapper_config_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def interim_scenario_dir(self) -> Path:
        return INTERIM_DIR / self.scenario_name

    @property
    def scenario_dir(self) -> Path:
        return SCENARIOS_DIR / self.scenario_name


@dataclass(frozen=True)
class PipelinePaths:
    boundary_geojson: Path
    clipped_osm: Path
    clipped_gtfs_dir: Path
    clipped_gtfs_zip_copy: Path

    unmapped_network: Path
    unmapped_detailed_link_geometry: Path
    unmapped_schedule: Path
    unmapped_vehicles: Path

    mapped_network: Path
    mapped_schedule: Path

    osm_config: Path
    mapper_config: Path

    logs_dir: Path


def make_paths(config: PipelineConfig) -> PipelinePaths:
    interim = config.interim_scenario_dir
    scenario = config.scenario_dir

    return PipelinePaths(
        boundary_geojson=interim / "boundary" / f"{config.scenario_name}_boundary_wgs84.geojson",
        clipped_osm=interim / "osm" / f"{config.scenario_name}_clipped.osm.gz",
        clipped_gtfs_dir=interim / "gtfs" / f"{config.scenario_name}_clipped_gtfs",
        clipped_gtfs_zip_copy=interim / "gtfs" / f"{config.scenario_name}_clipped_gtfs_debug.zip",

        unmapped_network=scenario / "network" / f"{config.scenario_name}_unmapped_multimodal_network.xml.gz",
        unmapped_detailed_link_geometry=scenario / "network" / f"{config.scenario_name}_detailedLinkGeometry.csv.gz",
        unmapped_schedule=scenario / "schedule" / f"{config.scenario_name}_unmapped_transit_schedule.xml.gz",
        unmapped_vehicles=scenario / "vehicles" / f"{config.scenario_name}_unmapped_vehicles.xml.gz",

        mapped_network=scenario / "network" / f"{config.scenario_name}_mapped_network.xml.gz",
        mapped_schedule=scenario / "schedule" / f"{config.scenario_name}_mapped_transit_schedule.xml.gz",

        osm_config=scenario / "configs_generated" / "osm2multimodal_generated.xml",
        mapper_config=scenario / "configs_generated" / "transit_mapper_generated.xml",

        logs_dir=scenario / "logs",
    )