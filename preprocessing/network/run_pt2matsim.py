from __future__ import annotations

from pathlib import Path
from typing import Optional

from preprocessing.network.common import require_executable, run_command
from preprocessing.network.config import PipelineConfig, PipelinePaths
from preprocessing.network.config_utils import generate_mapper_config, generate_osm_config

def run_maven_java(
    main_class: str,
    args: list[str | Path],
    java_runner: str,
    cwd: Optional[Path],
) -> None:
    require_executable(java_runner)

    exec_args = " ".join(str(arg) for arg in args if str(arg) != "")

    command = [
        java_runner,
        "exec:java",
        f"-Dexec.mainClass={main_class}",
        f"-Dexec.args={exec_args}",
    ]

    run_command(command, cwd=cwd)


def create_unmapped_multimodal_network(
    config: PipelineConfig,
    paths: PipelinePaths,
) -> None:
    generate_osm_config(config, paths)

    run_maven_java(
        main_class=config.osm2multimodal_main,
        args=[paths.osm_config],
        java_runner=config.java_runner,
        cwd=config.matsim_project_dir,
    )


def create_unmapped_transit_schedule(
    config: PipelineConfig,
    paths: PipelinePaths,
) -> None:
    """
    Run Gtfs2TransitSchedule with positional arguments.

    Argument order:
    [0] GTFS folder
    [1] sample day: yyyymmdd, dayWithMostTrips, dayWithMostServices, or all
    [2] output coordinate system
    [3] output transit schedule file
    [4] optional output vehicles file
    [5] optional additional line info: empty, schedule, or .csv path
    """
    args: list[str | Path] = [
        paths.clipped_gtfs_dir,
        config.sample_day,
        config.target_crs,
        paths.unmapped_schedule,
        paths.unmapped_vehicles,
    ]

    if config.additional_line_info:
        args.append(config.additional_line_info)

    run_maven_java(
        main_class=config.gtfs2schedule_main,
        args=args,
        java_runner=config.java_runner,
        cwd=config.matsim_project_dir,
    )


def create_mapped_network_and_schedule(
    config: PipelineConfig,
    paths: PipelinePaths,
) -> None:
    generate_mapper_config(config, paths)

    run_maven_java(
        main_class=config.mapper_main,
        args=[paths.mapper_config],
        java_runner=config.java_runner,
        cwd=config.matsim_project_dir,
    )