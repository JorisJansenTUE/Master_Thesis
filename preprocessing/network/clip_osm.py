from __future__ import annotations

from preprocessing.network.common import create_parent_dirs, require_executable, run_command
from preprocessing.network.config import PipelineConfig, PipelinePaths
from preprocessing.network.geometry import write_boundary_geojson


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