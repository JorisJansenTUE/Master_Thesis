"""Numerical accessibility for a completed MATSim scenario (no plots or writes).

This is the shared routing entry point for the existing accessibility inspection
and the genetic-algorithm fitness calculation. All intervention changes are
already present in the scenario network and schedule.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from accessibility.calculations.cumulative import CumulativeAccessibility
from accessibility.inputs.grid.snapping import ScenarioRoadSnapper
from accessibility.inputs.links.linkstats import LinkTravelTimes
from accessibility.inputs.links.network import PTNetwork, load_road_networks
from accessibility.inputs.links.schedule import TransitSchedule
from accessibility.routing.pt import PTRouter
from accessibility.routing.road import RoadRouter


def calculate_accessibility(
    scenario_dir: str | Path,
    spatial_dir: str | Path,
    *,
    thresholds_min: tuple[float, ...] = (15.0, 20.0, 30.0),
    hour_start: int = 8,
    linkstats_statistic: str = "avg",
    bike_speed_m_s: float = 4.17,
    pt_transfer_penalty_s: float = 300.0,
    origin_snap_m: float = 500.0,
    job_snap_m: float = 300.0,
) -> pd.DataFrame:
    """Return per-origin car/bike/PT opportunities using the actual scenario.

    Car and bike attachments are rebuilt for each modified network. PT walking
    connectors are the precomputed fixed inputs. The existing network loaders
    enforce no car on *_spec and use free-flow cost for bus *_spec links.
    """
    scenario_dir = Path(scenario_dir)
    spatial_dir = Path(spatial_dir)
    origins = pd.read_csv(spatial_dir / "origins.csv", dtype={"hex_id": str})
    jobs = pd.read_csv(spatial_dir / "jobs.csv", dtype={"job_id": str})
    origin_ids = origins["hex_id"].astype(str).tolist()

    linkstats = LinkTravelTimes.from_linkstats(
        scenario_dir / "targets" / "linkstats.txt.gz",
        hour_start=hour_start,
        statistic=linkstats_statistic,
    )
    car, bike = load_road_networks(
        scenario_dir / "network.xml.gz", linkstats, bike_speed_m_s=bike_speed_m_s,
    )
    snapper = ScenarioRoadSnapper(
        origins, jobs,
        max_origin_distance_m=origin_snap_m,
        max_job_distance_m=job_snap_m,
    )
    car_att = snapper.attach(car.graph)
    bike_att = snapper.attach(bike.graph)
    car_router = RoadRouter(car.graph, car_att.origins.connectors, car_att.jobs.connectors, jobs)
    bike_router = RoadRouter(bike.graph, bike_att.origins.connectors, bike_att.jobs.connectors, jobs)

    schedule = TransitSchedule.from_matsim_schedule(scenario_dir / "transit_schedule.xml.gz")
    pt_network = PTNetwork.from_schedule(
        schedule, linkstats, transfer_penalty_s=pt_transfer_penalty_s,
    )
    pt_origin = pd.read_csv(
        spatial_dir / "pt_origin_connectors.csv", dtype={"point_id": str, "stop_id": str},
    )
    pt_jobs = pd.read_csv(
        spatial_dir / "pt_job_connectors.csv", dtype={"point_id": str, "stop_id": str},
    )
    pt_router = PTRouter(pt_network, pt_origin, pt_jobs, jobs)

    cumulative = CumulativeAccessibility(thresholds_min)
    return pd.concat([
        cumulative.run(car_router, origin_ids, mode="car"),
        cumulative.run(bike_router, origin_ids, mode="bike"),
        cumulative.run(pt_router, origin_ids, mode="pt"),
    ], ignore_index=True)
