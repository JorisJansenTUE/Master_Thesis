from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .config import RunnerConfig
from .sampling import (
    normalize_probabilities,
    sample_categorical,
    weighted_sample_index,
)
from .spatial import sample_point_from_geometry
from .data import PersonAttributes, SyntheticPerson, ActivityLocation, InputData

class LocationSampler:
    """
    Samples home and work locations from STATPOP and STATENT hectare cells.

    Home locations are sampled from STATPOP cells, weighted by population.
    Work locations are sampled from STATENT cells, weighted by employment.
    """

    def __init__(
        self,
        cfg: RunnerConfig,
        statpop: gpd.GeoDataFrame,
        statent: gpd.GeoDataFrame,
        rng: random.Random,
    ) -> None:
        self.cfg = cfg
        self.statpop = statpop
        self.statent = statent
        self.rng = rng
        self.columns = cfg.columns

    def sample_home_location(self, municipality: str) -> ActivityLocation:
        point = self._sample_point_from_cells(
            cells=self.statpop,
            municipality=municipality,
            weight_col=self.columns["statpop_population"],
            location_type="home",
            fallback_to_all=False,
        )

        return ActivityLocation(
            x=point.x,
            y=point.y,
            municipality=str(municipality),
        )

    def sample_work_location(self, municipality: str) -> ActivityLocation:
        point = self._sample_point_from_cells(
            cells=self.statent,
            municipality=municipality,
            weight_col=self.columns["statent_jobs"],
            location_type="work",
            fallback_to_all=True,
        )

        return ActivityLocation(
            x=point.x,
            y=point.y,
            municipality=str(municipality),
        )

    def _sample_point_from_cells(
        self,
        cells: gpd.GeoDataFrame,
        municipality: str,
        weight_col: str,
        location_type: str,
        fallback_to_all: bool,
    ):
        municipality_col = self.columns["municipality_id"]

        candidates = cells[
            cells[municipality_col].astype(str) == str(municipality)
        ].copy()

        if len(candidates) == 0:
            if not fallback_to_all:
                raise ValueError(
                    f"No cells found for {location_type} municipality "
                    f"{municipality}."
                )

            print(
                f"Warning: no cells found for {location_type} municipality "
                f"{municipality}. Falling back to all available cells."
            )

            candidates = cells.copy()

        weights = candidates[weight_col].fillna(0).to_numpy(dtype=float)

        if len(weights) == 0:
            raise ValueError(
                f"No candidate cells available for {location_type} municipality "
                f"{municipality}."
            )

        if np.sum(weights) <= 0:
            print(
                f"Warning: all {location_type} weights are zero for municipality "
                f"{municipality}. Sampling uniformly."
            )

        index = weighted_sample_index(weights, self.rng)
        geometry = candidates.iloc[index].geometry

        return sample_point_from_geometry(geometry, self.rng)

class ModeChoiceInitializer:
    def __init__(
        self,
        cfg: RunnerConfig,
        modal_splits: pd.DataFrame | None,
        rng: random.Random,
    ) -> None:
        self.cfg = cfg
        self.modal_splits = modal_splits
        self.rng = rng
        self.columns = cfg.columns

    def sample_initial_mode(
        self,
        origin_municipality: str,
        distance_m: float,
        has_car: bool | None = None,
        has_bike: bool | None = None,
    ) -> str:
        probabilities = self._distance_based_mode_probabilities(distance_m)

        use_modal_splits = bool(
            self.cfg.raw["modes"]["initial_mode_assignment"].get(
                "use_modal_splits",
                False,
            )
        )

        if use_modal_splits and self.modal_splits is not None:
            probabilities = self._correct_with_modal_split(
                origin_municipality=origin_municipality,
                base_probabilities=probabilities,
            )

        probabilities = self._apply_availability_constraints(
            probabilities=probabilities,
            has_car=has_car,
            has_bike=has_bike,
        )

        return sample_categorical(probabilities, self.rng)

    def _distance_based_mode_probabilities(
        self,
        distance_m: float,
    ) -> dict[str, float]:
        bands = self.cfg.raw["modes"]["initial_mode_assignment"]["distance_bands"]

        for band in bands:
            if distance_m <= float(band["max_distance_m"]):
                return dict(band["probabilities"])

        return dict(bands[-1]["probabilities"])

    def _correct_with_modal_split(
        self,
        origin_municipality: str,
        base_probabilities: dict[str, float],
    ) -> dict[str, float]:
        assert self.modal_splits is not None

        municipality_col = self.columns["modal_split_municipality"]

        subset = self.modal_splits[
            self.modal_splits[municipality_col].astype(str) == str(origin_municipality)
        ]

        if len(subset) == 0:
            return base_probabilities

        row = subset.iloc[0]

        observed = {
            "car": float(row[self.columns["modal_split_car"]]),
            "pt": float(row[self.columns["modal_split_pt"]]),
            "bike": float(row[self.columns["modal_split_bike"]]),
            "walk": float(row[self.columns["modal_split_walk"]]),
        }

        # Accept either fractions, e.g. 0.35, or percentages, e.g. 35.
        if sum(observed.values()) > 1.5:
            observed = {
                mode: value / 100.0
                for mode, value in observed.items()
            }

        alpha = float(
            self.cfg.raw["modes"]["initial_mode_assignment"].get(
                "modal_split_blend_alpha",
                0.5,
            )
        )

        corrected = {
            mode: alpha * base_probabilities.get(mode, 0.0)
            + (1.0 - alpha) * observed.get(mode, 0.0)
            for mode in base_probabilities.keys()
        }

        return normalize_probabilities(corrected)

    @staticmethod
    def _apply_availability_constraints(
        probabilities: dict[str, float],
        has_car: bool | None,
        has_bike: bool | None,
    ) -> dict[str, float]:
        constrained = dict(probabilities)

        if has_car is False and "car" in constrained:
            constrained["car"] = 0.0

        if has_bike is False and "bike" in constrained:
            constrained["bike"] = 0.0

        return normalize_probabilities(constrained)

class PersonAttributeSampler:
    """
    Samples simple synthetic person attributes required by the MATSim/Eqasim population.

    For simplicity:
        - We only sample attributes for commuters, and assume all are employed. 
        - Age is sampled from a truncated normal distribution
        - Other attributes are sampled as independent Bernoulli variables. 
        - The probabilities and parameters can be configured in the input
    """

    def __init__(
        self,
        cfg: RunnerConfig,
        rng: random.Random,
    ) -> None:
        self.cfg = cfg
        self.rng = rng

    def sample(self) -> PersonAttributes:
        attr_cfg = self.cfg.raw["person_attributes"]

        has_license = self.rng.random() < float(attr_cfg["license_probability"])

        has_car = self.rng.random() < float(
            attr_cfg["car_availability_probability"]
        )

        has_bike = self.rng.random() < float(
            attr_cfg["bike_availability_probability"]
        )

        pt_subscription = self.rng.random() < float(
            attr_cfg["pt_subscription_probability"]
        )

        age = self._sample_working_age()

        return PersonAttributes(
            age=age,
            employed=True,
            has_license=has_license,
            car_avail="always" if has_car and has_license else "never",
            has_bike=has_bike,
            pt_subscription=pt_subscription,
        )

    def _sample_working_age(self) -> int:
        age_mean = float(
            self.cfg.raw["person_attributes"].get(
                "working_age_mean",
                42,
            )
        )

        age_std = float(
            self.cfg.raw["person_attributes"].get(
                "working_age_std",
                12,
            )
        )

        min_age = int(
            self.cfg.raw["person_attributes"].get(
                "working_age_min",
                18,
            )
        )

        max_age = int(
            self.cfg.raw["person_attributes"].get(
                "working_age_max",
                67,
            )
        )

        age = int(round(self.rng.gauss(age_mean, age_std)))
        return min(max(age, min_age), max_age)

class PopulationSynthesizer:
    """
    Creates a synthetic MATSim population from:
    - commuter OD matrix,
    - STATPOP hectare population grid,
    - STATENT hectare employment grid,
    - optional municipal modal splits.

    The output is a list of SyntheticPerson objects with home-work-home plans.
    """

    def __init__(
        self,
        cfg: RunnerConfig,
        input_data: InputData,
    ) -> None:
        self.cfg = cfg
        self.input_data = input_data
        self.columns = cfg.columns
        self.rng = random.Random(cfg.random_seed)

        self.location_sampler = LocationSampler(
            cfg=cfg,
            statpop=input_data.statpop,
            statent=input_data.statent,
            rng=self.rng,
        )

        self.attribute_sampler = PersonAttributeSampler(
            cfg=cfg,
            rng=self.rng,
        )

        self.mode_initializer = ModeChoiceInitializer(
            cfg=cfg,
            modal_splits=input_data.modal_splits,
            rng=self.rng,
        )

    def synthesize(self) -> list[SyntheticPerson]:
        persons: list[SyntheticPerson] = []
        person_counter = 0

        od_origin_col = self.columns["od_origin"]
        od_destination_col = self.columns["od_destination"]
        od_flow_col = self.columns["od_flow"]

        include_intrazonal = bool(
            self.cfg.raw["demand"].get(
                "include_intrazonal",
                True,
            )
        )

        min_commuters = int(
            self.cfg.raw["demand"].get(
                "min_commuters_per_od_after_sampling",
                0,
            )
        )

        for od_row in self.input_data.od.itertuples(index=False):
            origin = str(getattr(od_row, od_origin_col))
            destination = str(getattr(od_row, od_destination_col))
            flow = float(getattr(od_row, od_flow_col))

            if origin == destination and not include_intrazonal:
                continue

            sampled_flow = self._sample_od_flow(
                raw_flow=flow,
                min_commuters=min_commuters,
            )

            if sampled_flow <= 0:
                continue

            for _ in range(sampled_flow):
                person = self._create_commuter(
                    person_id=f"person_{person_counter:08d}",
                    origin=origin,
                    destination=destination,
                )

                persons.append(person)
                person_counter += 1

        return persons

    def _sample_od_flow(
        self,
        raw_flow: float,
        min_commuters: int,
    ) -> int:
        if raw_flow <= 0:
            return 0

        sampled_flow = int(round(raw_flow * self.cfg.sample_fraction))

        if min_commuters > 0:
            sampled_flow = max(min_commuters, sampled_flow)

        return sampled_flow

    def _create_commuter(
        self,
        person_id: str,
        origin: str,
        destination: str,
    ) -> SyntheticPerson:
        home = self.location_sampler.sample_home_location(origin)
        work = self.location_sampler.sample_work_location(destination)

        attributes = self.attribute_sampler.sample()

        distance_m = home.distance_to(work)

        initial_mode = self.mode_initializer.sample_initial_mode(
            origin_municipality=origin,
            distance_m=distance_m,
            has_car=attributes.car_avail == "always",
            has_bike=attributes.has_bike,
        )

        departure_s = self._sample_departure_time()
        work_end_s = self._sample_work_end_time(departure_s)

        return SyntheticPerson(
            person_id=person_id,
            home=home,
            work=work,
            departure_time=seconds_to_hhmmss(departure_s),
            work_end_time=seconds_to_hhmmss(work_end_s),
            initial_mode=initial_mode,
            attributes=attributes,
        )

    def _sample_departure_time(self) -> int:
        time_cfg = self.cfg.raw["time"]

        mean_s = parse_hhmmss(time_cfg["morning_departure_mean"])
        std_s = float(time_cfg["morning_departure_std_minutes"]) * 60.0

        departure_s = self.rng.gauss(mean_s, std_s)

        earliest_s = parse_hhmmss(
            time_cfg.get(
                "morning_departure_earliest",
                "05:00:00",
            )
        )

        latest_s = parse_hhmmss(
            time_cfg.get(
                "morning_departure_latest",
                "10:30:00",
            )
        )

        return int(
            min(
                max(departure_s, earliest_s),
                latest_s,
            )
        )

    def _sample_work_end_time(
        self,
        departure_s: int,
    ) -> int:
        time_cfg = self.cfg.raw["time"]

        mean_duration_s = float(time_cfg["work_duration_mean_hours"]) * 3600.0
        std_s = float(time_cfg["work_duration_std_minutes"]) * 60.0
        duration_s = self.rng.gauss(mean_duration_s, std_s)

        min_duration_s = float(time_cfg.get("work_duration_min_hours",5.5)) * 3600.0

        max_duration_s = float(time_cfg.get("work_duration_max_hours",10.5)) * 3600.0

        duration_s = min(max(duration_s, min_duration_s),max_duration_s)

        return int(departure_s + duration_s)
       

def parse_hhmmss(value: str) -> int:
    h, m, s = [int(part) for part in value.split(":")]
    return h * 3600 + m * 60 + s


def seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(round(seconds))
    seconds = max(0, seconds)

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:02d}"