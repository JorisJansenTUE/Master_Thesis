from dataclasses import dataclass
import math

import geopandas as gpd
import pandas as pd


@dataclass
class InputData:
    statpop: gpd.GeoDataFrame
    statent: gpd.GeoDataFrame
    municipalities: gpd.GeoDataFrame
    od: pd.DataFrame
    modal_splits: pd.DataFrame | None = None

@dataclass
class ActivityLocation:
    x: float
    y: float
    municipality: str

    def distance_to(self, other: "ActivityLocation") -> float:
        return math.hypot(
            other.x - self.x,
            other.y - self.y,
        )


@dataclass
class PersonAttributes:
    age: int
    employed: bool
    has_license: bool
    car_avail: str
    has_bike: bool
    pt_subscription: bool


@dataclass
class SyntheticPerson:
    person_id: str
    home: ActivityLocation
    work: ActivityLocation
    departure_time: str
    work_end_time: str
    initial_mode: str
    attributes: PersonAttributes