from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InterventionType(Enum):
    PROTECTED_BIKE_LANE = "protected_bike_lane"
    BICYCLE_STREET = "bicycle_street"
    DESTINATION_ONLY = "destination_only"
    DEDICATED_BUS_LANE = "dedicated_bus_lane"
    PT_BIKE_STREET = "pt_bike_street"


class RoadSpaceVariant(Enum):
    KEEP_CAR = "keep_car"
    REMOVE_CAR = "remove_car"
    REMOVE_ONE_LANE = "remove_one_lane"


@dataclass(frozen=True)
class Intervention:
    """
    Intervention that can be applied to a physical street segment.

    Parameters
    ----------
    segment_id
        ID of the physical StreetSegment.

    intervention_type
        Type of intervention.

    intervention_direction
        Direction in which buslane is applied

    road_space_variant
        Change to car road space required to create the intervention. This is for PBL and Buslane interventions only as this often involves removing a lane for cars

    target_link_ids
        Directional MATSim car links affected by the intervention:
        - DEDICATED_BUS_LANE → direction in which bus lane is created AND road_space_variant is applied;
        - PBL: REMOVE_ONE_LANE → directions losing a lane;
        - PBL: CLOSE_CAR → directions losing car access;
        - non-directional intervention → None.
   
    """

    segment_id: str
    intervention_type: InterventionType
    road_space_variant: RoadSpaceVariant | None = None
    target_link_ids: tuple[str, ...] = ()