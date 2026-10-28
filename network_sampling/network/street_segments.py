from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from network_sampling.network.matsim_network import MatsimLink, MatsimNetwork


DEFAULT_OSM_ID_ATTRIBUTES = (
    "osm:way:id",
    "osm_way_id",
    "osm_id",
    "osmId",
)

DEFAULT_ROAD_MODES = frozenset(
    {
        "car",
        "car_passenger",
        "truck",
        "bike",
        "bus",
    }
)


@dataclass(frozen=True)
class StreetSegment:
    """ 
    One physical street segment represented by one or more MATSim links.
    StreetSegments are used as the physical planning unit for intervention
    sampling, budgets, corridors and clusters.
    """

    id: str
    node_a: str
    node_b: str
    link_ids: tuple[str, ...]
    length_m: float
    total_width_m: float | None
    osm_way_id: str | None


class StreetSegmentIndex:
    """
    Maps baseline MATSim links to physical street segments.
    The index converts the directed MATSim road network into a stable set
    of physical street segments. Opposite directional links between the
    same MATSim nodes are grouped together, unless they belong to distinct
    OSM ways.
    """

    def __init__(
        self,
        segments: dict[str, StreetSegment],
        link_to_segment: dict[str, str],
    ):
        self._segments = segments
        self._link_to_segment = link_to_segment

    @classmethod
    def from_network(
        cls,
        network: MatsimNetwork,
        *,
        osm_id_attributes: tuple[str, ...] = DEFAULT_OSM_ID_ATTRIBUTES,
        road_modes: frozenset[str] = DEFAULT_ROAD_MODES,
        width_attribute: str = "width",
    ) -> "StreetSegmentIndex":
        """ 
        Construct physical street segments from a MATSim network.

        Parameters
        ----------
        network
            Baseline MATSim network.

        osm_id_attributes
            Possible MATSim link attribute names containing the OSM way ID.

        road_modes
            Modes used to determine whether a MATSim link belongs to the
            road network.

        width_attribute
            MATSim link attribute containing the directional road width.
        
        """

        node_pair_groups: dict[
            tuple[str, str],
            list[MatsimLink],
        ] = {}

        for link in network.links.values():
            if not (link.allowed_modes & road_modes):
                continue

            node_pair = _canonical_node_pair(
                link.from_node,
                link.to_node,
            )

            node_pair_groups.setdefault(
                node_pair,
                [],
            ).append(link)

        segments: dict[str, StreetSegment] = {}
        link_to_segment: dict[str, str] = {}

        for node_pair, links in node_pair_groups.items():
            parallel_groups = _split_parallel_ways(
                links,
                osm_id_attributes,
            )

            multiple_groups = len(parallel_groups) > 1

            for osm_way_id, grouped_links in parallel_groups:
                segment_id = _make_segment_id(
                    node_pair,
                    osm_way_id,
                    multiple_groups,
                )

                segment = StreetSegment(
                    id=segment_id,
                    node_a=node_pair[0],
                    node_b=node_pair[1],
                    link_ids=tuple(
                        sorted(
                            link.id
                            for link in grouped_links
                        )
                    ),
                    length_m=median(
                        link.length
                        for link in grouped_links
                    ),
                    total_width_m=_get_total_width(
                        grouped_links,
                        width_attribute,
                    ),
                    osm_way_id=osm_way_id,
                )

                if segment_id in segments:
                    raise ValueError(
                        f"Duplicate street segment ID: {segment_id}"
                    )

                segments[segment_id] = segment

                for link in grouped_links:
                    if link.id in link_to_segment:
                        raise ValueError(
                            f"MATSim link {link.id} belongs to "
                            f"multiple street segments."
                        )

                    link_to_segment[link.id] = segment_id

        return cls(
            segments=segments,
            link_to_segment=link_to_segment,
        )

    def get(
        self,
        segment_id: str,
    ) -> StreetSegment:
        return self._segments[segment_id]

    def get_for_link(
        self,
        link_id: str,
    ) -> StreetSegment:
        return self._segments[
            self._link_to_segment[link_id]
        ]

    def all(
        self,
    ) -> tuple[StreetSegment, ...]:
        return tuple(
            self._segments.values()
        )

    def __len__(
        self,
    ) -> int:
        return len(self._segments)

    @property
    def link_to_segment(
        self,
    ) -> dict[str, str]:
        return dict(
            self._link_to_segment
        )


def _canonical_node_pair(
    from_node: str,
    to_node: str,
) -> tuple[str, str]:

    if from_node <= to_node:
        return from_node, to_node

    return to_node, from_node


def _get_osm_way_id(
    link: MatsimLink,
    osm_id_attributes: tuple[str, ...],
) -> str | None:

    for attribute_name in osm_id_attributes:
        value = link.attributes.get(
            attribute_name
        )

        if value is not None:
            return str(value)

    return None


def _split_parallel_ways(
    links: list[MatsimLink],
    osm_id_attributes: tuple[str, ...],
) -> list[
    tuple[
        str | None,
        list[MatsimLink],
    ]
]:

    known_osm_ids = {
        osm_way_id
        for link in links
        if (
            osm_way_id := _get_osm_way_id(
                link,
                osm_id_attributes,
            )
        )
        is not None
    }

    if len(known_osm_ids) <= 1:
        return [
            (
                next(
                    iter(known_osm_ids),
                    None,
                ),
                links,
            )
        ]

    groups: dict[
        str | None,
        list[MatsimLink],
    ] = {}

    for link in links:
        osm_way_id = _get_osm_way_id(
            link,
            osm_id_attributes,
        )

        groups.setdefault(
            osm_way_id,
            [],
        ).append(link)

    return list(
        groups.items()
    )


def _get_total_width(
    links: list[MatsimLink],
    width_attribute: str,
) -> float | None:

    if not links:
        return None

    widths: list[float] = []

    for link in links:
        value = link.attributes.get(
            width_attribute
        )

        if value is None:
            return None

        try:
            width = float(value)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid width '{value}' on MATSim link {link.id}."
            ) from exc

        if width < 0:
            raise ValueError(
                f"Negative width {width} on MATSim link {link.id}."
            )

        widths.append(width)

    return sum(widths)


def _make_segment_id(
    node_pair: tuple[str, str],
    osm_way_id: str | None,
    multiple_groups: bool,
) -> str:

    base_id = (
        f"{node_pair[0]}__{node_pair[1]}"
    )

    if not multiple_groups:
        return base_id

    suffix = (
        osm_way_id
        if osm_way_id is not None
        else "unknown"
    )

    return (
        f"{base_id}__osm_{suffix}"
    )