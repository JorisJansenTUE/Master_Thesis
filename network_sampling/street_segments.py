from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from network_sampling.matsim_network import (
    MatsimLink,
    MatsimNetwork,
)


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
    Canonical representation of one physical street segment.

    A physical street segment can contain:
      - one MATSim link, for example a one-way street;
      - two MATSim links representing opposite travel directions;
      - in exceptional cases, more links if several MATSim links belong
        to the same physical street section.

    The StreetSegment is intended to be the unit used for:
      - intervention sampling;
      - intervention budgets;
      - corridor and cluster generation;
      - later GNN representation.

    MATSim links remain the simulation-level representation.
    """

    id: str

    node_a: str
    node_b: str

    link_ids: tuple[str, ...]

    # Physical street length.
    #
    # Opposite travel directions describe the same physical street
    # length, so this is not the sum of MATSim link lengths.
    length_m: float

    # Total physical carriageway width.
    #
    # In the current MATSim network, directional link widths represent
    # portions of the total carriageway width. Therefore, directional
    # widths are summed.
    #
    # None means that width information was not available for every
    # MATSim link belonging to the street segment.
    total_width_m: float | None

    osm_way_id: str | None


    @property
    def number_of_links(self) -> int:
        return len(self.link_ids)


class StreetSegmentIndex:
    """
    Index mapping baseline MATSim links onto physical street segments.

    The index is constructed from the baseline MATSim network and should
    remain fixed across generated intervention scenarios.

    This is important because generated intervention links, such as
    parallel protected bicycle links, should not later become new
    sampling units.
    """

    def __init__(
        self,
        segments: dict[str, StreetSegment],
        link_to_segment: dict[str, str],
        segment_directions: dict[
            str,
            tuple[tuple[str, str], ...],
        ],
    ):
        self._segments = segments
        self._link_to_segment = link_to_segment
        self._segment_directions = segment_directions

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
            Candidate MATSim link attribute names containing the original
            OSM way ID.

        road_modes
            Modes used to identify links that belong to the road network.

        width_attribute
            MATSim link attribute containing the directional physical
            width in metres.

        Returns
        -------
        StreetSegmentIndex
        """

        # -------------------------------------------------------------
        # Step 1:
        # Group road links by unordered MATSim node pair.
        #
        # A -> B and B -> A therefore initially belong to the same
        # physical street candidate.
        # -------------------------------------------------------------

        node_pair_groups: dict[
            tuple[str, str],
            list[MatsimLink],
        ] = {}

        for link in network.links.values():

            # Ignore network links unrelated to the road network.
            #
            # This prevents, for example, rail-only links from becoming
            # intervention candidates.
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

        segment_directions: dict[
            str,
            tuple[tuple[str, str], ...],
        ] = {}

        # -------------------------------------------------------------
        # Step 2:
        # Within each node pair, distinguish different OSM ways where
        # necessary.
        #
        # Normally:
        #
        #     A -> B
        #     B -> A
        #
        # belong to the same physical road.
        #
        # But if multiple distinct OSM ways occur between identical
        # MATSim nodes, they should not automatically be merged.
        # -------------------------------------------------------------

        for node_pair, links in node_pair_groups.items():

            parallel_groups = _split_parallel_ways(
                links=links,
                osm_id_attributes=osm_id_attributes,
            )

            multiple_groups = len(parallel_groups) > 1

            for osm_way_id, grouped_links in parallel_groups:

                segment_id = _make_segment_id(
                    node_pair=node_pair,
                    osm_way_id=osm_way_id,
                    multiple_groups=multiple_groups,
                )

                # -----------------------------------------------------
                # Physical length
                # -----------------------------------------------------
                #
                # Both travel directions describe the same physical road
                # length, so their lengths must not be summed.
                #
                # Median is used to tolerate small numerical differences
                # between directional MATSim links.
                # -----------------------------------------------------

                lengths = [
                    link.length
                    for link in grouped_links
                ]

                length_m = median(lengths)

                # -----------------------------------------------------
                # Physical total width
                # -----------------------------------------------------
                #
                # In the current network construction, each directional
                # MATSim link contains the share of total carriageway
                # width belonging to that direction.
                #
                # Example:
                #
                #   total OSM width = 9 m
                #   A -> B = 6 m
                #   B -> A = 3 m
                #
                # StreetSegment total width = 9 m.
                #
                # If any directional width is missing, return None rather
                # than silently underestimating the total road width.
                # -----------------------------------------------------

                total_width_m = _get_total_width(
                    links=grouped_links,
                    width_attribute=width_attribute,
                )

                link_ids = tuple(
                    sorted(
                        link.id
                        for link in grouped_links
                    )
                )

                segment = StreetSegment(
                    id=segment_id,
                    node_a=node_pair[0],
                    node_b=node_pair[1],
                    link_ids=link_ids,
                    length_m=length_m,
                    total_width_m=total_width_m,
                    osm_way_id=osm_way_id,
                )

                if segment_id in segments:
                    raise ValueError(
                        f"Duplicate street segment ID: {segment_id}"
                    )

                segments[segment_id] = segment

                segment_directions[segment_id] = tuple(
                    (
                        link.from_node,
                        link.to_node,
                    )
                    for link in grouped_links
                )

                # -----------------------------------------------------
                # Reverse lookup:
                #
                # MATSim link ID -> physical street segment ID
                # -----------------------------------------------------

                for link in grouped_links:

                    if link.id in link_to_segment:
                        raise ValueError(
                            f"MATSim link {link.id} was assigned to "
                            f"multiple street segments."
                        )

                    link_to_segment[link.id] = segment_id

        return cls(
            segments=segments,
            link_to_segment=link_to_segment,
            segment_directions=segment_directions,
        )

    # -----------------------------------------------------------------
    # Access methods
    # -----------------------------------------------------------------

    def get(
        self,
        segment_id: str,
    ) -> StreetSegment:
        """
        Retrieve a street segment by its segment ID.
        """

        try:
            return self._segments[segment_id]

        except KeyError as exc:
            raise KeyError(
                f"Unknown street segment: {segment_id}"
            ) from exc

    def get_for_link(
        self,
        link_id: str,
    ) -> StreetSegment:
        """
        Retrieve the physical street segment associated with a MATSim
        link.
        """

        try:
            segment_id = self._link_to_segment[link_id]

        except KeyError as exc:
            raise KeyError(
                f"MATSim link is not part of a street segment: "
                f"{link_id}"
            ) from exc

        return self._segments[segment_id]

    def all(
        self,
    ) -> tuple[StreetSegment, ...]:
        """
        Return all physical street segments.
        """

        return tuple(
            self._segments.values()
        )

    def directions(
        self,
        segment_id: str,
    ) -> tuple[tuple[str, str], ...]:
        """
        Return the MATSim travel directions represented by a physical
        street segment.
        """

        try:
            return self._segment_directions[segment_id]

        except KeyError as exc:
            raise KeyError(
                f"Unknown street segment: {segment_id}"
            ) from exc

    def is_bidirectional(
        self,
        segment_id: str,
    ) -> bool:
        """
        Determine whether a street segment has MATSim links in both
        directions.
        """

        segment = self.get(segment_id)

        directions = set(
            self.directions(segment_id)
        )

        return (
            (segment.node_a, segment.node_b) in directions
            and
            (segment.node_b, segment.node_a) in directions
        )

    def __len__(self) -> int:
        return len(self._segments)

    @property
    def link_to_segment(
        self,
    ) -> dict[str, str]:
        """
        Return a copy of the MATSim-link-to-segment mapping.
        """

        return dict(
            self._link_to_segment
        )


# ---------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------

def _canonical_node_pair(
    from_node: str,
    to_node: str,
) -> tuple[str, str]:
    """
    Return a deterministic unordered node pair.

    A -> B and B -> A therefore receive the same key.
    """

    if from_node <= to_node:
        return from_node, to_node

    return to_node, from_node


def _get_osm_way_id(
    link: MatsimLink,
    osm_id_attributes: tuple[str, ...],
) -> str | None:
    """
    Retrieve the OSM way ID from a MATSim link.

    Multiple possible attribute names are supported because MATSim/OSM
    conversion pipelines may use different naming conventions.
    """

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
    """
    Split links between the same MATSim node pair when they belong to
    different OSM ways.

    Normal case
    -----------
    If there is zero or one known OSM way ID, all links remain grouped.

    Parallel-way case
    -----------------
    If multiple distinct OSM way IDs occur between the same node pair,
    links are separated by OSM way ID.
    """

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

    # Normal case:
    # all directional links represent the same physical OSM way.
    if len(known_osm_ids) <= 1:

        osm_way_id = next(
            iter(known_osm_ids),
            None,
        )

        return [
            (
                osm_way_id,
                links,
            )
        ]

    # Multiple physical OSM ways occur between the same MATSim nodes.
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
    """
    Calculate total physical carriageway width.

    Directional widths are summed because each MATSim directional link
    contains only its corresponding part of the physical carriageway.

    Returns None if one or more links do not contain width information.
    """

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
                f"Invalid width value '{value}' on MATSim link "
                f"{link.id} for attribute '{width_attribute}'."
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
    """
    Construct a deterministic street-segment ID.

    Usually the unordered MATSim node pair is sufficient:

        123__456

    If multiple distinct OSM ways occur between the same nodes, the OSM
    way ID is appended:

        123__456__osm_987654
    """

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