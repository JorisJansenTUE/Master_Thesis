from __future__ import annotations

import gzip
from pathlib import Path

from lxml import etree

from network_sampling.network.matsim_network import MatsimNetwork
from network_sampling.network.street_segments import StreetSegmentIndex


DEFAULT_TRANSIT_MODES = frozenset({"bus"})


class TransitUsageIndex:
    """
    Index of MATSim links used by scheduled transit routes.

    The index is constructed from the transit schedule and stores the links
    traversed by routes of the selected transit modes. Street-segment usage
    and travel directions are derived from these links when requested.
    """

    def __init__(
        self,
        network: MatsimNetwork,
        segment_index: StreetSegmentIndex,
        used_link_ids: frozenset[str],
    ):
        self._network = network
        self._segment_index = segment_index
        self._used_link_ids = used_link_ids

    @classmethod
    def from_schedule(
        cls,
        path: str | Path,
        network: MatsimNetwork,
        segment_index: StreetSegmentIndex,
        *,
        transit_modes: frozenset[str] = DEFAULT_TRANSIT_MODES,
    ) -> "TransitUsageIndex":
        """
        Construct the index from a MATSim transit schedule.

        Only transit routes matching the selected modes and containing at
        least one scheduled departure are included.

        Parameters
        ----------
        path
            Path to the MATSim transit schedule (.xml or .xml.gz).

        network
            Baseline MATSim network referenced by the transit schedule.

        segment_index
            Mapping between MATSim links and physical street segments.

        transit_modes
            Transit modes to include in the index.

        Returns
        -------
        TransitUsageIndex
            Index containing all MATSim links used by the selected routes.
        """

        path = Path(path)

        parser = etree.XMLParser(
            load_dtd=False,
            no_network=True,
            resolve_entities=False,
        )

        opener = (
            gzip.open
            if path.suffix == ".gz"
            else open
        )

        with opener(path, "rb") as file:
            root = etree.parse(
                file,
                parser,
            ).getroot()

        used_link_ids: set[str] = set()

        for transit_route in root.iter("transitRoute"):
            mode = transit_route.findtext("transportMode")

            if (mode is None or mode.strip() not in transit_modes):
                continue

            if transit_route.find("departures/departure") is None:
                continue

            route_element = transit_route.find("route")

            if route_element is None:
                continue

            for link_element in route_element.findall(
                "link"
            ):
                link_id = link_element.get(
                    "refId"
                )

                if link_id is None:
                    continue

                if link_id not in network.links:
                    raise ValueError(
                        f"Transit schedule references unknown "
                        f"MATSim link {link_id}."
                    )

                used_link_ids.add(
                    link_id
                )

        return cls(
            network=network,
            segment_index=segment_index,
            used_link_ids=frozenset(
                used_link_ids
            ),
        )

    def uses_link(
        self,
        link_id: str,
    ) -> bool:
        """Return whether a scheduled transit route uses a MATSim link."""

        return link_id in self._used_link_ids

    def links(
        self,
        segment_id: str,
    ) -> tuple[str, ...]:
        """Return scheduled transit links belonging to a street segment."""

        segment = self._segment_index.get(
            segment_id
        )

        return tuple(
            link_id
            for link_id in segment.link_ids
            if link_id in self._used_link_ids
        )

    def directions(
        self,
        segment_id: str,
    ) -> frozenset[tuple[str, str]]:
        """Return directions in which scheduled transit uses a street segment."""

        return frozenset(
            (
                self._network.links[link_id].from_node,
                self._network.links[link_id].to_node,
            )
            for link_id in self.links(segment_id)
        )

    def uses_segment(
        self,
        segment_id: str,
    ) -> bool:
        """Return whether scheduled transit uses a street segment."""

        return bool(
            self.links(segment_id)
        )

    def __len__(
        self,
    ) -> int:
        return len(self._used_link_ids)