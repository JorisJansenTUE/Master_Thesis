from __future__ import annotations

import networkx as nx

from network_sampling.network.matsim_network import MatsimNetwork
from network_sampling.network.street_segments import StreetSegmentIndex


def build_street_graph(
    network: MatsimNetwork,
    segment_index: StreetSegmentIndex,
) -> nx.MultiGraph:
    """
    Construct a physical street graph from the StreetSegmentIndex.

    MATSim nodes are represented as graph nodes and physical street
    segments as edges. A MultiGraph is used because multiple street
    segments can occur between the same pair of nodes.

    Parameters
    ----------
    network
        Baseline MATSim network containing the node coordinates.

    segment_index
        Mapping between MATSim links and physical street segments.

    Returns
    -------
    nx.MultiGraph
        Physical street graph used for intervention sampling.
    """

    graph = nx.MultiGraph()

    for segment in segment_index.all():
        for node_id in (
            segment.node_a,
            segment.node_b,
        ):
            if node_id not in graph:
                node = network.nodes[node_id]

                graph.add_node(
                    node_id,
                    x=node.x,
                    y=node.y,
                )

        graph.add_edge(
            segment.node_a,
            segment.node_b,
            key=segment.id,
            length_m=segment.length_m,
            total_width_m=segment.total_width_m,
        )

    return graph