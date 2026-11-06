from __future__ import annotations

import networkx as nx

from network_sampling.network.matsim_network import MatsimNetwork


DEFAULT_MODES = ("car", "bike", "truck")


def build_mode_graph(network: MatsimNetwork, mode: str) -> nx.DiGraph:
    """Construct the directed MATSim network for one transport mode."""

    graph = nx.DiGraph()

    for link in network.links.values():
        if mode in link.allowed_modes:
            graph.add_edge(link.from_node, link.to_node)

    return graph


def check_connectivity(
    baseline: MatsimNetwork,
    modified: MatsimNetwork,
    modes: tuple[str, ...] = DEFAULT_MODES,
) -> dict[str, bool]:
    """
    Check whether modal networks remain strongly connected.

    Baseline modal nodes are added to the modified graph so that removing
    all modal access to a node is also detected.
    """

    results = {}

    for mode in modes:
        baseline_graph = build_mode_graph(baseline, mode)
        modified_graph = build_mode_graph(modified, mode)

        modified_graph.add_nodes_from(baseline_graph.nodes)

        if not nx.is_strongly_connected(baseline_graph):
            raise ValueError(f"Baseline {mode} network is not strongly connected.")

        results[mode] = nx.is_strongly_connected(modified_graph)

    return results


def validate_connectivity(
    baseline: MatsimNetwork,
    modified: MatsimNetwork,
    modes: tuple[str, ...] = DEFAULT_MODES,
) -> None:
    """Raise an error if an intervention disconnects a modal network."""

    results = check_connectivity(baseline, modified, modes)
    failed = [mode for mode, connected in results.items() if not connected]

    if failed:
        raise ValueError(
            f"Interventions disconnect the following modes: {', '.join(failed)}"
        )