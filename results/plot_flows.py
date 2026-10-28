from pathlib import Path
import gzip
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from shapely.geometry import LineString, Point
from shapely.prepared import prep

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.widgets import Slider

from preprocessing.utils import DATA_DIR    


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------

SCENARIO = Path(
    DATA_DIR / "NDA_Eth_Model" / "Locarno" / "simulation_output_v2"
)

NETWORK_FILE = SCENARIO / "output_network.xml.gz"
FLOW_FILE = SCENARIO / "output_traffic_flow.csv"

ANALYSIS_BOUNDARY = DATA_DIR / "raw" / "shapes" / "Locarno_project_boundary.shp"
# Use this instead to plot the full network:
# ANALYSIS_BOUNDARY = None


# Thickness of links
MIN_LINEWIDTH = 0.25
MAX_LINEWIDTH = 5.0

# Colormap
CMAP = "turbo"

# None -> use maximum flow across all 24 hours.
# You can manually set e.g. 500 if desired.
FLOW_MAX = None



def read_analysis_boundary(boundary_file):
    """Read and prepare the analysis boundary."""

    if boundary_file is None:
        print("No analysis boundary specified. Using full network.")
        return None
    print("Reading analysis boundary...")

    boundary = gpd.read_file(boundary_file)

    if boundary.empty:
        raise ValueError(
            f"Analysis boundary contains no features: {boundary_file}"
        )

    print(f"  Original CRS: {boundary.crs}")

    if boundary.crs is None:
        raise ValueError("Analysis boundary has no CRS defined.")

    # Combine all features into one geometry
    geometry = boundary.geometry.union_all()


   
    print(f"  Boundary features: {len(boundary):,}")

    return prep(geometry)

# ------------------------------------------------------------------
# Read MATSim network
# ------------------------------------------------------------------

def read_network(network_file):
    """
    Read nodes and links from a MATSim network.

    Returns
    -------
    nodes : dict
        node_id -> (x, y)

    links : dict
        link_id -> (from_node, to_node)
    """

    print("Reading MATSim network...")

    nodes = {}
    links = {}

    with gzip.open(network_file, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):

            tag = elem.tag.split("}")[-1]

            if tag == "node":
                nodes[elem.attrib["id"]] = (
                    float(elem.attrib["x"]),
                    float(elem.attrib["y"]),
                )

            elif tag == "link":
                links[elem.attrib["id"]] = (
                    elem.attrib["from"],
                    elem.attrib["to"],
                )

            elem.clear()

    print(f"  Nodes: {len(nodes):,}")
    print(f"  Links: {len(links):,}")

    return nodes, links


# ------------------------------------------------------------------
# Read traffic flow
# ------------------------------------------------------------------

def read_flows(flow_file):
    """
    Read MATSim output_traffic_flow.csv.

    The file has:
        linkId;bin0;bin1;...;bin23

    with decimal commas.
    """

    print("Reading traffic flows...")

    df = pd.read_csv(
        flow_file,
        sep=";",
        decimal=",",
        dtype={"linkId": str},
    )

    hour_columns = [
        f"bin{i}"
        for i in range(24)
        if f"bin{i}" in df.columns
    ]

    print(f"  Links with flow data: {len(df):,}")
    print(f"  Hour bins: {len(hour_columns)}")

    return df, hour_columns


# ------------------------------------------------------------------
# Prepare network geometry and flow matrix
# ------------------------------------------------------------------

def prepare_data(nodes, links, flows, hour_columns, analysis_geometry):
    """Match flow data to links and optionally restrict them to the analysis area."""

    flow_lookup = flows.set_index("linkId")

    segments = []
    flow_values = []
    matched_link_ids = []

    missing_flow = 0
    outside_boundary = 0

    for link_id, (from_node, to_node) in links.items():

        if from_node not in nodes or to_node not in nodes:
            continue

        x1, y1 = nodes[from_node]
        x2, y2 = nodes[to_node]

        segment = [(x1, y1), (x2, y2)]

        if analysis_geometry is not None:
            line = LineString(segment)

            if not analysis_geometry.intersects(line):
                outside_boundary += 1
                continue

        segments.append(segment)
        matched_link_ids.append(link_id)

        if link_id in flow_lookup.index:
            values = flow_lookup.loc[link_id, hour_columns].to_numpy(dtype=float)
        else:
            values = np.zeros(len(hour_columns))
            missing_flow += 1

        flow_values.append(values)

    flow_values = np.asarray(flow_values)

    print(f"  Network segments used: {len(segments):,}")

    if analysis_geometry is not None:
        print(f"  Links outside boundary: {outside_boundary:,}")

    print(f"  Links without flow data: {missing_flow:,}")

    if len(segments) == 0:
        raise ValueError("No MATSim links found in the selected analysis area.")

    return segments, flow_values, matched_link_ids

# ------------------------------------------------------------------
# Convert flows to line widths
# ------------------------------------------------------------------

def calculate_linewidths(values, flow_max):
    """
    Scale line thickness using sqrt(flow).

    Square-root scaling makes low/moderate flows visible without
    letting the busiest links become excessively thick.
    """

    values = np.clip(values, 0, flow_max)

    scaled = np.sqrt(values / flow_max)

    return MIN_LINEWIDTH + scaled * (
        MAX_LINEWIDTH - MIN_LINEWIDTH
    )


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

def plot_hourly_flows(segments, flow_values, hour_columns):

    global FLOW_MAX

    # --------------------------------------------------------------
    # Fixed flow scale across ALL hours
    # --------------------------------------------------------------

    if FLOW_MAX is None:
        flow_max = np.nanmax(flow_values)
    else:
        flow_max = FLOW_MAX

    print(f"Maximum displayed flow: {flow_max:.1f} veh/h")

    norm = Normalize(
        vmin=0,
        vmax=flow_max,
        clip=True,
    )

    cmap = plt.get_cmap(CMAP)

    # --------------------------------------------------------------
    # Initial hour
    # --------------------------------------------------------------

    initial_hour = 7

    if initial_hour >= len(hour_columns):
        initial_hour = 0

    initial_flows = flow_values[:, initial_hour]

    # --------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    # Leave room underneath for slider
    plt.subplots_adjust(
        bottom=0.13,
        right=0.90,
    )

    # --------------------------------------------------------------
    # Network
    # --------------------------------------------------------------

    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=calculate_linewidths(
            initial_flows,
            flow_max,
        ),
    )

    collection.set_array(initial_flows)

    ax.add_collection(collection)

    # Fit view to network
    ax.autoscale()

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.axis("off")

    # --------------------------------------------------------------
    # Title
    # --------------------------------------------------------------

    title = ax.set_title(
        f"MATSim traffic flow — "
        f"{initial_hour:02d}:00–{initial_hour + 1:02d}:00",
        fontsize=15,
    )

    # --------------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------------

    sm = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        fraction=0.035,
        pad=0.02,
    )

    cbar.set_label(
        "Traffic flow [vehicles/hour]"
    )

    # --------------------------------------------------------------
    # Slider
    # --------------------------------------------------------------

    slider_ax = fig.add_axes(
        [0.15, 0.045, 0.65, 0.03]
    )

    hour_slider = Slider(
        ax=slider_ax,
        label="Hour",
        valmin=0,
        valmax=len(hour_columns) - 1,
        valinit=initial_hour,
        valstep=1,
    )

    # --------------------------------------------------------------
    # Update function
    # --------------------------------------------------------------

    def update(value):

        hour = int(hour_slider.val)

        current_flows = flow_values[:, hour]

        # Update colours
        collection.set_array(current_flows)

        # Update thickness
        collection.set_linewidths(
            calculate_linewidths(
                current_flows,
                flow_max,
            )
        )

        # Update title
        title.set_text(
            f"MATSim traffic flow — "
            f"{hour:02d}:00–{hour + 1:02d}:00"
        )

        fig.canvas.draw_idle()

    hour_slider.on_changed(update)

    # --------------------------------------------------------------
    # Optional keyboard controls
    # --------------------------------------------------------------

    def key_press(event):

        current = int(hour_slider.val)

        if event.key == "right":
            hour_slider.set_val(
                min(current + 1, len(hour_columns) - 1)
            )

        elif event.key == "left":
            hour_slider.set_val(
                max(current - 1, 0)
            )

    fig.canvas.mpl_connect(
        "key_press_event",
        key_press,
    )

    plt.show()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    analysis_geometry = read_analysis_boundary(ANALYSIS_BOUNDARY)

    nodes, links = read_network(NETWORK_FILE)
    flows, hour_columns = read_flows(FLOW_FILE)

    segments, flow_values, link_ids = prepare_data(
        nodes,
        links,
        flows,
        hour_columns,
        analysis_geometry,
    )

    plot_hourly_flows(segments, flow_values, hour_columns)


if __name__ == "__main__":
    main()