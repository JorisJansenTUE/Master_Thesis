# Assign Road Widths

This standalone preprocessing pipeline estimates carriageway widths for an OpenStreetMap (OSM) network using cadastral data. The estimated widths can subsequently be transferred to a MATSim network for use in road-space allocation and optimization.

## 1. Width Estimation

The pipeline estimates carriageway widths by generating transects perpendicular to each OSM road geometry and measuring their intersections with cadastral road surfaces.

To reduce the influence of extreme measurements, the **median transect intersection length** is used as the estimated carriageway width for each OSM way.

## 2. Error Correction and Fallback Widths

After the initial estimation, `helpers/fix_errors.py` applies rule-based corrections to missing or unreliable width measurements.

When a measurement is missing or classified as unreliable, the width is estimated using:

$$ W_{\text{fallback}} = N_{\text{lanes}} \times W_{\text{median, highway}}$$

where:
- $N_{\text{lanes}}$ is the number of lanes specified in OSM, or an assumed lane count when this information is unavailable.
- $W_{\text{median, highway}}$ is the median width per lane for the corresponding OSM highway type.

These highway-specific median lane widths were calibrated using reliable cadastral measurements in `analyze_width_per_lane.py`.

The pipeline also applies several Locarno-specific corrections, which are further commented on in `fix_errors.py` including:
- Primary roads represented by two separate one-way OSM ways that share a cadastral road surface.
- Exceptions for Via Stazione.


## 3. Configuration

Input paths, output paths, and estimation parameters are configured in the `main()` function of `pipeline.py`.

When clipping is enabled (`clip=True`), the OSM network and cadastral data are first clipped to the specified project boundary.

When clipping is disabled (`clip=False`), the pipeline uses the existing input files without performing additional clipping.

## 4. Running the Pipeline

From the repository root, execute:

```bash
python -m preprocessing.assign_road_widths.pipeline
```

The pipeline produces an OSM network containing the estimated carriageway widths, including any fallback corrections.

## 5. Transferring Widths to MATSim

The estimated final carriageway widths can be transferred to an existing MATSim network using the OSM way IDs preserved during network conversion.

Use `add_osm_widths_to_matsim.py` to match OSM ways to MATSim links and add the final carriageway width as a link attribute.

```bash
python -m preprocessing.assign_road_widths.add_osm_widths_to_matsim \
    --osm path/to/network_with_widths.osm.gz \
    --network path/to/matsim_network.xml.gz \
    --output path/to/matsim_network_with_widths.xml.gz
```

The script adds the `carriageway_width_m` attribute to matching MATSim links without modifying other network attributes.

**Note:** Multiple MATSim links may correspond to the same OSM way. The transferred width represents the physical carriageway width and should not be interpreted as independently available road space for each directed link.