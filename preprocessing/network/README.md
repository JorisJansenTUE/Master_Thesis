# MATSim Network Creation Pipeline

This pipeline creates a MATSim-ready multimodal network using the
[PT2MATSim](https://github.com/matsim-org/pt2matsim) package.

It is designed to automate the full preprocessing workflow from raw OSM, GTFS, and project-boundary data to a mapped MATSim network and transit schedule.

The pipeline is scenario-based: each run writes its outputs to a separate scenario folder, making it easy to create and compare multiple study areas or modelling
setups.

---

## Workflow

The pipeline performs the following steps:

1. **Clip the OSM file to the project shapefile**
   - Input: a large `.osm.pbf` file, for example a national or regional extract.
   - Output: a smaller clipped `.osm.gz` file.
   - The clipping is done with `osmium`.

2. **Create the unmapped multimodal network**
   - Uses PT2MATSim's `Osm2MultimodalNetwork`.
   - The modelling settings are read from the OSM converter XML template.
   - The pipeline automatically overwrites only the input and output paths.

3. **Clip the GTFS feed to the project shapefile**
   - Input: a national or regional GTFS `.zip`.
   - The pipeline keeps:
     - all stops inside the project area;
     - one stop before entering the project area;
     - one stop after leaving the project area.
   - This avoids keeping full long-distance routes while still preserving local route continuity.
   - `transfers.txt` is deliberately dropped because it is optional and often causes issues after clipping.

4. **Create the unmapped transit schedule**
   - Uses PT2MATSim's `Gtfs2TransitSchedule`.
   - This step does **not** use an XML config file.
   - Instead, it passes the required command-line arguments directly:
     - GTFS folder;
     - sample day;
     - output CRS;
     - output schedule file;
     - output vehicles file;
     - optional line information.

5. **Create the mapped network and mapped transit schedule**
   - Uses PT2MATSim's public transit mapper.
   - The mapper settings are read from the mapper XML template.
   - The pipeline automatically overwrites only the relevant input and output paths.

## Requirements

The pipeline requires:

- Python environment with:
  - `geopandas`
  - `pandas`
  - `shapely`
  - `lxml`
- `osmium` installed and available on `PATH`
- Java 17
- Maven
- A Maven-based MATSim project with PT2MATSim available as a dependency

## Project Structure
The pipeline assumes that all files are placed in `project_root/preprocessing/network`.\
Furthermore, all project-wide paths are expected to be defined in `preprocessing/utils.py`.

 **Note:** Alter all path strings in this file to match your project structure.

## Input Files
Each scenario requires three main input files:

### 1. OSM file
A sufficiently large OSM extract, preferably as `.osm.pbf`
The pipeline will clip this file and write a smaller `.osm.gz` file which works with the PT2MATSim converter

### 2. Project shapefile
A polygon shapefile defining the study area.

The shapefile must have a valid CRS. This can be a local projected CRS, the shapefile will reporject it internally to WGS84/EPSG:4326 for OSM and GTFS clipping, before projecting the entire network to the desired CRS.

### 3. GTFS feed
A national or regional GTFS zip file, with at least the following required files: `agency.txt`, `routes.txt`, `trips.txt`, `stop_times.txt`, `stops.txt`

## Config Files
In addition to the main inputfiles, PT2MATSim requires a config file for two tasks: parsing of the OSM Network and mapping the transit schedule onto the network.

The pipeline assumes by default that these two config files are :
- `configs/pt2matsim/osm2multimodal_template.xml`
- `configs/pt2matsim/transit_mapper_template.xml`

You can use these or customize them for a specific scenario, please check the [PT2MATSim Wiki](https://github.com/matsim-org/pt2matsim/wiki) for more details.

Additionally, overrides can also be added as an additional argument to the pipeline using `--osm-override` or `--mapper-overide` followed by the value to be changed, e.g. `--osm-override maxLinkLength=500.0`.

## Usage
Run the pipeline as a python module from the repository root:

    python -m preprocessing.network.pipeline `
    --scenario <scenario_name> `
    --osm path/to/osm/country.osm.pbf `
    --shape path/to/shape/city.shp `
    --gtfs path/to/gtfs/gtfs_country.zip `
    --sample-day yyyymmdd `
    --additional-line-info schedule `
    --overwrite

#### Main Arguments

`--scenario`: Name of the scenario, this determines the name of the output folders of project:
```
data/interim/scenario_name
data/scenarios/scenario_name
```

`--osm`: Path to the input OSM file.

`--shape`: Path to the project-area shapefile.

`--gtfs`: Path to the GTFS zip file.

#### Optional Arguments

`--target-crs`: Optional. Desired output CRS for MATSim-ready files. Defaults to `"EPSG:2056"`. Use a projected CRS in metres that is appropriate for the study area.

`--sample-day`: Optional. Service day used by [`Gtfs2TransitSchedule`](https://github.com/matsim-org/pt2matsim/wiki/Creating-an-unmapped-MATSim-transit-schedule). Defaults to `"dayWithMostTrips"`. Can also be set to a specific date, for example `"20260422"`.

`--additional-line-info`: Optional. Additional line information argument passed to `Gtfs2TransitSchedule`. Defaults to `"schedule"`. This can be useful for retaining extra route or line information during schedule creation.

`--shape-buffer-degrees`: Optional. Buffer applied to the project shapefile after it is reprojected to WGS84. Defaults to `0.0`. This should normally stay at `0.0`, but a small positive value can be used if stops or OSM features very close to the boundary are accidentally excluded.

`--overwrite`: Optional flag. If included, existing output folders for the scenario are deleted before the pipeline is rerun. Use this when repeating a run after changing input files, config templates, or pipeline settings.

`--java-runner`: Optional. Command used to call Maven. Defaults to `"mvn"`. On some Windows setups, `"mvn.cmd"` may be required instead.

`--osm-config-template`: Optional. Path to the PT2MATSim `Osm2MultimodalNetwork` config template. Defaults to `configs/pt2matsim/osm2multimodal_template.xml`. The pipeline overwrites only the relevant input/output paths and CRS; the modelling settings remain defined in this template.

`--mapper-config-template`: Optional. Path to the PT2MATSim public transit mapper config template. Defaults to `configs/pt2matsim/transit_mapper_template.xml`. The pipeline overwrites only the relevant input/output paths and CRS; the mapping settings remain defined in this template.

`--osm-override`: Optional. Override a parameter in the OSM converter config from the command line. Can be used multiple times. Example: `--osm-override maxLinkLength=500.0`.

`--mapper-override`: Optional. Override a parameter in the public transit mapper config from the command line. Can be used multiple times. Example: `--mapper-override maxLinkCandidateDistance=90.0`.

## Output files
All output files will be written to two folders:

- `data/interim/scenario_name`: Includes the clipped OSM and GTFS and the reprojected WSG84 boundary 
- `data/scenarios/scenario_name`: Includes all files required for MATSim simulations (Network, Schedule, Vehicles), generated config files used by PT2Matsim and logs of the pipeline.