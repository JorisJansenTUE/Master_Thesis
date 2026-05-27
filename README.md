# Master Thesis – MATSim & Surrogate Modelling

## Overview

This repository contains a minimal working setup for my master thesis, which will combine:

* MATSim simulations
* Python Surrogate Modelling
* Python Data preprocessing


__!! WIP !!__

## Quick Start

### 1. Conda Environment setup
In powershell run once:
```bash
conda env create -f environment.yml
```
Then to activate the environment run:
```bash
conda activate thesis
```

### 2. Creating a MATSim Network (Locarno Example)
OSM Data used for the network structure was taken from [Geofabrik](https://download.geofabrik.de/europe/switzerland.html).   
GTFS data was downloaded from [Geodienste](https://data.opentransportdata.swiss/en/dataset/timetable-2026-gtfs2020)  
The Project boundary shape was drawn in QGIS.

The Network is then created using the pipeline as found in `preprocessing/network`. Run the pipeline as a python module from the repository root:

```powershell
python -m preprocessing.network.pipeline `
  --scenario locarno `
  --osm data/raw/osm/Switzerland-260413 `
  --shape data/raw/shapes/Locarno_QGIS_2056.shp `
  --gtfs data/raw/gtfs/gtfs_swiss_20260422.zip `
  --sample-day 20260422 `
  --additional-line-info schedule `
  --overwrite
```  
Check the README in the module folder for more specific information about the pipeline.

#### Additional features to be added:
- Add compatibility with `preprocessing/assign_road_widths` 
- Add additional heuristic after network creation that compares generated max speed with max speed in OSM and overwrites with the OSM speed when not equal.
- Refine which road types to keep and which not (e.g. include service roads or not?)
- Refine where bicycles should be allowed

### 3. Creating a Synthetic Population

## Notes

* Initial version is road-only
* Will be expanded later
