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

### 2. Extract OSM data (Locarno example)
OSM Data used for the network structure was taken from Geofabrik (https://download.geofabrik.de/europe/switzerland.html). The actual research area (Locarno in our case) can then be extracted by running osmium through powershell:
```bash
osmium extract `
  --bbox 8.65,46.05,9.05,46.35 `
  --set-bounds `
  -o data/osm/locarno.osm.pbf `
  data/osm/switzerland.osm.pbf
```
The bounding box can be determined using osm.org
### 3. MATSim

* Convert `.osm` → `network.xml`
* Run simulation from `/matsim/`

---

## Notes

* Initial version is road-only
* MATSim is used as a black-box model
* Will be expanded later
