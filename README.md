# Master Thesis – MATSim & Surrogate Modelling

## Overview

This repository contains a minimal working setup for my master thesis, which will combine:

* MATSim simulations
* Python Surrogate Modelling
* Python Data preprocessing
* LaTeX paper with Zotero-based references


__!! WIP !!__

## Quick Start

### 1. Python setup

```bash
pip install -r requirements.txt
```

### 2. Extract OSM data (Locarno example)

```python
from pyrosm import OSM

bbox = [8.65, 46.05, 9.05, 46.35]
osm = OSM("switzerland-latest.osm.pbf", bounding_box=bbox)
osm.to_xml("data/osm/locarno.osm")
```

### 3. MATSim

* Convert `.osm` → `network.xml`
* Run simulation from `/matsim/`

---

## Paper

* Located in `/paper/`
* Compile with:

```bash
latexmk -pdf main.tex
```

### Zotero

* Export library to `/paper/references.bib` (Better BibTeX auto-export recommended)

---

## Notes

* Initial version is road-only
* MATSim is used as a black-box model
* Will be expanded later
