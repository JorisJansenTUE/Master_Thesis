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
The bounding box can be determined using XXX

Next we require a more detailed specification of the studied research area as a shapefile. One could decide to use administrative borders for this and extract them using the function `Get_Shapefile` in `preprocessing.data_helperfunctions.py` but for our scenario this is not ideal and thus a shape was drawn manually in QGIS and exported:

<include figure

### 3. MATSim
The following Tutorials were used to create all java code in the thesis project folder:

CreateNetwork_v2:\
https://github.com/matsim-org/matsim-code-examples/blob/dev.x/src/main/java/org/matsim/codeexamples/network/RunCreateNetworkFromOSM.java 

CreatePopulation (with modification):\
https://github.com/matsim-org/matsim-code-examples/blob/dev.x/src/main/java/org/matsim/codeexamples/population/demandGenerationFromShapefile/CreateDemand.java

---
To convert the extracted OSM file `.osm.pbf` to a MatSim ready `.xml` we use `MATSim/matsim-example-project/src/java/thesis/network/CreateNetwork_v2.java` :
``` bash
#osmium cat data/osm/locarno.osm.pbf -o data/osm/locarno.osm => only when using V1
cd MATSim/matsim-example-project
mvn exec:java "-Dexec.mainClass=thesis.network.CreateNetwork_v2"
```
Next we can use the created network file and shape file to create a randomly sampled population using `CreateSyntheticPopulationFromShapefile.java`. \
Run the following:

```bash
cd MATSim/matsim-example-project
mvn exec:java "-Dexec.mainClass=thesis.population.CreateSyntheticPopulationFromShapefile"
```

New version WIP:"
```bash
osmium extract `                  
  -p data/osm/locarno_QGIS_drawn.geojson `                                                     
  -o data/osm/locarno_smallest.osm.pbf `
  data/osm/locarno.osm.pbf
osmium cat ../../data/osm/locarno_smallest.osm.pbf -o ../../data/osm/locarno_smallest.osm.gz
mvn exec:java "-Dexec.mainClass=org.matsim.pt2matsim.run.Osm2MultimodalNetwork" "-Dexec.args=scenarios/input_config/multi_modal.xml"

```

## Notes

* Initial version is road-only
* Will be expanded later
