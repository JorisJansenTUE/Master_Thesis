# Assign Road Widths
This is currently a standalone element that adds roadwidths to an OSM network using Cadastral data.

The inputs are configures in the main function of `pipeline.py` alter the paths here to the correct location.

It assumes all data has already been clipped to the correct shape, this is done outside the pipeline using `cadastral_clipper.py` and `Osmium`.

>**TO DO:** Integrate into the Networkpipeline and improve standalone operation with clipping
