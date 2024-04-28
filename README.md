# Analyzing AI Global NWP models in Forecasting Bomb-Cyclogenesis

In this project we sought to analyze how different AI-based global NWP models forecasting bomb-cyclogenesis. Bomb Cyclogenensis refers to when a extratropical cyclone undergoes rapid intensification (24 hPa in 24 hours at 60 degrees latitude). The data used for this project comprise of the following:

- Pangu-Weather [reference](https://www.nature.com/articles/s41586-023-06185-3)
- FourCastNet v2 [reference](https://arxiv.org/abs/2306.03838)
- ECMWF Reanalysis version 5 (ERA5) [source](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)

Below are the imports needed to run this project

```
# Data
import numpy as np
import netCDF4 as nf
import pandas as pd
import xarray as xr
import datetime
# Plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
import sys,os
# Image Rendering
import imageio
# Notebook specific
from IPython.display import Image
```

## Below are the how the different scripts contributed to this project.

- aws_download.py: This was used to download the data from the amazon cloud to the supercomputer
- ai_cyclone_track.ipynb: This script was used to do some initial data exploration, getting a feel for the data structure, and start working on some plotting concepts
- cyclone_tracking_v1.py: This script was used to work on an objective cyclone tracking algorithm. A variety of different functions to track and plot the results were developed. While the process of cyclone tracking was initially promising, delays and setbacks to producing a reliable tracking algorithm eventually lead to the abandonment of this strategy
- cyclone_analysis.py: This script embodied most of the analysis that was done for the case studies used in this project. This contained methods for extracing the cyclone centers, saving the data, and producing an array of plots
- compare_cyclones.ipynb: This python notebook was the basis for the cyclone analysis script. Over one case study, a variety of plots were produced for analysis, which then migrated into function form over to the cyclone analysis script
- event_cropping.ipynb: This notebook was developed after the cyclone_analysis.py script to be used to iterate quickly over the variety of the different cases covered by the media within our timeframe to integrate into the settings files which are located in the `settings/` directory.

In the `settings/` directory, the settings files for each of the cases are located. This includes the start and end times of the event, and the domain to be used for identifying and tracking the cyclone.
