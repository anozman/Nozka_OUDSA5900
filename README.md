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
