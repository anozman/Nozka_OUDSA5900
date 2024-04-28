#-------------------------------------------------------------------------------------------------------
# IMPORTS
#-------------------------------------------------------------------------------------------------------
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
# General
import scipy
import openpyxl
import tqdm
import sys,os

import imageio
#from IPython.display import Image

#-------------------------------------------------------------------------------------------------------
# CONSTANTS
#-------------------------------------------------------------------------------------------------------
# Plotting projections
crs_us = ccrs.LambertConformal(central_longitude=260.0)
crs_global_atlantic = ccrs.PlateCarree(central_longitude=0.0)
crs_pnw = ccrs.LambertConformal(central_longitude=240.0)

# Plotting colorbar nomalization
pmin = 940
pmax = 1040

root_data_path = '/ourdisk/hpc/ai2es/datasets/noaa-oar-mlwp-data/'
root_save_path = '/ourdisk/hpc/ai2es/alexnozka/bomb_cyclones/results/'


#-------------------------------------------------------------------------------------------------------
# PLOTTING FUNCTIONS
#-------------------------------------------------------------------------------------------------------
def plot_background_no_lims(ax):
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    return ax

def plot_subset_background(ax, lonmin, lonmax, latmin, latmax):
    ax.set_extent([lonmin, lonmax, latmin, latmax])
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    return ax

def find_storm_domain(dataset):
    bounds = [None] * 4
    bounds[0] = dataset['longitude'].min()
    bounds[1] = dataset['longitude'].max()
    bounds[2] = dataset['latitude'].min()
    bounds[3] = dataset['latitude'].max()
    return bounds

def minp_density_plot(pangu_preds, fourcast_preds, era_tracks, time_arr, save_path, start_day, year):
    #time_arr_cut = np.arange(len(time_arr))  # Assuming time_arr is an array of timestamps
    time_arr_cut = time_arr

    # Extract pressure values from dictionaries for Pangu-Weather predictions
    pangu_pressure_lists = [pangu_preds[lead_time] for lead_time in sorted(pangu_preds.keys())]
    for ppreds in range(0,len(pangu_pressure_lists)):
        pangu_pressure_lists[ppreds] = [pdict['msl'] for pdict in pangu_pressure_lists[ppreds]]
        #print("\nppreds:\n", pangu_pressure_lists[ppreds])
        #print('\n')
    #pangu_pressure_lists = [ppreds['msl'] for ppreds in pangu_pressure_lists]
    #pangu_pressure_litsts = filter(lambda x: len(x) == len(time_arr_cut), pangu_pressure_lists)

    # Extract pressure values from dictionaries for FourCastNet v2 predictions
    fourcast_pressure_lists = [fourcast_preds[lead_time] for lead_time in sorted(fourcast_preds.keys())]
    for fpreds in range(0,len(fourcast_pressure_lists)):
        fourcast_pressure_lists[fpreds] = [fdict['msl'] for fdict in fourcast_pressure_lists[fpreds]]
    #fourcast_pressure_lists = [fpreds['msl'] for fpreds in fourcast_pressure_lists]
    #fourcast_pressure_lists = filter(lambda x: len(x) == len(time_arr_cut), fourcast_pressure_lists)

    # Filter out lists with a length different from time_arr_cut
    pangu_pressure_lists = list(filter(lambda x: len(x) == len(time_arr_cut), pangu_pressure_lists))
    fourcast_pressure_lists = list(filter(lambda x: len(x) == len(time_arr_cut), fourcast_pressure_lists))


    # Calculate minimum pressure for ERA5
    era_mins = [track['msl'] for track in era_tracks]

    #print("\n")
    #print("Pangu-weather shape: %s\n"%(str(np.shape(pangu_pressure_lists))) , pangu_pressure_lists)
    #print("\n")
    #print("Fourcast shape: %s\n"%(str(np.shape(pangu_pressure_lists))) , fourcast_pressure_lists)
    #print("\n")
    #print("ERA5 shape: %s\n"%(str(np.shape(era_mins))), era_mins)

    # Calculate percentiles for Pangu-Weather predictions
    pangu_median = np.median(pangu_pressure_lists, axis=0)
    pangu_10th_percentile = np.percentile(pangu_pressure_lists, 10, axis=0)
    pangu_90th_percentile = np.percentile(pangu_pressure_lists, 90, axis=0)

    # Calculate percentiles for FourCastNet v2 predictions
    fourcast_median = np.median(fourcast_pressure_lists, axis=0)
    fourcast_10th_percentile = np.percentile(fourcast_pressure_lists, 10, axis=0)
    fourcast_90th_percentile = np.percentile(fourcast_pressure_lists, 90, axis=0)

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True, facecolor='w')

    # Plot Pangu-Weather data with 10th and 90th percentile shading
    ax.plot(time_arr_cut, pangu_median, label='Pangu-Weather', color='blue', linewidth=3)
    ax.fill_between(time_arr_cut, pangu_10th_percentile, pangu_90th_percentile, color='blue', alpha=0.2)

    # Plot FourCastNet v2 data with 10th and 90th percentile shading
    ax.plot(time_arr_cut, fourcast_median, label='FourCastNet v2', color='red', linewidth=3)
    ax.fill_between(time_arr_cut, fourcast_10th_percentile, fourcast_90th_percentile, color='red', alpha=0.2)

    # Plot ERA5 data
    ax.plot(time_arr_cut, era_mins, label='ERA5', color='black', linewidth=3)

    # Set plot title and labels
    ax.set_title("Minimum pressure over time", fontsize=14)
    ax.set_ylabel("Central Pressure (Pa)")
    ax.legend()

    fig.savefig("%s%s_%s_minp_density_plot.png"%(save_path, start_day, year))

def minp_density_plot_contoured(pangu_preds, fourcast_preds, era_tracks, time_arr, save_path, start_day, year):
    time_arr_cut = time_arr

    # Extract pressure values from dictionaries for Pangu-Weather predictions
    pangu_pressure_lists = [pangu_preds[lead_time] for lead_time in sorted(pangu_preds.keys())]
    for ppreds in range(0,len(pangu_pressure_lists)):
        pangu_pressure_lists[ppreds] = [pdict['msl'] for pdict in pangu_pressure_lists[ppreds]]

    # Extract pressure values from dictionaries for FourCastNet v2 predictions
    fourcast_pressure_lists = [fourcast_preds[lead_time] for lead_time in sorted(fourcast_preds.keys())]
    for fpreds in range(0,len(fourcast_pressure_lists)):
        fourcast_pressure_lists[fpreds] = [fdict['msl'] for fdict in fourcast_pressure_lists[fpreds]]

    # Filter out lists with a length different from time_arr_cut
    pangu_pressure_lists = list(filter(lambda x: len(x) == len(time_arr_cut), pangu_pressure_lists))
    fourcast_pressure_lists = list(filter(lambda x: len(x) == len(time_arr_cut), fourcast_pressure_lists))

    # Calculate minimum pressure for ERA5
    era_mins = [track['msl'] for track in era_tracks]

    # Calculate min and max values
    pangu_min = np.min(pangu_pressure_lists, axis=0)
    pangu_max = np.max(pangu_pressure_lists, axis=0)
    fourcast_min = np.min(fourcast_pressure_lists, axis=0)
    fourcast_max = np.max(fourcast_pressure_lists, axis=0)

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True, facecolor='w')

    # Plot Pangu-Weather data with min and max shading
    ax.plot(time_arr_cut, pangu_min, label='Pangu-Weather Min', color='blue', linewidth=1)
    ax.plot(time_arr_cut, pangu_max, label='Pangu-Weather Max', color='blue', linewidth=1)
    ax.fill_between(time_arr_cut, pangu_min, pangu_max, color='blue', alpha=0.2)

    # Plot FourCastNet v2 data with min and max shading
    ax.plot(time_arr_cut, fourcast_min, label='FourCastNet v2 Min', color='red', linewidth=1)
    ax.plot(time_arr_cut, fourcast_max, label='FourCastNet v2 Max', color='red', linewidth=1)
    ax.fill_between(time_arr_cut, fourcast_min, fourcast_max, color='red', alpha=0.2)

    # Plot ERA5 data
    ax.plot(time_arr_cut, era_mins, label='ERA5', color='black', linewidth=3)

    # Contour dotted lines at every 20% interval
    for percentile in range(20, 100, 20):
        pangu_percentile_val = np.percentile(pangu_pressure_lists, percentile, axis=0)
        fourcast_percentile_val = np.percentile(fourcast_pressure_lists, percentile, axis=0)
        ax.plot(time_arr_cut, pangu_percentile_val, linestyle='--', color='blue', linewidth=0.5)
        ax.plot(time_arr_cut, fourcast_percentile_val, linestyle='--', color='red', linewidth=0.5)
        ax.text(time_arr_cut[-1], pangu_percentile_val[-1], f"{percentile}%", fontsize=8, color='blue')
        ax.text(time_arr_cut[-1], fourcast_percentile_val[-1], f"{percentile}%", fontsize=8, color='red')

    # Set plot title and labels
    ax.set_title("Minimum pressure over time", fontsize=14)
    ax.set_ylabel("Central Pressure (Pa)")
    ax.legend()

    fig.savefig("%s%s_%s_minp_density_contoured.png"%(save_path, start_day, year))

def minp_forecast_plot(pangu_preds, fourcast_preds, era_tracks, time_arr, save_path, start_day, year):
    time_arr_cut = time_arr

    # Extract pressure values from dictionaries for Pangu-Weather predictions
    pangu_pressure_lists = [pangu_preds[lead_time] for lead_time in sorted(pangu_preds.keys())]
    for ppreds in range(0, len(pangu_pressure_lists)):
        pangu_pressure_lists[ppreds] = [pdict['msl'] for pdict in pangu_pressure_lists[ppreds]]

    # Extract pressure values from dictionaries for FourCastNet v2 predictions
    fourcast_pressure_lists = [fourcast_preds[lead_time] for lead_time in sorted(fourcast_preds.keys())]
    for fpreds in range(0, len(fourcast_pressure_lists)):
        fourcast_pressure_lists[fpreds] = [fdict['msl'] for fdict in fourcast_pressure_lists[fpreds]]

    # Filter out lists with a length different from time_arr_cut
    pangu_pressure_lists = list(filter(lambda x: len(x) == len(time_arr_cut), pangu_pressure_lists))
    fourcast_pressure_lists = list(filter(lambda x: len(x) == len(time_arr_cut), fourcast_pressure_lists))

    # Calculate minimum pressure for ERA5
    era_mins = [track['msl'] for track in era_tracks]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True, facecolor='w')

    # Plot Pangu-Weather data
    for i, lead_time in enumerate(sorted(pangu_preds.keys())):
        if i % 2 == 0:  # Plot every other lead time
            pressure_list = pangu_preds[lead_time]
            #ax.plot(time_arr_cut, pangu_pressure_lists[i], label=f'Pangu-Weather, Lead Time: {lead_time} hr', color='blue', linewidth=1, linestyle='--')
            ax.plot(time_arr_cut, pangu_pressure_lists[i], color='blue', linewidth=1, linestyle='--')
            ax.text(time_arr_cut[0], pangu_pressure_lists[i][0], f'{lead_time} hr', fontsize=8, color='blue')
            ax.text(time_arr_cut[-1], pangu_pressure_lists[i][-1], f'{lead_time} hr', fontsize=8, color='blue')

    # Plot FourCastNet v2 data
    for j, lead_time in enumerate(sorted(fourcast_preds.keys())):
        if j % 2 == 0:  # Plot every other lead time
            pressure_list = fourcast_preds[lead_time]
            #ax.plot(time_arr_cut, fourcast_pressure_lists[j], label=f'FourCastNet v2, Lead Time: {lead_time} hr', color='red', linewidth=1, linestyle='--')
            ax.plot(time_arr_cut, fourcast_pressure_lists[j], color='red', linewidth=1, linestyle='--')
            ax.text(time_arr_cut[0], fourcast_pressure_lists[j][0], f'{lead_time} hr', fontsize=8, color='red')
            ax.text(time_arr_cut[-1], fourcast_pressure_lists[j][-1], f'{lead_time} hr', fontsize=8, color='red')

    # Plot ERA5 data
    ax.plot(time_arr_cut, era_mins, label='ERA5', color='black', linewidth=3)

    # Set plot title and labels
    ax.set_title("Minimum pressure over time", fontsize=14)
    ax.set_ylabel("Central Pressure (Pa)")

    # Legend with dotted lines
    ax.plot([], [], color='blue', linestyle='--', label='Pangu-Weather')
    ax.plot([], [], color='red', linestyle='--', label='FourCastNet v2')
    ax.legend()

    fig.savefig(f"{save_path}{start_day}_{year}_minp_forecast_plot.png")

def minp_shaded(pangu_preds, fourcast_preds, era_tracks, time_arr, save_path, start_day, year):
    min_time = min(time_arr)
    # Convert time_arr to numeric values representing the difference in hours from the minimum value
    time_arr_cut_numeric = np.array([(t - min_time).astype('timedelta64[h]').astype(int) for t in time_arr])

    # Extract pressure values from dictionaries for Pangu-Weather predictions
    pangu_pressure_lists = [pangu_preds[lead_time] for lead_time in sorted(pangu_preds.keys())]
    for ppreds in range(0, len(pangu_pressure_lists)):
        pangu_pressure_lists[ppreds] = [pdict['msl'] for pdict in pangu_pressure_lists[ppreds]]

    # Extract pressure values from dictionaries for FourCastNet v2 predictions
    fourcast_pressure_lists = [fourcast_preds[lead_time] for lead_time in sorted(fourcast_preds.keys())]
    for fpreds in range(0, len(fourcast_pressure_lists)):
        fourcast_pressure_lists[fpreds] = [fdict['msl'] for fdict in fourcast_pressure_lists[fpreds]]

    # Filter out lists with a length different from time_arr
    pangu_pressure_lists = list(filter(lambda x: len(x) == len(time_arr), pangu_pressure_lists))
    fourcast_pressure_lists = list(filter(lambda x: len(x) == len(time_arr), fourcast_pressure_lists))

    # Calculate minimum pressure for ERA5
    era_mins = [track['msl'] for track in era_tracks]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True, facecolor='w')

    # Plot Pangu-Weather data
    for i, lead_time in enumerate(sorted(pangu_preds.keys())):
        if i % 2 == 0:  # Plot every other lead time
            pressure_list = pangu_pressure_lists[i]
            color = plt.cm.Blues(1 - i / len(pangu_preds))  # Adjust color based on lead time index
            ax.plot(time_arr, pressure_list, color=color, linewidth=1, linestyle='--')
            ax.text(time_arr[0], pressure_list[0], f'{lead_time} hr', fontsize=8, color=color)
            ax.text(time_arr[-1], pressure_list[-1], f'{lead_time} hr', fontsize=8, color=color)

    # Plot FourCastNet v2 data
    for j, lead_time in enumerate(sorted(fourcast_preds.keys())):
        if j % 2 == 0:  # Plot every other lead time
            pressure_list = fourcast_pressure_lists[j]
            color = plt.cm.Reds(1 - j / len(fourcast_preds))  # Adjust color based on lead time index
            ax.plot(time_arr, pressure_list, color=color, linewidth=1, linestyle='--')
            ax.text(time_arr[0], pressure_list[0], f'{lead_time} hr', fontsize=8, color=color)
            ax.text(time_arr[-1], pressure_list[-1], f'{lead_time} hr', fontsize=8, color=color)

    # Plot ERA5 data
    ax.plot(time_arr, era_mins, label='ERA5', color='black', linewidth=3)

    # Set plot title and labels
    ax.set_title("Minimum pressure over time", fontsize=14)
    ax.set_ylabel("Central Pressure (Pa)")

    # Legend with dotted lines
    ax.plot([], [], color=plt.cm.Blues(0.5), linestyle='--', label='Pangu-Weather')
    ax.plot([], [], color=plt.cm.Reds(0.5), linestyle='--', label='FourCastNet v2')
    ax.legend()

    fig.savefig(f"{save_path}{start_day}_{year}_minp_shaded.png")

def cyclone_tracks_forecast_plot(pangu_preds, fourcast_preds, era_tracks, time_arr, save_path, start_day, year, domain, proj, time_slice):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 7), constrained_layout=True, subplot_kw={'projection': proj}, facecolor='w')
    plot_subset_background(ax, domain[0], domain[1], domain[2], domain[3])
    time_arr_cut = time_arr[(time_arr >= time_slice[0]) & (time_arr <= time_slice[1])]

    # Extract pressure values from dictionaries for Pangu-Weather predictions
    pangu_pressure_lists = [pangu_preds[lead_time] for lead_time in sorted(pangu_preds.keys())]
    for ppreds in range(0, len(pangu_pressure_lists)):
        pangu_pressure_lists[ppreds] = [[pdict['lon'], pdict['lat']] for pdict in pangu_pressure_lists[ppreds]]

    # Extract pressure values from dictionaries for FourCastNet v2 predictions
    fourcast_pressure_lists = [fourcast_preds[lead_time] for lead_time in sorted(fourcast_preds.keys())]
    for fpreds in range(0, len(fourcast_pressure_lists)):
        fourcast_pressure_lists[fpreds] = [[fdict['lon'], fdict['lat']] for fdict in fourcast_pressure_lists[fpreds]]

    # Calculate minimum pressure for ERA5
    era_track = [[track['lon'], track['lat']] for track in era_tracks]

    # Plot Pangu-Weather data
    for i, pressure_list in enumerate(pangu_pressure_lists):
        if i % 2 == 0:  # Plot every other lead time
            ax.plot([point[0] for point in pressure_list], [point[1] for point in pressure_list], color='blue', linestyle='--', linewidth=1, marker='o', transform=ccrs.PlateCarree())
            #ax.text(pressure_list[0][0], pressure_list[0][1], f'{sorted(pangu_preds.keys())[i]} hr', fontsize=8, color='blue')
            #ax.text(pressure_list[-1][0], pressure_list[-1][1], f'{sorted(pangu_preds.keys())[i]} hr', fontsize=8, color='blue')

    # Plot FourCastNet v2 data
    for j, pressure_list in enumerate(fourcast_pressure_lists):
        if j % 2 == 0:  # Plot every other lead time
            ax.plot([point[0] for point in pressure_list], [point[1] for point in pressure_list], color='red', linestyle='--', linewidth=1, marker='o', transform=ccrs.PlateCarree())
            #ax.text(pressure_list[0][0], pressure_list[0][1], f'{sorted(fourcast_preds.keys())[j]} hr', fontsize=8, color='red')
            #ax.text(pressure_list[-1][0], pressure_list[-1][1], f'{sorted(fourcast_preds.keys())[j]} hr', fontsize=8, color='red')

    ax.plot([point[0] for point in era_track], [point[1] for point in era_track], label='ERA5', color='black', linewidth=3)

    # Legend with model colors only
    ax.plot([], [], color='blue', linestyle='--', label='Pangu-Weather')
    ax.plot([], [], color='red', linestyle='--', label='FourCastNet v2')
    ax.legend()

    fig.savefig(f"{save_path}{start_day}_{year}_cyclone_tracks_forecast_plot.png")

def cyclone_tracks_contoured(pangu_preds, fourcast_preds, era_tracks, time_arr, save_path, start_day, year, domain, proj, time_slice):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 7), constrained_layout=True, subplot_kw={'projection': proj}, facecolor='w')
    plot_subset_background(ax, domain[0], domain[1], domain[2], domain[3])
    time_arr_cut = time_arr[(time_arr >= time_slice[0]) & (time_arr <= time_slice[1])]

    # Extract pressure values from dictionaries for Pangu-Weather predictions
    pangu_pressure_lists = [pangu_preds[lead_time] for lead_time in sorted(pangu_preds.keys())]
    for ppreds in range(0, len(pangu_pressure_lists)):
        pangu_pressure_lists[ppreds] = [[pdict['lon'], pdict['lat']] for pdict in pangu_pressure_lists[ppreds]]

    # Extract pressure values from dictionaries for FourCastNet v2 predictions
    fourcast_pressure_lists = [fourcast_preds[lead_time] for lead_time in sorted(fourcast_preds.keys())]
    for fpreds in range(0, len(fourcast_pressure_lists)):
        fourcast_pressure_lists[fpreds] = [[fdict['lon'], fdict['lat']] for fdict in fourcast_pressure_lists[fpreds]]

    # Calculate minimum pressure for ERA5
    era_track = [[track['lon'], track['lat']] for track in era_tracks]

    # Plot Pangu-Weather data
    for i, pressure_list in enumerate(pangu_pressure_lists[:2]):
        if i % 2 == 0:  # Plot every other lead time
            ax.plot([point[0] for point in pressure_list], [point[1] for point in pressure_list], color='blue', linestyle='--', linewidth=1, marker='o', transform=ccrs.PlateCarree())
            #ax.text(pressure_list[0][0], pressure_list[0][1], f'{sorted(pangu_preds.keys())[i]} hr', fontsize=8, color='blue')
            #ax.text(pressure_list[-1][0], pressure_list[-1][1], f'{sorted(pangu_preds.keys())[i]} hr', fontsize=8, color='blue')

    # Plot FourCastNet v2 data
    for j, pressure_list in enumerate(fourcast_pressure_lists[:2]):
        if j % 2 == 0:  # Plot every other lead time
            ax.plot([point[0] for point in pressure_list], [point[1] for point in pressure_list], color='red', linestyle='--', linewidth=1, marker='o', transform=ccrs.PlateCarree())
            #ax.text(pressure_list[0][0], pressure_list[0][1], f'{sorted(fourcast_preds.keys())[j]} hr', fontsize=8, color='red')
            #ax.text(pressure_list[-1][0], pressure_list[-1][1], f'{sorted(fourcast_preds.keys())[j]} hr', fontsize=8, color='red')

    # Plot ERA5 data
    ax.plot([point[0] for point in era_track], [point[1] for point in era_track], label='ERA5', color='black', linewidth=3)

    # Shade the area between the geospatial cones
    for p1, p2 in zip(pangu_pressure_lists[:2], fourcast_pressure_lists[:2]):
        lons1, lats1 = np.array(p1).T
        lons2, lats2 = np.array(p2).T
        ax.fill_betweenx(lats1, lons1, lons2, color='lightgrey', alpha=0.3)
        ax.fill_betweenx(lats2, lons1, lons2, color='lightgrey', alpha=0.3)

    # Legend with model colors only
    ax.plot([], [], color='blue', linestyle='--', label='Pangu-Weather')
    ax.plot([], [], color='red', linestyle='--', label='FourCastNet v2')
    ax.legend()

    fig.savefig(f"{save_path}{start_day}_{year}_cyclone_tracks_contoured.png")


def cyclone_tracks_shaded(pangu_preds, fourcast_preds, era_tracks, time_arr, save_path, start_day, year, domain, proj, time_slice):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 7), constrained_layout=True, subplot_kw={'projection': proj}, facecolor='w')
    plot_subset_background(ax, domain[0], domain[1], domain[2], domain[3])
    time_arr_cut = time_arr[(time_arr >= time_slice[0]) & (time_arr <= time_slice[1])]

    # Extract pressure values from dictionaries for Pangu-Weather predictions
    pangu_pressure_lists = [pangu_preds[lead_time] for lead_time in sorted(pangu_preds.keys())]
    for ppreds in range(0, len(pangu_pressure_lists)):
        pangu_pressure_lists[ppreds] = [[pdict['lon'], pdict['lat']] for pdict in pangu_pressure_lists[ppreds]]

    # Extract pressure values from dictionaries for FourCastNet v2 predictions
    fourcast_pressure_lists = [fourcast_preds[lead_time] for lead_time in sorted(fourcast_preds.keys())]
    for fpreds in range(0, len(fourcast_pressure_lists)):
        fourcast_pressure_lists[fpreds] = [[fdict['lon'], fdict['lat']] for fdict in fourcast_pressure_lists[fpreds]]

    # Calculate minimum pressure for ERA5
    era_track = [[track['lon'], track['lat']] for track in era_tracks]

    # Colormap setup
    cmap = plt.get_cmap('Greys')  # Adjust the colormap as per your preference
    norm = mcolors.Normalize(vmin=min(time_arr_cut), vmax=max(time_arr_cut))

    # Plot Pangu-Weather data
    for i, pressure_list in enumerate(pangu_pressure_lists):
        if i % 2 == 0:  # Plot every other lead time
            time_index = sorted(pangu_preds.keys())[i]
            color = cmap(norm(time_index))
            ax.plot([point[0] for point in pressure_list], [point[1] for point in pressure_list], color=color, linestyle='--', linewidth=1, marker='o', transform=ccrs.PlateCarree())
            #ax.text(pressure_list[0][0], pressure_list[0][1], f'{time_index} hr', fontsize=8, color=color)
            #ax.text(pressure_list[-1][0], pressure_list[-1][1], f'{time_index} hr', fontsize=8, color=color)

    # Plot FourCastNet v2 data
    for j, pressure_list in enumerate(fourcast_pressure_lists):
        if j % 2 == 0:  # Plot every other lead time
            time_index = sorted(fourcast_preds.keys())[j]
            color = cmap(norm(time_index))
            ax.plot([point[0] for point in pressure_list], [point[1] for point in pressure_list], color=color, linestyle='--', linewidth=1, marker='o', transform=ccrs.PlateCarree())
            #ax.text(pressure_list[0][0], pressure_list[0][1], f'{time_index} hr', fontsize=8, color=color)
            #ax.text(pressure_list[-1][0], pressure_list[-1][1], f'{time_index} hr', fontsize=8, color=color)

    ax.plot([point[0] for point in era_track], [point[1] for point in era_track], label='ERA5', color='black', linewidth=3)

    # Legend with model colors only
    ax.plot([], [], color='blue', linestyle='--', label='Pangu-Weather')
    ax.plot([], [], color='red', linestyle='--', label='FourCastNet v2')
    ax.legend()

    # Save the figure
    fig.savefig(f"{save_path}{start_day}_{year}_cyclone_tracks_shaded.png")

#-------------------------------------------------------------------------------------------------------
# CYCLONE FUNCTIONS
#-------------------------------------------------------------------------------------------------------
def bubblesort_files(file_arr):
    file_array = []
    for f in file_arr:
        file_array.append([int(f.split('_')[3]),f])

    for i in range(0,len(file_array)-1):
        swap = False
        for j in range(0,len(file_array)-1):
            if file_array[j][0] > file_array[j+1][0]:
                temp = file_array[j]
                file_array[j] = file_array[j+1]
                file_array[j+1] = temp
                swap = True
        if swap == False:
            break

    file_array = [file[1] for file in file_array]
    return file_array

def make_cyclone_dict(model_subset,time):
    minp = np.nanmin(model_subset)
    min_indices = np.unravel_index(np.nanargmin(model_subset), model_subset.shape)
    lon_values = model_subset['longitude'].values[min_indices[1]]
    lat_values = model_subset['latitude'].values[min_indices[0]]
    cyclone_dict = {'lon': lon_values, 'lat': lat_values, 'msl': minp, 'time': time}
    return cyclone_dict

def dict_to_dataframe(model_dict):
    df = pd.DataFrame()
    time_list = [model_dict[lead_time] for lead_time in sorted(model_dict.keys())]
    for time in range(0, len(time_list)):
        lons = [pred['lon'] for pred in time_list[time]]
        lats = [pred['lat'] for pred in time_list[time]]
        ps = [pred['msl'] for pred in time_list[time]]
        ts = [pred['time'] for pred in time_list[time]]
        df.append({'lead_time' : time, 'lons' : lons, 'lats' : lats, 'msl' : ps, 'times' : ts}, ignore_index=True)
    return df

def load_settings_from_file(file_path):
    settings = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            key, value = line.split('=')  # Split key and value
            # Check if the value is a list
            if '[' in value and ']' in value:
                value = value.replace('[', '').replace(']', '')  # Remove brackets
                value = [int(x.strip()) for x in value.split(',')]  # Convert to list of integers
            # Check if the key is HHMMSS_start or HHMMSS_end
            elif key in ['HHMMSS_start', 'HHMMSS_end']:
                # Format value with six zeros
                value = value.zfill(6)
            # Convert value to int if it represents an integer, otherwise keep it as string
            settings[key] = value
    return settings

if __name__ == "__main__":
    # Manual loading in files:

    #year = '2021'
    #start_day = '1024'
    #HHMMSS_start = "000000"
    #end_day = '1025'
    #HHMMSS_end = "000000"
    #domain=[200,250,35,60]
    #crs_use = crs_pnw

    # Loading in from file
    settings = load_settings_from_file('settings/1024_2021_settings.txt')
    #crs_use = crs_global_atlantic
    crs_use = crs_pnw

    # Extract settings
    year = str(settings['year'])
    start_day = str(settings['start_day'])
    HHMMSS_start = str(settings['HHMMSS_start'])
    end_day = str(settings['end_day'])
    HHMMSS_end = str(settings['HHMMSS_end'])
    domain = settings['domain']

    # I'm not sure how much this will actually be used, however it may be a good thing to have
    timeframe = (pd.to_datetime("%s-%s-%s %s:%s:%s"%(year,start_day[0:2],start_day[2:],HHMMSS_start[0:2],HHMMSS_start[2:4],HHMMSS_start[4:])),
                pd.to_datetime("%s-%s-%s %s:%s:%s"%(year,end_day[0:2],end_day[2:],HHMMSS_end[0:2],HHMMSS_end[2:4],HHMMSS_end[4:])))

    # This is how we will actually slice the different datasets, it goes a day on either side of the event
    time_slice = (
    pd.to_datetime("%s-%s-%s %s:%s:%s" % (year, str(int(start_day)-1)[:len(str(int(start_day)))-2].zfill(2), str(int(start_day)-1)[len(str(int(start_day)))-2:].zfill(2), HHMMSS_start[0:2], HHMMSS_start[2:4], HHMMSS_start[4:])),
    pd.to_datetime("%s-%s-%s %s:%s:%s" % (year, str(int(end_day)+1)[:len(str(int(start_day)))-2].zfill(2), str(int(end_day)+1)[len(str(int(end_day)))-2:].zfill(2), HHMMSS_end[0:2], HHMMSS_end[2:4], HHMMSS_end[4:]))
    )
    print("Timeframe: " , timeframe)

    start_files = str(int(start_day)-10)

    # Beginning the dataflow 

    era = xr.open_dataset('%s/sfc_mslp_6hr_%s.nc'%('ERA5',year))
    era = era.sel(time=slice(time_slice[0],time_slice[1]))
    era['msl'] /= 100

    time_arr = era['time'].to_numpy() # This will mostly be used in plots
    print("Times included in this forecast: %s\n"%(str(len(time_arr))) , time_arr)

    print("ERA5 dataset shape: " , np.shape(era['time'].to_numpy()))

    # Pulling the files of the various datastreams
    pangu_files = []
    fourcast_files = []
    for dirs in os.listdir("%s/PANG_v100/%s/"%(root_data_path,year)):
        if int(dirs) >= int(start_files) and int(dirs) <= int(start_day):
            pangu_files.extend(os.listdir("%s/PANG_v100/%s/%s" % (root_data_path, year, dirs)))
            fourcast_files.extend(os.listdir("%s/FOUR_v200/%s/%s"%(root_data_path,year,dirs)))

    # Now we need to sort the file lists by file datetime such that the files are in chronological order (furthest forecast time up to event)
    pangu_files = bubblesort_files(pangu_files)
    fourcast_files = bubblesort_files(fourcast_files)

    # Here I am going to check if the last file is after the time of the start of the event. If so, remove from the dataset
    temp_ds = xr.open_dataset("%s/PANG_v100/%s/%s/%s"%(root_data_path,year,pangu_files[len(pangu_files)-1].split('_')[3][4:8],pangu_files[len(pangu_files)-1]))
    if temp_ds['time'].isel(time=0) > timeframe[0]:
        pangu_files.pop()
    temp_ds = xr.open_dataset("%s/FOUR_v200/%s/%s/%s"%(root_data_path,year,fourcast_files[len(fourcast_files)-1].split('_')[3][4:8],fourcast_files[len(fourcast_files)-1]))
    if temp_ds['time'].isel(time=0) > timeframe[0]:
        fourcast_files.pop()

    #print("\nFiles: \n")
    #print(pangu_files)
    #print(fourcast_files)

    # This will be used to go file by file and extract the necessary data
    pangu_preds = {}
    fourcast_preds = {}
    era_tracks = []
    for f in range(0,len(pangu_files)):
        try:
            print("Current files: \t" +pangu_files[f]+"\t"+fourcast_files[f])
            # Loading data
            pangu_weather = xr.open_dataset("%s/PANG_v100/%s/%s/%s"%(root_data_path,year,pangu_files[f].split('_')[3][4:8],pangu_files[f]))
            fourcast = xr.open_dataset("%s/FOUR_v200/%s/%s/%s"%(root_data_path,year,fourcast_files[f].split('_')[3][4:8],fourcast_files[f]))

            # Slicing files
            pangu_weather = pangu_weather.sel(time=slice(time_slice[0],time_slice[1]))
            fourcast = fourcast.sel(time=slice(time_slice[0],time_slice[1]))

            # Converting the data from Pa to hPa
            pangu_weather['msl'] /=100
            fourcast['msl'] /=100

            # Starting to assemble each dictionary
            file_time = pd.to_datetime("%s-%s-%s %s:%s:%s"%(pangu_files[f].split('_')[3][0:4],
                                                            pangu_files[f].split('_')[3][4:6],
                                                            pangu_files[f].split('_')[3][6:8],
                                                            pangu_files[f].split('_')[3][8:],"00","00"))
            time_delta = (timeframe[0] - file_time).total_seconds()/3600 # Forecast lead time (hr)

            p_cyclone_tracks = []
            f_cyclone_tracks = []
            broke_flag = False
            for time in time_arr:
                #print("Current time: " , time)
                try:
                    pangu_subset = pangu_weather['msl'].sel(time=time).sel(longitude=slice(domain[0], domain[1]), latitude=slice(domain[3], domain[2]))
                    p_cyclone_tracks.append(make_cyclone_dict(pangu_subset,time))

                    four_subset = fourcast['msl'].sel(time=time).sel(longitude=slice(domain[0], domain[1]), latitude=slice(domain[3], domain[2]))
                    f_cyclone_tracks.append(make_cyclone_dict(four_subset,time))
                    #if f == 0:
                    #    era_subset = era['msl'].sel(time=time).sel(longitude=slice(domain[0], domain[1]), latitude=slice(domain[3], domain[2]))
                    #    era_tracks.append(make_cyclone_dict(era_subset,time))
                except KeyError:
                    broke_flag = True
                    break
            if broke_flag == False:
                pangu_preds[time_delta] = p_cyclone_tracks
                fourcast_preds[time_delta] = f_cyclone_tracks
        except IndexError:
            pass
    era_tracks = [make_cyclone_dict(era['msl'].sel(time=t).sel(longitude=slice(domain[0], domain[1]), latitude=slice(domain[3], domain[2])),t) for t in time_arr]
    print("ERA5 tracks: " , np.shape(era_tracks))
    print("Pangu_preds length: " , len(pangu_preds))
    print("Fourcast_preds: " , len(fourcast_preds))
    minp_density_plot(pangu_preds,fourcast_preds,era_tracks,time_arr,str(root_save_path + "%s_%s/"%(start_day,year)),start_day,year)
    minp_density_plot_contoured(pangu_preds,fourcast_preds,era_tracks,time_arr,str(root_save_path + "%s_%s/"%(start_day,year)),start_day,year)
    minp_forecast_plot(pangu_preds,fourcast_preds,era_tracks,time_arr,str(root_save_path + "%s_%s/"%(start_day,year)),start_day,year)
    minp_shaded(pangu_preds,fourcast_preds,era_tracks,time_arr,str(root_save_path + "%s_%s/"%(start_day,year)),start_day,year)

    cyclone_tracks_forecast_plot(pangu_preds, fourcast_preds, era_tracks, time_arr, str(root_save_path + "%s_%s/"%(start_day,year)), start_day, year, domain, crs_pnw, time_slice)
    cyclone_tracks_contoured(pangu_preds, fourcast_preds, era_tracks, time_arr, str(root_save_path + "%s_%s/"%(start_day,year)), start_day, year, domain, crs_pnw, time_slice)
    cyclone_tracks_shaded(pangu_preds, fourcast_preds, era_tracks, time_arr, str(root_save_path + "%s_%s/"%(start_day,year)), start_day, year, domain, crs_pnw, time_slice)

    # Save the data for future use
    pangu_df = dict_to_dataframe(pangu_preds)
    fourcast_df = dict_to_dataframe(fourcast_preds)

    print("\n Pangu dataframe: \n", pangu_df)
    print("\n Fourcastnet dataframe: \n", fourcast_df)
    pangu_df.to_csv("%s%s_%s/%s_%s_pangu_cyclones.csv"%(root_save_path,start_day,year,start_day,year))
    fourcast_df.to_csv("%s%s_%s/%s_%s_fourcast_cyclones.csv"%(root_save_path,start_day,year,start_day,year))
