# Data manipulation
import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
# Plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# General
import sys,os 

#----------------------------------------------------------------------------------------------------------------
# Constants
#----------------------------------------------------------------------------------------------------------------
# Local
root_path = '/mnt/d/VS-workspace/AI2ES/AI_NWP/'
data_root_path = '%smodel_data/noaa-oar-mlwp-data/'%(root_path)
# Schooner
#root_path = '/ourdisk/hpc/ai2es/alexnozka/'
#data_root_path = '/ourdisk/hpc/ai2es/datasets/noaa-oar-mlwp-data/'

# Plotting
crs_global_atlantic = ccrs.PlateCarree(central_longitude=0.0)
crs_global_pacific = ccrs.PlateCarree(central_longitude=180.0)

#----------------------------------------------------------------------------------------------------------------
# Cyclone Tracking Methods
#----------------------------------------------------------------------------------------------------------------

def calculate_laplacian(data):
    """
    Calculate the Laplacian of the 2D data array.
    """
    dx, dy = np.gradient(data)
    laplacian = np.gradient(dx, axis=0) + np.gradient(dy, axis=1)
    return laplacian

def calculate_background_pressure(latitude):
    """
    Calculate the climatological background pressure based on latitude.
    This is converted to Pa since that is what the data is in
    """
    background_pressure = 101325.0 - (10.0 * latitude)
    
    return background_pressure


def identify_cyclone_centers(mslp, laplacian_threshold):
    """
    Identify potential cyclone centers based on the Laplacian of MSLP.
    """
    laplacian = calculate_laplacian(mslp)
    potential_centers = np.argwhere(laplacian < -laplacian_threshold)
    return potential_centers

def identify_cyclone_centers_minp(mslp, min_pressure_diff, search_radius):
    """
    Identify potential cyclone centers based on the minimum pressure criterion.
    """
    centers = []
    mslp_padded = np.pad(mslp, search_radius, mode='edge')  # Pad the MSLP field to handle edge cases
    
    for i in range(search_radius, mslp.shape[0] + search_radius):
        for j in range(search_radius, mslp.shape[1] + search_radius):
            center_pressure = mslp_padded[i, j]
            min_pressure = np.min(mslp_padded[i - search_radius:i + search_radius + 1,
                                              j - search_radius:j + search_radius + 1])
            if center_pressure - min_pressure >= min_pressure_diff:
                centers.append((i - search_radius, j - search_radius))
    
    return centers

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the globe.
    """
    # Convert latitudes and longitudes to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Calculate central angle using haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Calculate distance
    distance = 6371e3 * c  # Earth radius in meters
    return distance

def track_cyclones_over_time(nc_file, laplacian_threshold, search_radius, min_pressure_threshold):
    """
    Track cyclone centers over multiple time steps.
    """
    # Open netCDF file
    nc_data = nc.Dataset(nc_file, 'r')
    
    # Initialize dictionary to store cyclones by ID
    tracked_cyclones = {}
    
    # Track cyclones for each time step
    for t, mslp in enumerate(nc_data.variables['msl']):
        # Identify cyclone centers for the current time step
        potential_centers = identify_cyclone_centers(mslp, laplacian_threshold)
        
        # Filter potential cyclone centers based on minimum pressure threshold
        potential_centers_filtered = [(center_y, center_x) for center_y, center_x in potential_centers
                                      if mslp[center_y, center_x] < min_pressure_threshold]
        
        # Create a new dictionary to store cyclones for the current timestep
        cyclones_at_timestep = {}
        
        # Track cyclones using centroid tracking
        for center_y, center_x in potential_centers_filtered:
            # Define a search area around the potential center
            search_area = mslp[max(0, center_y - search_radius):min(mslp.shape[0], center_y + search_radius + 1),
                               max(0, center_x - search_radius):min(mslp.shape[1], center_x + search_radius + 1)]
            min_pressure_y, min_pressure_x = np.unravel_index(np.argmin(search_area), search_area.shape)
            lat = min_pressure_y + max(0, center_y - search_radius)
            lon = min_pressure_x + max(0, center_x - search_radius)
            pressure = mslp[lat, lon]
            
            # Check proximity to existing cyclones
            found_existing_cyclone = False
            for id, steps in tracked_cyclones.items():
                if steps:  # Check if steps is not empty
                    prev_lat, prev_lon, prev_t, prev_pressure = steps[-1]
                    dist = calculate_distance(lat, lon, prev_lat, prev_lon)
                    if dist < search_radius or dist > 180:  # Account for wrapping around the globe
                        # Check for cyclolysis
                        if prev_t == t - 1 and (prev_t not in tracked_cyclones or not any(c[2] == t for c in tracked_cyclones[prev_t])):
                            found_existing_cyclone = True
                            break
                        else:
                            # Update existing cyclone with new center
                            tracked_cyclones[id].append((lat, lon, t, pressure))
                            found_existing_cyclone = True
                            break
            
            # If cyclone is not close to any existing cyclone and meets the minimum pressure threshold, create a new one
            if not found_existing_cyclone and pressure < min_pressure_threshold:
                tracked_cyclones[len(tracked_cyclones)] = [(lat, lon, t, pressure)]
    
    # Close netCDF file
    nc_data.close()
    
    return tracked_cyclones


def track_cyclones_over_time_wbp(nc_file, laplacian_threshold, search_radius):
    """
    Track cyclone centers over multiple time steps.
    "With Background Pressure (WBP)
    """
    # Open netCDF file
    nc_data = nc.Dataset(nc_file, 'r')
    
    # Convert mslp to a NumPy array if it's not already
    mslp_data = np.array(nc_data.variables['msl'][:])
    
    # Initialize dictionary to store cyclones by ID
    tracked_cyclones = {}
    
    # Track cyclones for each time step
    for t, mslp in enumerate(mslp_data):
        # Identify cyclone centers for the current time step
        potential_centers = identify_cyclone_centers(mslp, laplacian_threshold)
        
        # Create a new dictionary to store cyclones for the current timestep
        cyclones_at_timestep = {}
        
        # Track cyclones using centroid tracking
        for center_y, center_x in potential_centers:
            # Define a search area around the potential center
            min_y = max(0, center_y - search_radius)
            max_y = min(mslp_data.shape[1], center_y + search_radius + 1)
            min_x = max(0, center_x - search_radius)
            max_x = min(mslp_data.shape[2], center_x + search_radius + 1)
            
            search_area = mslp[min_y:max_y, min_x:max_x]
            min_pressure_y, min_pressure_x = np.unravel_index(np.argmin(search_area), search_area.shape)
            lat = min_pressure_y + min_y
            lon = min_pressure_x + min_x
            pressure = mslp[lat, lon]
            
            # Calculate the climatological background pressure based on latitude
            background_pressure = calculate_background_pressure(lat)
            
            # Check if the cyclone center pressure is lower than the climatological background pressure
            if pressure < background_pressure:
                # Check proximity to existing cyclones
                found_existing_cyclone = False
                for id, steps in tracked_cyclones.items():
                    if steps:  # Check if steps is not empty
                        prev_lat, prev_lon, prev_t, prev_pressure = steps[-1]
                        dist = calculate_distance(lat, lon, prev_lat, prev_lon)
                        if dist < search_radius or dist > 180:  # Account for wrapping around the globe
                            # Check for cyclolysis
                            if prev_t == t - 1:  # Cyclolysis occurs when the cyclone was present in the previous timestep
                                found_existing_cyclone = True
                                break
                            else:
                                # Update existing cyclone with new center
                                tracked_cyclones[id].append((lat, lon, t, pressure))
                        else:
                            found_existing_cyclone = True
                            break
                
                # If cyclone is not close to any existing cyclone and it's not a cyclolysis event, create a new one
                if not found_existing_cyclone:
                    tracked_cyclones[len(tracked_cyclones)] = [(lat, lon, t, pressure)]
    
    # Close netCDF file
    nc_data.close()
    
    return tracked_cyclones

def track_cyclones_over_time_minp(nc_file, min_pressure_diff, min_pressure_depth, search_radius):
    """
    Track cyclone centers over multiple time steps based on minimum pressure criterion.
    """
    # Open netCDF file
    nc_data = nc.Dataset(nc_file, 'r')
    
    # Initialize dictionary to store cyclones by ID
    tracked_cyclones = {}
    
    # Track cyclones for each time step
    for t, mslp in enumerate(nc_data.variables['msl']):
        # Convert mslp to a NumPy array if it's not already
        mslp = np.array(mslp)
        
        # Identify cyclone centers for the current time step
        potential_centers = identify_cyclone_centers_minp(mslp, min_pressure_diff, search_radius)
        
        # Precompute the search area for all potential cyclone centers
        padded_mslp = np.pad(mslp, search_radius, mode='edge')
        lat_indices = np.array([center[0] for center in potential_centers]) + search_radius
        lon_indices = np.array([center[1] for center in potential_centers]) + search_radius
        lat_indices = np.array(lat_indices, dtype=int)
        lon_indices = np.array(lon_indices, dtype=int)

        padded_mslp = np.pad(mslp, search_radius, mode='edge')
        lat_lower = np.maximum(0, lat_indices - search_radius)
        lat_upper = np.minimum(padded_mslp.shape[0], lat_indices + search_radius + 1)
        lon_lower = np.maximum(0, lon_indices - search_radius)
        lon_upper = np.minimum(padded_mslp.shape[1], lon_indices + search_radius + 1)

        #search_areas = padded_mslp[lat_indices[:, None] - search_radius: lat_indices[:, None] + search_radius + 1,
        #                           lon_indices[:, None] - search_radius: lon_indices[:, None] + search_radius + 1]
        search_areas = [padded_mslp[lat_lower[i]:lat_upper[i], :] for i in range(len(lat_indices))]
        
        # Track cyclones using centroid tracking
        for i, (center_y, center_x) in enumerate(potential_centers):
            lat, lon = center_y, center_x
            pressure = mslp[lat, lon]
            search_area = search_areas[i]
            
            # Check if the cyclone center pressure meets the criteria
            if (pressure - np.min(search_area)) >= min_pressure_diff or pressure <= min_pressure_depth:
                # Check proximity to existing cyclones
                found_existing_cyclone = False
                for id, steps in tracked_cyclones.items():
                    if steps:  # Check if steps is not empty
                        prev_lat, prev_lon, prev_t, prev_pressure = steps[-1]
                        dist = calculate_distance(lat, lon, prev_lat, prev_lon)
                        if dist < search_radius or dist > 180:  # Account for wrapping around the globe
                            # Check for cyclolysis
                            if prev_t == t - 1:  # Cyclolysis occurs when the cyclone was present in the previous timestep
                                found_existing_cyclone = True
                                break
                            else:
                                # Update existing cyclone with new center
                                tracked_cyclones[id].append((lat, lon, t, pressure))
                        else:
                            found_existing_cyclone = True
                            break
                
                # If cyclone is not close to any existing cyclone and it's not a cyclolysis event, create a new one
                if not found_existing_cyclone:
                    tracked_cyclones[len(tracked_cyclones)] = [(lat, lon, t, pressure)]
    
    # Close netCDF file
    nc_data.close()
    
    return tracked_cyclones

def identify_cyclone_center_by_min(mslp, search_radius):
    """
    Identify cyclone center by finding the deepest minimum point within the specified radius.
    """
    # Calculate gradient magnitude
    grad_y, grad_x = np.gradient(mslp)
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)
    
    # Apply Gaussian smoothing to gradient magnitude
    grad_mag_smoothed = gaussian_filter(grad_mag, sigma=search_radius)
    
    # Find the location of the deepest minimum within the specified radius
    min_y, min_x = np.unravel_index(np.argmin(grad_mag_smoothed), grad_mag_smoothed.shape)
    
    return min_y, min_x

def identify_cyclone_centers_by_min(nc_file, search_radius):
    """
    Identify cyclone centers over multiple time steps by finding the deepest minimum point within the specified radius.
    """
    # Open netCDF file
    nc_data = nc.Dataset(nc_file, 'r')
    
    # Initialize list to store cyclone centers
    cyclone_centers = []
    
    # Iterate over time steps
    for mslp in nc_data.variables['msl']:
        # Convert mslp to a NumPy array if it's not already
        mslp = np.array(mslp)
        
        # Identify cyclone center for the current time step
        center_y, center_x = identify_cyclone_center_by_min(mslp, search_radius)
        
        # Append cyclone center to the list
        cyclone_centers.append((int(center_y), int(center_x)))  # Ensure integer values

    # Ensure cyclone_centers is a list of tuples
    if not isinstance(cyclone_centers, list):
        cyclone_centers = [cyclone_centers]
        
    # Close netCDF file
    nc_data.close()
    
    return cyclone_centers

def track_cyclone_by_min(nc_file, min_gradient_search_radius, cyclone_linking_radius, min_pressure_threshold):
    """
    Track cyclone centers over multiple time steps by finding the deepest minimum point and linking cyclones within a specified radius.
    """
    # Identify cyclone centers over multiple time steps
    cyclone_centers = identify_cyclone_centers_by_min(nc_file, min_gradient_search_radius)
    
    # Initialize dictionary to store cyclones by ID
    tracked_cyclones = {}
    
    # Open netCDF file
    nc_data = nc.Dataset(nc_file, 'r')
    
    # Track cyclones for each time step
    for t, mslp in enumerate(nc_data.variables['msl']):
        # Convert mslp to a NumPy array if it's not already
        mslp = np.array(mslp)
        
        # Identify cyclone centers for the current time step
        potential_centers = cyclone_centers[t]
        print("Potential Centers: \n" , potential_centers)
        
        # Filter potential cyclone centers based on minimum pressure threshold
        potential_centers_filtered = [(center[0], center[1]) for center in potential_centers
                                      if mslp[center[0], center[1]] < min_pressure_threshold]
        
        # Create a new dictionary to store cyclones for the current timestep
        cyclones_at_timestep = {}
        
        # Track cyclones using centroid tracking
        for center_y, center_x in potential_centers_filtered:
            # Check proximity to existing cyclones
            found_existing_cyclone = False
            for id, steps in tracked_cyclones.items():
                if steps:  # Check if steps is not empty
                    prev_lat, prev_lon, prev_t, prev_pressure = steps[-1]
                    dist = calculate_distance(center_y, center_x, prev_lat, prev_lon)
                    if dist < cyclone_linking_radius or dist > 180:  # Account for wrapping around the globe
                        # Update existing cyclone with new center
                        tracked_cyclones[id].append((center_y, center_x, t, mslp[center_y, center_x]))
                        found_existing_cyclone = True
                        break
            
            # If cyclone is not close to any existing cyclone, create a new one
            if not found_existing_cyclone:
                tracked_cyclones[len(tracked_cyclones)] = [(center_y, center_x, t, mslp[center_y, center_x])]
    
    # Close netCDF file
    nc_data.close()
    
    return tracked_cyclones

def save_cyclones_to_file(tracked_cyclones, output_file):
    """
    Save tracked cyclones to a CSV file using pandas DataFrame.
    """
    fieldnames = ['Cyclone ID', 'Time Step', 'Latitude', 'Longitude', 'Pressure']
    cyclone_data = []

    for cyclone_id, steps in tracked_cyclones.items():
        for t, (lat, lon, time_step, pressure) in enumerate(steps):
            cyclone_data.append([cyclone_id, time_step, lat, lon, pressure])

    df = pd.DataFrame(cyclone_data, columns=fieldnames)
    df.to_csv(output_file, index=False)
    return df

def read_cyclones_from_file(file_path):
    """
    Read cyclone data from a CSV file and organize it into a DataFrame.
    Each entry is grouped by cyclone ID, and latitude, longitude, time, and pressure
    are stored as lists for each cyclone object.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Group data by Cyclone ID and aggregate lat, lon, time, and pressure as lists
    grouped_df = df.groupby('Cyclone ID').agg({'Latitude': list, 'Longitude': list, 
                                               'Time Step': list, 'Pressure': list})
    
    return grouped_df

#----------------------------------------------------------------------------------------------------------------
# Plotting Methods
#----------------------------------------------------------------------------------------------------------------

def plot_background_no_lims(ax):
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    return ax

def plot_cyclones_on_map(cyclones_df, projection, plot_name):
    """
    Plot cyclone tracks on a map with color-coded central pressure.
    """
    # Create a figure and axis with specified projection
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection=projection), facecolor='w', constrained_layout=True)

    # Plot background features
    ax = plot_background_no_lims(ax)

    # Determine color range for pressure
    min_pressure = cyclones_df['Pressure'].min()
    max_pressure = cyclones_df['Pressure'].max()
    min_pressure = np.nanmin(min_pressure)
    max_pressure = np.nanmax(max_pressure)
    pressure_range = max_pressure - min_pressure

    # Loop through cyclones
    for cyclone_id, row in cyclones_df.iterrows():
        lats = row['Latitude']
        lons = row['Longitude']
        times = row['Time Step']
        pressures = row['Pressure']
        
        # Plot cyclone track with color-coded pressure
        for i in range(len(lats) - 1):
            pressure_norm = (pressures[i] - min_pressure) / pressure_range
            color = plt.cm.viridis(pressure_norm)  # Use Viridis colormap for pressure
            # Handle wrapping around the edges
            if np.abs(lons[i+1] - lons[i]) > 180:
                continue  # Skip plotting this segment if it crosses the 180th meridian
            ax.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]], color=color, transform=projection)
            ax.plot(lons[i], lats[i], marker='o', markersize=4, color=color, transform=projection)
            #ax.text(lons[i], lats[i], f"{times[i]}h\n{pressures[i]} hPa", color='black', fontsize=8, ha='left', va='center', transform=projection)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min_pressure, vmax=max_pressure))
    sm._A = []  # Fake up the array of scalar values
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Central Pressure (Pa)')

    # Set latitude extent to include both hemispheres
    ax.set_extent([-180, 180, -90, 90], crs=projection)

    # Set plot title
    ax.set_title('Cyclone Tracks (Color-coded by Pressure)')

    # Save plot
    fig.savefig(plot_name)

    plt.close()
        

def plot_cyclones_filt_time(cyclones_df, projection, plot_name, timesteps, color_tracks=False):
    """
    Plot cyclone tracks on a map with color-coded central pressure within the specified timeframe.
    """
    # Filter cyclones based on specified time steps
    filtered_cyclones = cyclones_df[cyclones_df['Time Step'].apply(lambda x: (isinstance(x, list) and any(step >= timesteps[0] and step <= timesteps[1] for step in x)) or (isinstance(x, int) and x >= timesteps[0] and x <= timesteps[1]))]

    # Create a figure and axis with specified projection
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection=projection), facecolor='w', constrained_layout=True)

    # Plot background features
    ax = plot_background_no_lims(ax)

    # Determine color range for pressure
    min_pressure = cyclones_df['Pressure'].min()
    max_pressure = cyclones_df['Pressure'].max()
    min_pressure = np.nanmin(min_pressure)
    max_pressure = np.nanmax(max_pressure)
    pressure_range = max_pressure - min_pressure

    # Loop through cyclones
    for cyclone_id, row in filtered_cyclones.iterrows():
        lats = row['Latitude']
        lons = row['Longitude']
        pressures = row['Pressure']

        # Ensure lats, lons, and pressures are lists
        if not isinstance(lats, list):
            lats = [lats]
            lons = [lons]
            pressures = [pressures]

        # Plot cyclone track with color-coded pressure
        for i in range(len(lats) - 1):
            pressure_norm = (pressures[i] - min_pressure) / pressure_range
            color = plt.cm.viridis(pressure_norm)  # Use Viridis colormap for pressure
            # Handle wrapping around the edges
            if np.abs(lons[i+1] - lons[i]) > 180:
                continue  # Skip plotting this segment if it crosses the 180th meridian
            if color_tracks:
                cyclone_color = plt.cm.tab20(cyclone_id % 20)  # Assign a unique color to each cyclone ID
                ax.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]], color=cyclone_color, transform=projection)
            else:
                ax.plot([lons[i], lons[i+1]], [lats[i], lats[i+1]], color=color, transform=projection)
            ax.plot(lons[i], lats[i], marker='o', markersize=4, color=color, transform=projection)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min_pressure, vmax=max_pressure))
    sm._A = []  # Fake up the array of scalar values
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Central Pressure (Pa)')

    # Set latitude extent to include both hemispheres
    ax.set_extent([-180, 180, -90, 90], crs=projection)

    # Set plot title
    ax.set_title('Cyclone Tracks (Color-coded by Pressure)')

    # Save plot
    fig.savefig(plot_name)

    plt.close()


#----------------------------------------------------------------------------------------------------------------
# Parameter Tuning
#----------------------------------------------------------------------------------------------------------------
    
def tune_lagrangian_search(nc_file, save_path, base_file_name, base_plot_name, radius, laplace = None, min_pressure = None, pressure_dif = None):
    if laplace is None:
        laplace = ['NA']
    if min_pressure is None:
        min_pressure = ['NA']
    if pressure_dif is None:
        pressure_dif = ['NA']
    for lap in laplace:
        for rad in radius:
            for p in min_pressure:
                for pdif in pressure_dif:
                    print("Laplacian Threshold: " + str(lap))
                    print("Search Radius: " + str(rad))
                    print("Minimum Pressure: " + str(p))
                    print("Pressure Differential Threshold: " + str(pdif))
                    if 'minp' in base_file_name:
                        output_file = "%sminp/rad%01d_minp%04d_pdifferential%03d_%s"%(save_path,rad,p,pdif,base_file_name)
                        plot_name = "%sminp/rad%01d_minp%04d_pdifferential%03d_%s"%(save_path,rad,p,pdif,base_plot_name)
                        tracked_cyclones = track_cyclones_over_time_minp(nc_file, pdif, p, search_radius)
                    elif 'wbp' in base_file_name:
                        output_file = "%swbp/lap%03d_rad%01d_%s"%(save_path,lap,rad,base_file_name)
                        plot_name = "%swbp/lap%03d_rad%01d_%s"%(save_path,lap,rad,base_plot_name)
                        obj_plot_name = "%sdefault/lap%03d_rad%01d_minp%04d_0_10_%s"%(save_path,lap,rad,p,base_plot_name)
                        tracked_cyclones = track_cyclones_over_time_wbp(nc_file, laplacian_threshold, search_radius)
                    else:
                        output_file = "%sdefault/lap%03d_rad%01d_minp%04d_%s"%(save_path,lap,rad,p,base_file_name)
                        plot_name = "%sdefault/lap%03d_rad%01d_minp%04d_%s"%(save_path,lap,rad,p,base_plot_name)
                        obj_plot_name = "%sdefault/lap%03d_rad%01d_minp%04d_0_10_%s"%(save_path,lap,rad,p,base_plot_name)
                        tracked_cyclones = track_cyclones_over_time(nc_file, laplacian_threshold, search_radius, min_pressure_threshold)
                    save_cyclones_to_file(tracked_cyclones, output_file)
                    # Uploading file saved
                    file_path = output_file
                    cyclones_df = read_cyclones_from_file(file_path)
                    print(cyclones_df.head())
                    # Plotting
                    #plot_cyclones_on_map(cyclones_df, crs_global_pacific, plot_name)
                    plot_cyclones_filt_time(cyclones_df, crs_global_pacific, plot_name, [0,10])
                    plot_cyclones_filt_time(cyclones_df, crs_global_pacific, obj_plot_name, [0,10])



if __name__ == "__main__":
    # This is for grabbing the same date/time for comparing the model datasets
    # Will eventually be replaced by a bash script or python code to iterate through all the combinations
    year = '2021'
    day = '1022' 
    init = '00' # Either 00 or 12 
    # Example usage
    nc_file = '%sPANG_v100/%s/%s/PANG_v100_GFS_%s%s%s_f000_f240_06.nc'%(data_root_path, year, day, year, day, init)
    laplacian_threshold = 150  # Adjust as needed
    search_radius = 7  # Adjust as needed
    min_pressure_threshold = 100000  # Adjust as needed

    # Tracking the cyclones over time
    ### This is the old method that just uses the laplacian
    #tracked_cyclones = track_cyclones_over_time(nc_file, laplacian_threshold, search_radius, min_pressure_threshold)
    #output_file = '%s%stracked_cyclones_minp.csv'%(root_path, 'results/AI_NWP/')
    #plot_name = '%s%straced_cyclones_plot_minp.png'%(root_path, 'results/AI_NWP/')
    #output_file = 'tracked_cyclones.csv'
    #plot_name = 'tracked_cyclones.png'
    output_file = 'traced_cyclones_wbp.csv'
    plot_name = 'tracked_cyclones_wbp.png'

    # Running tuning
    laplacian_thresholds = [50,25,20,10]
    search_radii = [100,50,30,20,10]
    min_pressure_thresholds = [101000,100000,99000,98000]
    min_pressure_dif = None#[2000,1000,500]

    # Local
    tune_lagrangian_search(nc_file,"%stuning/"%(root_path), output_file, plot_name, search_radii, 
                           laplace=laplacian_thresholds, 
                           min_pressure=min_pressure_thresholds,
                           pressure_dif=min_pressure_dif)

    # Schooner
    #tune_lagrangian_search(nc_file,"%s%stuning/"%(root_path,'results/AI_NWP/'), output_file, plot_name, search_radii, 
    #                       laplace=laplacian_thresholds, 
    #                       min_pressure=min_pressure_thresholds,
    #                       pressure_dif=min_pressure_dif)
