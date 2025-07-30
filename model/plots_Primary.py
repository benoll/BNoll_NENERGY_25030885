# FILE: plots_Primary.py
# PROJECT: Africa Passenger Vehicle TCO Model
# SCRIPT DESCRIPTION: This is the plotting module that takes the output files
# from the main iteration scheme and plots accordingly.

# Import statements
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as col
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib as mpl
import externalParameters as ep
import time
import math
import string

import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.stats import truncnorm
import matplotlib.tri as tri


"""
******************************************************************************
Change path to input folder here:
"""
global INPUT_PATH
INPUT_PATH = r'/Users/benoll/Africa TCO Model_vSub/data/'

"""
******************************************************************************
Change path to output folder here:
"""
global OUTPUT_PATH
OUTPUT_PATH = r'/Users/benoll/Africa TCO Model_vSub/output/'

"""
******************************************************************************
Change path to plots folder here:
"""
global PLOTS_PATH
PLOTS_PATH = r'/Users/benoll/Africa TCO Model_vSub/output/__plots/'

"""
******************************************************************************
Change path to tables folder here:
"""
global TABLES_PATH
TABLES_PATH = r'/Users/benoll/Africa TCO Model_vSub/output/__tables/'

"""
******************************************************************************
Change path to other data folder here:
"""
global OTHER_PATH
OTHER_PATH = r'/Users/benoll/Africa TCO Model_vSub/data/_other/'

"""
******************************************************************************
Change path to Input Data PKL Files folder here:
"""
global INPUT_DATA_PKL_PATH
INPUT_DATA_PKL_PATH = r'/Users/benoll/Africa TCO Model_vSub/output/Input Data PKL Files/'

"""
******************************************************************************
"""

# Initialize the external parameters
ep.initialize(organize_data_bool=False, record_data_bool=False)

"""
Formating Functions and Constant Initialization:
"""

# Initialize unit values for a variable and technology
unit_dictionary = {
        'CostVehicleCAPEX - ICE-G' : 'USD',
        'CostVehicleCAPEX - BEV' : 'USD',
        'CostVehicleScrappage - ICE-G' : 'USD',
        'CostVehicleScrappage - BEV' : 'USD',
        'CostVehicleOM - ICE-G' : 'USD/year',
        'CostVehicleOM - BEV' : 'USD/year',
        'CostVehicleEnergy - ICE-G' : 'USD/L',
        'CostVehicleEnergy - BEV' : 'USD/kWh',
        'VehicleEnergyConsumption - ICE-G' : 'L/km',
        'VehicleEnergyConsumption - BEV' : 'kWh/km',
        'VehicleLifetime - ICE-G' : 'years',
        'VehicleLifetime - BEV' : 'years',
        'VehicleAnnualKmTravelled - ICE-G' : 'km',
        'VehicleAnnualKmTravelled - BEV' : 'km',
        'CostOfCapital - ICE-G' : '%',
        'CostOfCapital - BEV' : '%'
}

color_dictionary = {
        'ICE-G' : 'C1',
        'ICE-SF' : 'C2',
        'BEV' : 'C5'
}

opacity_dictionary = {
        'ICE-G' : 0.5,
        'ICE-SF' : 0.5,
        'BEV' : 0.5
}

# Initialize the excluded countries list
excluded_countries_list = ["Western Sahara", "Seychelles", "Djibouti", "Comoros", "Mauritius", "Sao Tome & Principe", "Cabo Verde"]

def pert_samples(min, mode, max, size):
    alpha = (4*mode + max - 5*min) / (max - min)
    beta = (5*max - min - 4*mode) / (max - min)
    return np.random.beta(alpha, beta, size) * (max - min) + min

def get_tech_unit(parameter, technology):
    # Initialize the key string
    key_string = f'{parameter} - {technology}'

    return unit_dictionary[key_string]

def get_tech_color(technology):
    return color_dictionary[technology]

def get_tech_alpha(technology):
    return opacity_dictionary[technology]

def darken_color(color, amount=0.3):
    # Convert hex color to RGB
    rgb = mcolors.hex2color(color)

    # Darken the color by reducing lightness
    darkened_rgb = tuple(max(0, c * (1 - amount)) for c in rgb)

    # Convert RGB back to hex
    return mcolors.to_hex(darkened_rgb)

def lighten_color(color, amount=0.2):
    """
    Lightens the given color by multiplying (1 + amount) to the RGB values.

    Parameters:
    color (str): The hex color string.
    amount (float): The amount to lighten the color by (0-1).

    Returns:
    str: The lightened hex color string.
    """
    # Convert hex color to RGB
    rgb = mcolors.hex2color(color)

    # Lighten the color by increasing the RGB values
    lightened_rgb = tuple(min(1, c * (1 + amount)) for c in rgb)

    # Convert RGB back to hex
    return mcolors.to_hex(lightened_rgb)

def create_colorbar_params(min_val, max_val, interval, color_start, color_end):
    """
    Generates the boundaries, color map, colorbar ticks, and labels for a custom colorbar.

    Parameters:
    - min_val (float): The minimum value for the colorbar.
    - max_val (float): The maximum value for the colorbar.
    - interval (float): The interval between each section of the colorbar.
    - color_start (str): The starting hex color code.
    - color_end (str): The ending hex color code.

    Returns:
    - boundaries (array): The boundaries for the colorbar intervals.
    - cmap_colors (array): The corresponding colors for each boundary interval.
    - cbar_ticks (array): The positions of the colorbar ticks.
    - cbar_tick_labels (array): The labels for the colorbar ticks.
    """

    # Create the boundaries array from min_val to max_val with the given interval
    boundaries = np.arange(min_val, max_val + interval, interval)

    # Number of color segments corresponding to the number of intervals
    num_segments = len(boundaries) - 1

    # Determine the middle indices for white
    mid_index_lower = num_segments // 2 - 1
    mid_index_upper = num_segments // 2

    # Helper function to lighten a color by blending it with white
    def lighten_color(color, amount=0.75):
        return mcolors.to_hex(
            np.array(mcolors.to_rgb(color)) * (1 - amount) + np.array([1, 1, 1]) * amount
        )

    # Generate the color segments:
    cmap_colors = (
        [mcolors.to_hex(color) for color in mcolors.LinearSegmentedColormap.from_list("", [color_start, lighten_color(color_start)])(np.linspace(0, 1, mid_index_lower))] +
        ["#ffffff", "#ffffff"] +
        [mcolors.to_hex(color) for color in mcolors.LinearSegmentedColormap.from_list("", [lighten_color(color_end), color_end])(np.linspace(0, 1, num_segments - mid_index_upper))]
    )

    # Create colorbar ticks and labels at the boundaries
    cbar_ticks = boundaries
    cbar_tick_labels = [f'{val:.2f}' for val in boundaries]

    return boundaries, cmap_colors, cbar_ticks, cbar_tick_labels


"""
Africa Map Matrix Plots
(using Cartopy)
"""

def mapMatrixAllApplicationsTCO(df, sale_status, scenario):
    # Define the map extent (longitudes and latitudes)
    # Approximate boundaries of Africa (min_lon, max_lon, min_lat, max_lat)
    map_extent = [-25.0, 60.0, -35.0, 40.0]

    # Set the year intervals
    years = [2025, 2030, 2040]
    #years = [2030, 2040]

    # Set up the figure and axes
    nrows=6
    ncols=3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})  # Adjust the size as needed

    # Initialize the colorbar parameters
    boundaries, cmap_colors, cbar_ticks, cbar_tick_labels = create_colorbar_params(-20, 20, 2, "#b35656", "#5589a7")


    # Create the custom colormap
    custom_cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm(boundaries, ncolors=custom_cmap.N, clip=True)

    # Path to Natural Earth shapefile for countries
    shpfilename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = list(reader.records())

    # Pre-initialize the alphabetical labels
    subplot_labels = list(string.ascii_lowercase[:nrows * ncols])
    label_counter = 0

    # Plot the data in subplots
    for i, year in enumerate(years):
        for j, application in enumerate(ep.getApplications()):


            # Calculate subplot index (row major order)
            idx = j * len(years) + i
            ax = axes.flatten()[idx]

            # Set map extent
            ax.set_extent(map_extent, crs=ccrs.PlateCarree())


            # Initialize the plot df
            plot_df = df[scenario].xs(key=(year, sale_status, application), level=('Year', 'Sale Status', 'Application'))

            # Iterate through the records and plot each country within Africa
            for country in countries:
                continent = country.attributes['CONTINENT'].strip('\x00')  # Strip null characters
                if continent == 'Africa':  # Filter only African countries
                    try:
                        country_name = country.attributes['NAME_LONG'].strip('\x00')
                        tech_df = plot_df.xs(key=(country_name), level=('Country'))
                    except:
                        country_name = country.attributes['NAME_LONG'].strip('\x00')

                        if country_name == 'Democratic Republic of the Congo':
                            country_name = 'Congo (Democratic Republic of)'
                        elif country_name == 'Central African Republic':
                            country_name = 'Central Africa Republic'
                        elif country_name == 'Republic of the Congo':
                            country_name = 'Congo (Republic of)'
                        elif country_name == 'Kingdom of eSwatini':
                            country_name = 'Eswatini'
                        elif country_name == 'The Gambia':
                            country_name = 'Gambia'
                        elif country_name == 'Somaliland':
                            country_name = 'Somalia'



                    if country_name in excluded_countries_list:
                        # Color the excluded countries grey
                        color = "grey"
                        cax = ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                                          facecolor=color, edgecolor='grey', linewidth=1, label=country.attributes['NAME_LONG'])

                    else:
                        tech_df = plot_df.xs(key=(country_name), level=('Country'))

                        """
                        Plot Type #1: Color of the TCO percentage change (BEV - ICE-G) / ICE-G
                        """
                        ice_g_tco = tech_df['ICE-G']
                        bev_tco = tech_df['BEV']
                        metric = ((ice_g_tco - bev_tco) / ice_g_tco) * 100
                        color =  custom_cmap(norm(metric))
                        cax = ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                                          facecolor=color, edgecolor='grey', linewidth=0.5, label=country.attributes['NAME_LONG'])

                        metric_value = r'$\frac{TCO_{ICE-G} - TCO_{BEV}}{TCO_{ICE-G}}$'
                        arrow_title_top = r'$TCO_{BEV-SOG} < TCO_{ICE-Fos}$'
                        arrow_title_bot = r'$TCO_{BEV-SOG} > TCO_{ICE-Fos}$'
                        sub_title = r'BEV vs. ICE-G'
                        filename_label = "bevcomp"


                        # """
                        # Plot Type #2: Color of the TCO percentage change (ICE-SF - ICE-G) / ICE-G
                        # """
                        # ice_g_tco = tech_df['ICE-G']
                        # ice_sf_tco = tech_df['ICE-SF']
                        # metric = ((ice_g_tco - ice_sf_tco) / ice_g_tco) * 100
                        # color =  custom_cmap(norm(metric))
                        # cax = ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                        #                   facecolor=color, edgecolor='grey', linewidth=0.5, label=country.attributes['NAME_LONG'])
                        #
                        # metric_value = r'$\frac{TCO_{ICE-G} - TCO_{ICE-SF}}{TCO_{ICE-G}}$'
                        # arrow_title_top = r'$TCO_{ICE-Syn} < TCO_{ICE-Fos}$'
                        # arrow_title_bot = r'$TCO_{ICE-Syn} > TCO_{ICE-Fos}$'
                        # sub_title = r'ICE-SF vs. ICE-G'
                        # filename_label = "syncomp"


            # Add an alphabetical label
            label = subplot_labels[label_counter]
            ax.text(0.0, 1.1, f"({label})", transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"))
            label_counter += 1


            # Title and other features
            if year == years[0]:
                # Calculate the y position for the label
                y_pos = 1 - (j + 1.1) / (nrows+1) # This centers the label in the subplot row
                try:
                    [seg, size] = application.split('-')
                    ylabel = f'{seg}\n({size})'
                except:
                    ylabel = application
                fig.text(0.07, y_pos, ylabel, rotation='horizontal', transform=fig.transFigure, fontsize=12, ha='center')
                #ax.set_ylabel(application)
            if application == ep.getApplications()[0]:
                ax.set_title(f'{year}\n', fontsize=15)
            ax.set_ylabel(application)

            # Hide the axes (this will remove the figure frame entirely)
            ax.axis('off')


    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # Only needed for the colorbar
    cbar_ax = fig.add_axes([0.72, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), cax=cbar_ax, boundaries=boundaries, ticks=boundaries, orientation='vertical', fraction=0.035, pad=0.01)

    # Customize the colorbar ticks and labels
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{float(val):.0f}%" for val in cbar_tick_labels])

    ### Step 1: Get the Colorbar Position ###
    cbar_pos = cbar.ax.get_position()  # [x0, y0, width, height]
    cb_x = cbar_pos.x1 + 0.02  # X position (to the right of the color bar)
    cb_mid_y = (cbar_pos.y0 + cbar_pos.y1) / 2  # Middle of the color bar
    cb_top_y = cbar_pos.y1  # Top of the color bar
    cb_bottom_y = cbar_pos.y0  # Bottom of the color bar

    ### Step 2: Add a horizontal line ###
    hline_x_start = cb_x + 0.03  # Start of horizontal line
    hline_x_end = cb_x + 0.08  # End of horizontal line
    plt.plot([hline_x_start, hline_x_end], [cb_mid_y, cb_mid_y], color="black", linestyle='--', lw=0.8, transform=fig.transFigure, clip_on=False)

    ### Step 3: Add vertical lines for the arrows ###
    vline_x = (hline_x_start + hline_x_end) / 2  # Middle of the horizontal line
    vline_top = cb_mid_y + 0.11  # Extend upwards
    vline_bottom = cb_mid_y - 0.11  # Extend downwards

    # Draw vertical lines (this ensures proper alignment)
    adjust_vert_line_x = 0.00
    plt.plot([vline_x - adjust_vert_line_x, vline_x - adjust_vert_line_x], [cb_mid_y, vline_top], color="black", lw=1.5, transform=fig.transFigure, clip_on=False)
    plt.plot([vline_x - adjust_vert_line_x, vline_x - adjust_vert_line_x], [cb_mid_y, vline_bottom], color="black", lw=1.5, transform=fig.transFigure, clip_on=False)

    ### Step 4: Add arrowheads at the ends of the vertical lines ###
    plt.plot(vline_x, vline_top, marker='^', markersize=7, color="black", transform=fig.transFigure, clip_on=False)  # Up arrow
    plt.plot(vline_x, vline_bottom, marker='v', markersize=7, color="black", transform=fig.transFigure, clip_on=False)  # Down arrow

    ### Step 5: Add explanatory text ###
    fig.text(vline_x - 0.02, vline_top + 0.02, arrow_title_top, fontsize=9, ha="left", va="center", transform=fig.transFigure)
    fig.text(vline_x - 0.02, vline_bottom - 0.02, arrow_title_bot, fontsize=9, ha="left", va="center", transform=fig.transFigure)

    cbar.ax.xaxis.set_label_position('bottom')  # Move label to the bottom
    cbar.ax.set_xlabel('[% difference]', labelpad=20, fontsize=9, loc='center')


    # Adjust spacing
    fig.subplots_adjust(left=0.125, right=0.7, top=0.9, bottom=0.05, wspace=0.210, hspace=0.1)
    #fig.subplots_adjust(left=0.15, right=0.72, top=0.9, bottom=0.05, wspace=0.005, hspace=0.1)      # For the SynFuel plot with only 2 time intervals

    # Add a Suptitle
    fig.suptitle(f'Scenario: {scenario}, Sale Status: {sale_status} \n ({sub_title})')

    # Save figue
    if sale_status == 'New':
        filename = PLOTS_PATH + '0_MapAllAppsTCO_' + scenario + '_' + sale_status+ '_' + filename_label + '.png'
    else:
        filename = PLOTS_PATH + '1_MapAllAppsTCO_' + scenario + '_' + sale_status+ '_' + filename_label +'.png'
    dpi = 300
    plt.savefig(filename, dpi=dpi)

    # Show the plot
    #plt.show()

def mapMatrixAllApplicationsLCAAbatement(df, sale_status, model_scenario, ssp_scenario):
    # Path to Natural Earth shapefile for countries
    shpfilename = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = list(reader.records())

    # Define the map extent (longitudes and latitudes)
    # Approximate boundaries of Africa (min_lon, max_lon, min_lat, max_lat)
    map_extent = [-25.0, 60.0, -40.0, 40.0]

    # Set the year intervals
    years = [2025, 2030, 2040]
    #years = [2030, 2040]

    # Set up the figure and axes
    nrows=6
    ncols=3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()})  # Adjust the size as needed


    # Initialize the colorbar parameters
    boundaries, cmap_colors, cbar_ticks, cbar_tick_labels = create_colorbar_params(-350, 350, 50, "#019176", "#7E105E")

    # Create the custom colormap
    custom_cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm(boundaries, ncolors=custom_cmap.N, clip=True)

    # Pre-initialize the alphabetical labels
    subplot_labels = list(string.ascii_lowercase[:nrows * ncols])
    label_counter = 0

    # Plot the data in subplots
    for j, application in enumerate(ep.getApplications()):
        for i, year in enumerate(years):


            # Calculate subplot index (row major order)
            idx = j * len(years) + i
            ax = axes.flatten()[idx]

            # Set map extent
            ax.set_extent(map_extent, crs=ccrs.PlateCarree())


            # Initialize the plot df
            plot_df = df[model_scenario].xs(key=(year, sale_status, application), level=('Year', 'Sale Status', 'Application'))

            # Iterate through the records and plot each country within Africa
            for country in countries:
                continent = country.attributes['CONTINENT'].strip('\x00')  # Strip null characters
                if continent == 'Africa':  # Filter only African countries
                    try:
                        country_name = country.attributes['NAME_LONG'].strip('\x00')
                        tech_df = plot_df.xs(key=(country_name), level=('Country'))
                    except:
                        country_name = country.attributes['NAME_LONG'].strip('\x00')

                        if country_name == 'Democratic Republic of the Congo':
                            country_name = 'Congo (Democratic Republic of)'
                        elif country_name == 'Central African Republic':
                            country_name = 'Central Africa Republic'
                        elif country_name == 'Republic of the Congo':
                            country_name = 'Congo (Republic of)'
                        elif country_name == 'Kingdom of eSwatini':
                            country_name = 'Eswatini'
                        elif country_name == 'The Gambia':
                            country_name = 'Gambia'
                        elif country_name == 'Somaliland':
                            country_name = 'Somalia'


                    if country_name in excluded_countries_list:
                        # Color the excluded countries grey
                        color = "darkgrey"
                        cax = ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                                          facecolor=color, edgecolor='black', linewidth=1, label=country.attributes['NAME_LONG'])

                    else:
                        tech_df = plot_df.xs(key=(country_name), level=('Country'))

                        """
                        Plot Type #1: Color of the abatement cost (BEV_tco - ICE-G_tco) / (ICE-G_em - BEV_em)
                        """
                        ice_g_tco = tech_df['ICE-G']        # (USD/km)
                        bev_tco = tech_df['BEV']            # (USD/km)
                        ice_g_EM = ep.computeLCA('ICE-G', country_name, application, year, ssp_scenario, model_scenario)/1000         # (tCO2eq/km)
                        bev_EM = ep.computeLCA('BEV', country_name, application, year, ssp_scenario, model_scenario)/1000         # (tCO2eq/km)
                        metric = ((bev_tco - ice_g_tco) / (ice_g_EM - bev_EM))
                        #print(f'{country_name}, {year}, {application}, {metric}')
                        color =  custom_cmap(norm(metric))
                        cax = ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                                          facecolor=color, edgecolor='grey', linewidth=0.5, label=country.attributes['NAME_LONG'])

                        metric_value = r'[USD/tCO2eq]'
                        sub_title = r'BEV vs. ICE-G'
                        filename_label = "bevcomp"

                        # """
                        # Plot Type #2: Color of the abatement cost (ICE-SF_tco - ICE-G_tco) / (ICE-G_em - ICE-SF_em)
                        # """
                        # ice_g_tco = tech_df['ICE-G']        # (USD/km)
                        # ice_sf_tco = tech_df['ICE-SF']            # (USD/km)
                        # ice_g_EM = ep.computeLCA('ICE-G', country_name, application, year, ssp_scenario, model_scenario)/1000         # (tCO2eq/km)
                        # ice_sf_EM = ep.computeLCA('ICE-SF', country_name, application, year, ssp_scenario, model_scenario)/1000         # (tCO2eq/km)
                        # metric = ((ice_sf_tco - ice_g_tco) / (ice_g_EM - ice_sf_EM))
                        # #print(f'{country_name}, {year}, {application}, {metric}')
                        # color =  custom_cmap(norm(metric))
                        # cax = ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                        #                   facecolor=color, edgecolor='grey', linewidth=0.5, label=country.attributes['NAME_LONG'])
                        #
                        # metric_value = r'[USD/tCO2eq]'
                        # sub_title = r'ICE-SF vs. ICE-G'
                        # filename_label = "syncomp"


            # Add an alphabetical label
            label = subplot_labels[label_counter]
            ax.text(0.0, 1.1, f"({label})", transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white"))
            label_counter += 1

            # Title and other features
            if year == years[0]:
                # Calculate the y position for the label
                y_pos = 1 - (j + 1.1) / (nrows+1) # This centers the label in the subplot row
                try:
                    [seg, size] = application.split('-')
                    ylabel = f'{seg}\n({size})'
                except:
                    ylabel = application
                fig.text(0.07, y_pos, ylabel, rotation='horizontal', transform=fig.transFigure, fontsize=12, ha='center')
                #ax.set_ylabel(application)
            if application == ep.getApplications()[0]:
                ax.set_title(f'{year}\n', fontsize=15)
            ax.set_ylabel(application)

            # Hide the axes (this will remove the figure frame entirely)
            ax.axis('off')

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # Only needed for the colorbar
    cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), cax=cbar_ax, boundaries=boundaries, ticks=boundaries, orientation='vertical', fraction=0.02, pad=0.01)
    # Adjust label position using axis properties
    cbar.ax.xaxis.set_label_position('bottom')
    # Move the label to the bottom
    cbar.ax.xaxis.set_label_coords(0.5, -0.1)

    # Customize the colorbar ticks and labels
    cbar.set_ticks(cbar_ticks)  # Specify the positions of the ticks
    cbar.set_ticklabels([f"{float(val):.0f}" for val in cbar_tick_labels])  # Specify the labels for the ticks

    cbar.ax.xaxis.set_label_position('bottom')  # Move label to the bottom
    cbar.ax.set_xlabel('[USD/tCO2eq]', labelpad=20, fontsize=10, loc='center')

    # Adjust spacing
    fig.subplots_adjust(left=0.1, right=0.81, top=0.9, bottom=0.05, wspace=0.210, hspace=0.1)
    #fig.subplots_adjust(left=0.145, right=0.81, top=0.9, bottom=0.05, wspace=0.005, hspace=0.1)      # For the SynFuel plot with only 2 time intervals

    # Add a Suptitle
    fig.suptitle(f'Scenario: {model_scenario}, Sale Status: {sale_status} \n ({sub_title})')

    # Save figue
    if sale_status == 'New':
        filename = PLOTS_PATH + '0_AbatementMapAllApps_' + model_scenario + '_' + sale_status+ '_' + filename_label + '.png'
    else:
        filename = PLOTS_PATH + '1_AbatementMapAllApps_' + model_scenario + '_' + sale_status+ '_' + filename_label + '.png'
    dpi = 300
    plt.savefig(filename, dpi=dpi)

    # Show plot
    #plt.show()


"""
Sensitivity Plots:
"""
def plot_tornado_country_aggregate(base_df, up_df, down_df, sale_status, application):
    """
    Plot a tornado chart for a specified parameter comparing the effect of sensitivity analysis,
    aggregated over the country.

    Args:
    base_df (DataFrame): DataFrame with base values.
    up_df (DataFrame): DataFrame with values when parameters are increased.
    down_df (DataFrame): DataFrame with values when parameters are decreased.
    """

    # Step 1: Create subplots based on Sale Status/Application and Year
    sale_status_list = base_df.index.get_level_values('Sale Status').unique()
    application_list = base_df.index.get_level_values('Application').unique()
    year_list = base_df.index.get_level_values('Year').unique()
    technology_list = base_df.index.get_level_values('Technology').unique()

    nrows = 1
    ncols = len(year_list)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 10), sharex=True, sharey=False)

    color_up = '#377EB8'
    color_down = '#E41A1C'

    # Pre-initialize sorted indices to keep order of parameters the same as left most sensitivity chart (2025)
    sorted_indices = 0

    # Initialize dataframe for data save to excel
    df_save = pd.DataFrame(columns=['Parameters','UpDiff2025', 'DownDiff2025','UpDiff2030', 'DownDiff2030', 'UpDiff2040', 'DownDiff2040'])


    for i, year in enumerate(ep.getYears()):

        ax = axes.flatten()[i]

        # Initialize the dataframes
        base_values = base_df.loc[(year, slice(None), slice(None), slice(None)), :]
        up_values = up_df.loc[(year, slice(None), slice(None), slice(None)), :]
        down_values = down_df.loc[(year, slice(None), slice(None), slice(None)), :]

        base_metric = ((base_df.loc[(year, slice(None), slice(None), 'ICE-G'), :].values - base_df.loc[(year, slice(None), slice(None), 'BEV'), :].values) / base_df.loc[(year, slice(None), slice(None), 'ICE-G'), :].values) * 100
        up_metric_array = ((up_values.loc[(year, slice(None), slice(None), 'ICE-G'), :].values - up_values.loc[(year, slice(None), slice(None), 'BEV'), :].values) / up_values.loc[(year, slice(None), slice(None), 'ICE-G'), :].values) * 100
        down_metric_array = ((down_values.loc[(year, slice(None), slice(None), 'ICE-G'), :].values - down_values.loc[(year, slice(None), slice(None), 'BEV'), :].values) / down_values.loc[(year, slice(None), slice(None), 'ICE-G'), :].values) * 100

        # Calculate differences between base and sensitivity runs
        up_diff = np.array((up_metric_array[0] - base_metric[0]))# / base_metric[0])
        down_diff = np.array((down_metric_array[0] - base_metric[0]))# / base_metric[0])
        parameters = np.array(up_values.columns.values)

        # print(up_diff)
        # print(down_diff)
        # print(stopp)

        if year == 2025:
            # Calculate total effect for sorting (absolute differences)
            total_effect = np.abs(up_diff) + np.abs(down_diff)

            # Sort indices based on the total effect, descending
            sorted_indices = np.argsort(total_effect)

            # Add to the df save
            df_save['Parameters'] = parameters[sorted_indices]

        # Reorder parameters, up_diff, and down_diff based on sorted indices
        parameters_sorted = parameters[sorted_indices]
        up_diff_sorted = up_diff[sorted_indices]
        down_diff_sorted = down_diff[sorted_indices]

        # Plot each technology bar for all parameters on the y-axis
        ax.barh(parameters_sorted, up_diff_sorted, color=color_up, alpha=0.6, label='+20%', left=0, edgecolor='black')
        ax.barh(parameters_sorted, down_diff_sorted, color=color_down, alpha=0.6, label='-20%', left=0, edgecolor='black')

        # Draw a vertical line at x=0 for reference
        ax.axvline(x=0, color='black', linewidth=0.5)

        # Find the maximum absolute value to balance the x-axis
        xmax = max(np.abs(up_diff_sorted).max(), np.abs(down_diff_sorted).max())
        buffer = 5
        ax.set_xlim(-xmax-buffer, xmax+buffer)

        ax.set_title(f"{year}")
        ax.legend(loc='lower left')

        # Record values for excel file save
        df_save[f"UpDiff{year}"] = up_diff_sorted
        df_save[f"DownDiff{year}"] = down_diff_sorted

    plt.suptitle(f"Sensitivitity Metric (x%-pts): TCO Percentage Difference between ICE-G and BEV \n ({sale_status}, {application}, all countries averaged)")
    # Adjust spacing
    fig.subplots_adjust(left=0.150, right=0.976, top=0.9, bottom=0.1, wspace=0.795, hspace=0.11)

    # Save data to excel file
    file_path = TABLES_PATH + "_SensitivityChartData.xlsx"
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df_save.to_excel(writer)

    # Save figue
    filename = PLOTS_PATH + '3_TornadoCharts_' + sale_status + '_' + application + '.png'
    dpi = 300
    plt.savefig(filename, dpi=dpi)

    # Show Plot
    plt.show()


"""
Data Organization Functions:
"""

def organizeAbatementCostBarChartData(tco_df_mean, tco_df_std, model_scenario, ssp_scenario, sale_status, application, tech1, tech2):

    # Initialize a newly formated dataframe (Index: all countries, Columns: all years)
    organized_df_mean = pd.DataFrame(index=ep.getCountries(), columns=[2025, 2030, 2040])

    for country in ep.getCountries():
        # Compute the TCO values
        tco_tech1 = tco_df_mean.loc[(slice(None), country, sale_status, application, tech1), model_scenario]  # (USD/km)
        tco_tech2 = tco_df_mean.loc[(slice(None), country, sale_status, application, tech2), model_scenario]  # (USD/km)

        # Compute the lifecycle emissions values
        lca_tech1 = ep.computeLCA(tech1, country, application, [2025, 2030, 2040], ssp_scenario, model_scenario)/1000         # (tCO2eq/km)
        lca_tech2 = ep.computeLCA(tech2, country, application, [2025, 2030, 2040], ssp_scenario, model_scenario)/1000         # (tCO2eq/km)

        # Compute the abatement cost
        abatement_cost = ((tco_tech2.values - tco_tech1.values) / (lca_tech1.values  - lca_tech2.values))

        # Add to the dataframe
        organized_df_mean.at[country] = abatement_cost


    organized_df_mean.to_excel(TABLES_PATH + "_AbatementCostBarChart_mean.xlsx")


    # print(organized_df_mean)
    # print(stopp)

def organizeTCOWaterfallChartData(df_mean, df_std, model_scenario, sale_status, application):

    # Import the country motorization rates to calculate the weighted average
    xls_country_motorization_rate = pd.ExcelFile(INPUT_PATH + "CountryMotorizationRateData.xlsx")
    CountryMotorizationRate_data = pd.read_excel(xls_country_motorization_rate, index_col=[0])

    # Step 1: Extract the MultiIndex levels from df_mean
    multi_index = df_mean.index  # Extracts MultiIndex structure from df_mean

    # Step 2: Create a new DataFrame by repeating motorization rates for each combination in df_mean
    motorization_df = (
        CountryMotorizationRate_data
        .reindex(multi_index.get_level_values('Country'))  # Align country names
        .rename(columns={'Motorization Rate (Vehicles per 1000 inhabitants)': 'MotorizationRate'})  # Rename for clarity
    )

    # Step 3: Set the index structure to match df_mean
    motorization_df.index = multi_index  # Assign MultiIndex from df_mean

    # Merge df_mean with the motorization rate DataFrame on 'Country'
    df_mean = df_mean.reset_index()
    df_std = df_std.reset_index()
    motorization_df = motorization_df.reset_index()


    # Merge to add Motorization Rate as a column
    df_mean = df_mean.merge(motorization_df[['MotorizationRate']], left_index=True, right_index=True, how="left")
    df_std = df_std.merge(motorization_df[['MotorizationRate']], left_index=True, right_index=True, how="left")

    # Merge mean and standard deviation dataframes to compute total variance properly
    df_combined = df_mean.merge(df_std, on=['Country', 'Year', 'Sale Status', 'Application', 'Technology'], suffixes=('_mean', '_std'))


    # Function to compute weighted average on the TCO mean
    def weighted_avg_mean(group):
        weights = group['MotorizationRate']
        return (group.drop(columns=['MotorizationRate', 'Country']) * weights.values[:, None]).sum() / weights.sum()

    # Compute the weighted average
    organized_df_mean = df_mean.groupby(['Year', 'Sale Status', 'Application', 'Technology']).apply(weighted_avg_mean)

    # Function to compute the total weighted standard deviation
    def weighted_total_std(group):
        weights = group['MotorizationRate_mean']

        # Weighted mean TCO (already computed, but recalculating here within the function)
        weighted_mean = (group.filter(like='_mean') * weights.values[:, None]).sum() / weights.sum()

        # Compute across-country variance component
        squared_diff = (group.filter(like='_mean') - weighted_mean) ** 2
        weighted_squared_diff = squared_diff.mul(weights.values[:, None], axis=0)  # Multiply by weights
        across_country_variance = weighted_squared_diff.sum() / weights.sum()

        # # Compute within-country variance component (using standard deviations)
        # within_country_variance = (group.filter(like='_std') ** 2).mul(weights.values[:, None], axis=0).sum() / weights.sum()

        # # Total weighted variance
        # total_weighted_variance = across_country_variance.values + within_country_variance.values
        # total_weighted_std_dev = np.sqrt(total_weighted_variance)

        total_weighted_std_dev = np.sqrt(across_country_variance.values)

        # Convert back to a Pandas Series, restoring indices from across_country_variance
        total_weighted_std_dev = pd.Series(total_weighted_std_dev, index=across_country_variance.index)

        # Remove '_mean' suffix from index labels
        total_weighted_std_dev.index = total_weighted_std_dev.index.str.replace('_mean', '', regex=True)

        return total_weighted_std_dev

    # Compute the total weighted standard deviation
    organized_df_std = df_combined.groupby(['Year', 'Sale Status', 'Application', 'Technology']).apply(weighted_total_std).drop(columns=['MotorizationRate'])

    # Reset the index to maintain the correct format
    organized_df_mean = organized_df_mean.loc[(slice(None), sale_status, application, slice(None)), :]
    organized_df_std = organized_df_std.loc[(slice(None), sale_status, application, slice(None)), :]

    # Add an additional column for VehicleCAPEX - VehicleCAPEXCoC
    values_array = organized_df_mean['TCOVehicleCAPEX'] - organized_df_mean['TCOVehicleCAPEXCoC']
    organized_df_mean.insert(1, 'TCOVehicleCAPEXNoCoC', values_array)  # Position 1 means second column

    # Add an additional column for VehicleCAPEXNoCoC (take the VehicleCAPEXCoC std)
    values_array = organized_df_std['TCOVehicleCAPEX']
    organized_df_std.insert(1, 'TCOVehicleCAPEXNoCoC', values_array)  # Position 1 means second column

    # Organize the error bar dataframe
    organized_df_error_bar = organized_df_mean.copy()

    tco_parameter_list = ['TCOVehicleCAPEXNoCoC', 'TCOVehicleCAPEXCoC', 'TCOVehicleScrappage', 'TCOSASCAPEX', 'TCOVehicleEnergy', 'TCOOM', 'TCO']

    for i, parameter in enumerate(tco_parameter_list):
        # Initialize the starting y-axis point
        if parameter == 'TCO':
            starting_line_array = organized_df_mean[tco_parameter_list[0:i]].sum(axis=1)
        else:
            starting_line_array = organized_df_mean[tco_parameter_list[0:i+1]].sum(axis=1)

        # Next add the lower limit values
        organized_df_error_bar[parameter + '_LOWER'] = starting_line_array - organized_df_std[parameter]

        # First add the upper limit values
        organized_df_error_bar[parameter + '_UPPER'] = starting_line_array + organized_df_std[parameter]

    # Include a unit multiplier
    # (USD/km -> USD/100km)
    multiplier = 100
    organized_df_mean *= multiplier
    organized_df_std *= multiplier
    organized_df_error_bar *= multiplier


    # Define the file path to save the Excel file
    file_path = TABLES_PATH + "_TCOWaterfallChartData.xlsx"

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        organized_df_mean.to_excel(writer, sheet_name='Mean')
        organized_df_std.to_excel(writer, sheet_name='Std')
        organized_df_error_bar.to_excel(writer, sheet_name='ErrorBar')

def organizeLCAWaterfallChartData(model_scenario, sale_status, application):
    # Set the country (LCA is independent of country)
    country = 'Algeria'

    # 1): Get the total kilometers travelled (ICE-G values taken as the comparable baseline)
    # Get the average annual km travelled (km/year) NOTE: parameter is independent of country
    akt = ep.getValues(model_scenario, 'VehicleAnnualKmTravelled', 'mode').loc[(2025, country, sale_status, application, 'ICE-G'), :].values[0]
    # Get the average lifetime of the vehicle (years) NOTE: parameter is independent of country
    vehicle_lifetime = ep.getValues(model_scenario, 'VehicleLifetime', 'mode').loc[(2025, country, sale_status, application, 'ICE-G'), :].values[0]
    # Compute the average total km travelled over full lifetime of the vehicle (km)
    total_km = akt*vehicle_lifetime

    # Conversion Factor
    diesel_kg_to_liter = 0.85


    # 2): Organize the dataframe structure for storage
    # Initialize an empty DataFrame with MultiIndex and columns
    index = pd.MultiIndex(levels=[[], [], [], []], codes=[[], [], [], []], names=['Year', 'Sale Status', 'Application', 'Technology'])
    organized_df_mean = pd.DataFrame(index=index, columns=['Production Emissions', 'SAS Production Emissions', 'Use Emissions', 'Total Emissions'])
    organized_df_lower = pd.DataFrame(index=index, columns=['Production Emissions', 'SAS Production Emissions', 'Use Emissions', 'Total Emissions'])
    organized_df_upper = pd.DataFrame(index=index, columns=['Production Emissions', 'SAS Production Emissions', 'Use Emissions', 'Total Emissions'])

    ssp_scenario_list = ['SSP2-RCP2.6', 'SSP2-RCP6']

    # 3): Loop through each year
    for i, year in enumerate(ep.getYears()):

        ##### ICE-G #####
        # ----------VEHICLE PRODUCTION EMISSIONS-------------------
        # (kgCO2eq/km)
        vehicle_production_emissions_low = ep.getLCAValues('ICEManufacturingEmissions', application, 'SSP2-RCP2.6', year) / total_km
        vehicle_production_emissions_high = ep.getLCAValues('ICEManufacturingEmissions', application, 'SSP2-RCP6', year) / total_km
        vehicle_production_emissions_mean = (vehicle_production_emissions_low + vehicle_production_emissions_high) / 2

        # ----------VEHICLE USE EMISSIONS-------------------
        # Get the fossil fuel production emissions (kgCO2eq/L)
        ff_prod_emissions_factor_high = ep.getLCAValues('FosFuelProdEmissions', application, 'Max', year)*(1/1000)*diesel_kg_to_liter
        ff_prod_emissions_factor_low = ep.getLCAValues('FosFuelProdEmissions', application, 'Min', year)*(1/1000)*diesel_kg_to_liter
        # Get the ICE vehicle combustion emissions (kgCO2eq/L)
        vehicle_comb_emissions_factor_mid = ep.getLCAValues('ICECombustionEmissions', application, 'Mean', year)*(1/1000)*diesel_kg_to_liter
        # Get the average fuel consumption (L/km)
        fuel_consumption = ep.getValues(model_scenario, 'VehicleEnergyConsumption', 'mode').loc[(year, country, sale_status, application, 'ICE-G'), :].values[0]
        # Compute the vehicle use emissions (kgCO2eq/km)
        vehicle_use_emissions_low = (ff_prod_emissions_factor_low + vehicle_comb_emissions_factor_mid)*fuel_consumption
        vehicle_use_emissions_high = (ff_prod_emissions_factor_high + vehicle_comb_emissions_factor_mid)*fuel_consumption
        vehicle_use_emissions_mean = (vehicle_use_emissions_high + vehicle_use_emissions_low) / 2

        # ----------TOTAL EMISSIONS-------------------
        # (kgCO2eq/km)
        total_emissions_low = vehicle_production_emissions_low + vehicle_use_emissions_low
        total_emissions_high = vehicle_production_emissions_high + vehicle_use_emissions_high
        total_emissions_mean = vehicle_production_emissions_mean + vehicle_use_emissions_mean

        # Store data
        organized_df_mean.loc[(year, sale_status, application, 'ICE-G')] = [vehicle_production_emissions_mean, 0, vehicle_use_emissions_mean, total_emissions_mean]
        organized_df_lower.loc[(year, sale_status, application, 'ICE-G')] = [vehicle_production_emissions_low, 0, vehicle_use_emissions_low, total_emissions_low]
        organized_df_upper.loc[(year, sale_status, application, 'ICE-G')] = [vehicle_production_emissions_high, 0, vehicle_use_emissions_high, total_emissions_high]


        ##### ICE-SF #####
        # ----------VEHICLE PRODUCTION EMISSIONS-------------------
        # (same as ICE-G)

        # ----------VEHICLE USE EMISSIONS-------------------
        # Get the fossil fuel production emissions (kgCO2eq/L)
        sf_prod_emissions_factor_high = ep.getLCAValues('SynFuelProdEmissions', application, 'Max', year)*(1/1000)*diesel_kg_to_liter
        sf_prod_emissions_factor_low = ep.getLCAValues('SynFuelProdEmissions', application, 'Min', year)*(1/1000)*diesel_kg_to_liter
        # Get the average fuel consumption (L/km)
        fuel_consumption = ep.getValues(model_scenario, 'VehicleEnergyConsumption', 'mode').loc[(year, country, sale_status, application, 'ICE-SF'), :].values[0]
        # Compute the vehicle use emissions (kgCO2eq/km)
        vehicle_use_emissions_low = sf_prod_emissions_factor_low*fuel_consumption
        vehicle_use_emissions_high = sf_prod_emissions_factor_high*fuel_consumption
        vehicle_use_emissions_mean = (vehicle_use_emissions_high + vehicle_use_emissions_low) / 2

        # ----------TOTAL EMISSIONS-------------------
        # (kgCO2eq/km)
        total_emissions_low = vehicle_production_emissions_low + vehicle_use_emissions_low
        total_emissions_high = vehicle_production_emissions_high + vehicle_use_emissions_high
        total_emissions_mean = vehicle_production_emissions_mean + vehicle_use_emissions_mean

        # Store data
        organized_df_mean.loc[(year, sale_status, application, 'ICE-SF')] = [vehicle_production_emissions_mean, 0,  vehicle_use_emissions_mean, total_emissions_mean]
        organized_df_lower.loc[(year, sale_status, application, 'ICE-SF')] = [vehicle_production_emissions_low, 0,  vehicle_use_emissions_low, total_emissions_low]
        organized_df_upper.loc[(year, sale_status, application, 'ICE-SF')] = [vehicle_production_emissions_high, 0,  vehicle_use_emissions_high, total_emissions_high]


        ##### BEV #####
        # ----------VEHICLE PRODUCTION EMISSIONS-------------------
        # (kgCO2eq/km)
        vehicle_production_emissions_low = ep.getLCAValues('BEVManufacturingEmissions', application, 'SSP2-RCP2.6', year) / total_km
        vehicle_production_emissions_high = ep.getLCAValues('BEVManufacturingEmissions', application, 'SSP2-RCP6', year) / total_km
        vehicle_production_emissions_mean = (vehicle_production_emissions_high + vehicle_production_emissions_low) / 2

        # ----------SAS PRODUCTION EMISSIONS-------------------
        # SOLAR PV
        # Get the solar PV panel manufacturing emissions (kgCO2eq/m2)
        solarPV_emissions_factor_low = ep.getLCAValues('SASManufacturingEmissions', 'Solar PV', 'SSP2-RCP2.6', year)
        solarPV_emissions_factor_high = ep.getLCAValues('SASManufacturingEmissions', 'Solar PV', 'SSP2-RCP6', year)
        # Conversion factor for a m2 of solar pv to kW (m2/kW) [NOTE: assume 1000 W/m2]
        pv_m2_to_kW_conversion = 1
        # Get the solar PV capacity oversize factor
        solarPV_oversize_factor = ep.getValues(model_scenario, 'SolarPVOversizeFactor', 'mode').loc[(year, country, sale_status, application, 'BEV'), :].values[0]
        # Get the solar PV capacity (kW)
        solar_pv_capacity = ep.getValues(model_scenario, 'SolarPVCapacity', 'mode').loc[(year, country, sale_status, application, 'BEV'), :].values[0] * (1 + solarPV_oversize_factor)
        # Calculate solar PV emissions (kgCO2eq/km)
        solar_pv_emissions_low = solarPV_emissions_factor_low*pv_m2_to_kW_conversion*solar_pv_capacity / total_km
        solar_pv_emissions_high = solarPV_emissions_factor_high*pv_m2_to_kW_conversion*solar_pv_capacity / total_km

        # INVERTER
        # Determine the number of inverter replacements
        inverter_lifetime = ep.getValues(model_scenario, 'InverterLifetime', 'mode').loc[(year, country, sale_status, application, 'BEV'), :].values[0]
        inverter_replacement_factor = math.ceil(vehicle_lifetime/inverter_lifetime)
        # Get the inverter manufacturing emissions (kgCO2eq/km)
        inverter_emissions_low = (ep.getLCAValues('SASManufacturingEmissions', 'Inverter', 'SSP2-RCP2.6', year)*inverter_replacement_factor) / total_km
        inverter_emissions_high = (ep.getLCAValues('SASManufacturingEmissions', 'Inverter', 'SSP2-RCP6', year)*inverter_replacement_factor) / total_km

        # STATIONARY BATTERY
        # Get the stationary battery manufacturing emissions (kgCO2eq/kg)
        stationary_battery_emissions_factor_low = ep.getLCAValues('SASManufacturingEmissions', 'Stationary Battery', 'SSP2-RCP2.6', year)
        stationary_battery_emissions_factor_high = ep.getLCAValues('SASManufacturingEmissions', 'Stationary Battery', 'SSP2-RCP6', year)
        # Get the stationary battery energy density (kWh/kg)
        stationary_battery_energy_density_low = ep.getLCAValues('BatteryEnergyDensity', 'Stationary Battery', 'Low', year)
        stationary_battery_energy_density_high = ep.getLCAValues('BatteryEnergyDensity', 'Stationary Battery', 'High', year)
        # Get the stationary battery capacity oversize factor
        stationary_battery_oversize_factor = ep.getValues('BaseScenario', 'StatBatteryOversizeFactor', 'mode').loc[(year, country, sale_status, application, 'BEV'), :].values[0]
        # Get the stationary battery capacity (kWh)
        stationary_battery_capacity = ep.getValues(model_scenario, 'StationaryBatteryCapacity', 'mode').loc[(year, country, sale_status, application, 'BEV'), :].values[0] * (1 + stationary_battery_oversize_factor)
        # Calculate battery emissions (kgCO2eq/km) NOTE: low emissions matches with high battery energy density (inverse relationship)
        stationary_battery_emissions_low = stationary_battery_emissions_factor_low*(1/stationary_battery_energy_density_high)*stationary_battery_capacity / total_km
        stationary_battery_emissions_high = stationary_battery_emissions_factor_high*(1/stationary_battery_energy_density_low)*stationary_battery_capacity / total_km

        # Compute the average total emissions from SAS production (kgCO2eq/km)
        sas_production_emissions_low = solar_pv_emissions_low + inverter_emissions_low + stationary_battery_emissions_low
        sas_production_emissions_high = solar_pv_emissions_high + inverter_emissions_high + stationary_battery_emissions_high
        sas_production_emissions_mean = (sas_production_emissions_high + sas_production_emissions_low) / 2

        # ----------VEHICLE USE EMISSIONS-------------------
        # (kgCO2eq/km)
        vehicle_use_emissions_low = 0
        vehicle_use_emissions_high = 0
        vehicle_use_emissions_mean = 0

        #------------TOTAL LIFECYCLE------------------
        # (kgCO2eq/km)
        total_emissions_low = vehicle_production_emissions_low + sas_production_emissions_low + vehicle_use_emissions_low
        total_emissions_high = vehicle_production_emissions_high + sas_production_emissions_high + vehicle_use_emissions_high
        total_emissions_mean = vehicle_production_emissions_mean + sas_production_emissions_mean + vehicle_use_emissions_mean

        # Store data
        # (kgCO2eq/km)
        organized_df_mean.loc[(year, sale_status, application, 'BEV')] = [vehicle_production_emissions_mean + sas_production_emissions_mean, sas_production_emissions_mean, vehicle_use_emissions_mean, total_emissions_mean]
        organized_df_lower.loc[(year, sale_status, application, 'BEV')] = [vehicle_production_emissions_low + sas_production_emissions_low, sas_production_emissions_mean, vehicle_use_emissions_low, total_emissions_low]
        organized_df_upper.loc[(year, sale_status, application, 'BEV')] = [vehicle_production_emissions_high + sas_production_emissions_high, sas_production_emissions_mean, vehicle_use_emissions_high, total_emissions_high]

    # 4): Organize error bar data
    # Organize the error bar dataframe
    organized_df_error_bar = organized_df_mean.copy()

    lca_parameter_list = ['Production Emissions', 'Use Emissions', 'Total Emissions']

    for i, parameter in enumerate(lca_parameter_list):
        # Initialize the starting y-axis point
        if parameter == 'Total Emissions':
            starting_line_array = organized_df_mean[lca_parameter_list[0:0]].sum(axis=1)
        else:
            starting_line_array = organized_df_mean[lca_parameter_list[0:i]].sum(axis=1)

        # print(starting_line_array)
        # print(stopp)

        # Next add the lower limit values
        organized_df_error_bar[parameter + '_LOWER'] = starting_line_array + organized_df_lower[parameter]

        # First add the upper limit values
        organized_df_error_bar[parameter + '_UPPER'] = starting_line_array + organized_df_upper[parameter]

    # Include a unit multiplier [kgCO2/km -> kgCO2/100km]
    multiplier = 100
    organized_df_mean *= multiplier
    organized_df_upper *= multiplier
    organized_df_lower *= multiplier
    organized_df_error_bar *= multiplier

    # 5): Save data
    # Define the file path to save the Excel file
    file_path = TABLES_PATH + "_LCAWaterfallChartData.xlsx"

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        organized_df_mean.to_excel(writer, sheet_name='Mean')
        organized_df_upper.to_excel(writer, sheet_name='Upper')
        organized_df_lower.to_excel(writer, sheet_name='Lower')
        organized_df_error_bar.to_excel(writer, sheet_name='ErrorBar')


"""
PLOT CALLS (x2)
"""

################################################################################
## MAP PLOTS - TCO Cost Comparison
# Organize input data
tco_df_mean = pd.read_pickle(OUTPUT_PATH + "\TCO_DF_mean.pkl")
tco_df_std = pd.read_pickle(OUTPUT_PATH + "\TCO_DF_std.pkl")

# Set the scenario
scenarios_df = ep.getScenarios()

for scenario in scenarios_df.index:
    print(f'Plotting maps for the scenario: {scenario}')

    mapMatrixAllApplicationsTCO(tco_df_mean, 'New', scenario)
###############################################################################

################################################################################
## MAP PLOTS - Abatement Costs
# Organize input data
tco_df_mean = pd.read_pickle(OUTPUT_PATH + "\TCO_DF_mean.pkl")
tco_df_std = pd.read_pickle(OUTPUT_PATH + "\TCO_DF_std.pkl")

# Set the model scenarios
scenarios_df = ep.getScenarios()
# Set the SSP Scenario
ssp_scenario = 'SSP2-RCP2.6'

for scenario in scenarios_df.index:
    print(f'Plotting LCA Abatement Cost maps for the scenario: {scenario}')

    mapMatrixAllApplicationsLCAAbatement(tco_df_mean, 'New', scenario, ssp_scenario)
################################################################################


"""
DATA ORGANIZATION CALLS (x3)
"""

################################################################################
## DATA ORGANIZATION - Abatement Cost Bar Chart
# Organize input data
tco_df_mean = pd.read_pickle(OUTPUT_PATH + "\TCO_DF_mean.pkl")
tco_df_std = pd.read_pickle(OUTPUT_PATH + "\TCO_DF_std.pkl")

# Set the model scenario
model_scenario = 'BaseScenario'
# Set the SSP scenario
ssp_scenario = 'SSP2-RCP2.6'
# Set the sale status
sale_status = 'New'
# Set the application
application = '4wheeler-small'
# Set the technology pair
tech1 = 'ICE-G'
tech2 = 'BEV'

print(f'Organizing abatement cost data for: {model_scenario}, {sale_status}, {application}, [{tech1} vs. {tech2}]')
organizeAbatementCostBarChartData(tco_df_mean, tco_df_std, model_scenario, ssp_scenario, sale_status, application, tech1, tech2)

################################################################################

################################################################################
## DATA ORGANIZATION - TCO Component Waterfall Chart
# Organize input data
tco_waterfall_df_mean = pd.read_pickle(OUTPUT_PATH + "\BaseScenario_DF_mean.pkl")
tco_waterfall_df_std = pd.read_pickle(OUTPUT_PATH + "\BaseScenario_DF_std.pkl")

# Set the model scenario
model_scenario = 'BaseScenario'
# Set the sale status
sale_status = 'New'
# Set the application
application = '4wheeler-small'

print(f'Organizing TCO component waterfall data for: {model_scenario}, {sale_status}, {application}')
organizeTCOWaterfallChartData(tco_waterfall_df_mean, tco_waterfall_df_std, model_scenario, sale_status, application)

################################################################################

################################################################################
## DATA ORGANIZATION - LCA Component Waterfall Chart

# Set the model scenario
model_scenario = 'BaseScenario'
# Set the sale status
sale_status = 'New'
# Set the application
application = '4wheeler-small'

print(f'Organizing LCA component waterfall data for: {model_scenario}, {sale_status}, {application}')
organizeLCAWaterfallChartData(model_scenario, sale_status, application)

################################################################################




























#
