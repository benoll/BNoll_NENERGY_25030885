# FILE: plot_AfricaHighRes.py
# PROJECT: Africa Passenger Vehicle TCO Model
# MODULE DESCRIPTION: This is the plotting module for the high resolution LCOC
# figure of Africa.

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

import cartopy.io.shapereader as shpreader
import matplotlib.gridspec as gridspec
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

start_time = time.time()

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
    # Check if all inputs are the same
    if min == mode == max:
        return np.full(size, mode)  # Return an array filled with `mode`

    # Otherwise compute the pert approximated beta distribution
    alpha = (4*mode + max - 5*min) / (max - min)
    beta = (5*max - min - 4*mode) / (max - min)
    return np.random.beta(alpha, beta, size) * (max - min) + min

def getWesternSaharaPoints():

    latitude = [22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27]
    longitude = [-16, -15, -14, -16, -15, -14, -15, -14, -13, -14, -13, -14, -13, -12, -11, -10, -13, -12, -11, -10, -9]

    return [latitude, longitude]

def getDjiboutiPoints():

    latitude = [11, 12]
    longitude = [42, 43]

    return [latitude, longitude]

def getSeychellesPoints():

    latitude = [-5]
    longitude = [55]

    return [latitude, longitude]

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
Internal Functions:
"""

def calculateTCO(data_df, scenario, year, application, sale_status, record):
    """
    This function:
    takes: a multiindex DataFrame with each column containing the values of a
    parameter (i.e. one specific parameter set of a scenario)

    then:
        1. computes TCO and intermediate results:

            TCOVehicleCAPEX = (CostVehicleCAPEX * CRF) / AKT
            TCOVehicleCAPEXCoC = (CostVehicleCAPEX * CRF) / AKT, where CRF->0
            TCOVehicleScrappage = (CostVehicleScrappage * 1/(1+i)^N * CRF) / AKT
            TCOOM = ((1/N) * (∑(n) CostOM/(1+i)^n) / AKT
            TCOVehicleEnergy = ((1/N) * (∑(n) CostVehicleEnergy/(1+i)^n) / AKT


        2. records results:
            a) values of of this scenario in one file for each type of result
            b) values of all types of results of this scenario in one file

    returns: -
    """

    # Initialize the number of Monte Carlo draws
    n_draws = ep.getMonteCarloDraws()
    # Initialize the number of data points
    n_points = len(data_df)

    ## Calculate all distributed input parameters for the specific scenario
    print("Computing VehicleAnnualKmTravelled")
    # Annual Kilometers Travelled [km]
    mode = ep.getValues(scenario, 'VehicleAnnualKmTravelled', 'mode').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    max = ep.getValues(scenario, 'VehicleAnnualKmTravelled', 'max').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    min = ep.getValues(scenario, 'VehicleAnnualKmTravelled', 'min').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    annualKmTravelled = pert_samples(min, mode, max, (n_points,n_draws))


    print("Computing VehicleLifetime")
    # Vehicle Lifetime [years]
    mode = ep.getValues(scenario, 'VehicleLifetime', 'mode').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    max = ep.getValues(scenario, 'VehicleLifetime', 'max').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    min = ep.getValues(scenario, 'VehicleLifetime', 'min').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    vehicleLifetime_ice_g = np.round(pert_samples(min, mode, max, (n_points,n_draws)))
    mode = ep.getValues(scenario, 'VehicleLifetime', 'mode').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    max = ep.getValues(scenario, 'VehicleLifetime', 'max').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    min = ep.getValues(scenario, 'VehicleLifetime', 'min').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    vehicleLifetime_ice_sf = np.round(pert_samples(min, mode, max, (n_points,n_draws)))
    mode = ep.getValues(scenario, 'VehicleLifetime', 'mode').loc[(year, 'Algeria', sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'VehicleLifetime', 'max').loc[(year, 'Algeria', sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'VehicleLifetime', 'min').loc[(year, 'Algeria', sale_status, application, 'BEV'), 'BaseValue']
    vehicleLifetime_bev = np.round(pert_samples(min, mode, max, (n_points,n_draws)))

    print("Computing (in a loop) CostOfCapital")
    # Cost of Capital [unitless]
    mean_ice_g = np.zeros((n_points,1))
    mean_ice_sf = np.zeros((n_points,1))
    mean_bev = np.zeros((n_points,1))
    std_ice_g = np.zeros((n_points,1))
    std_ice_sf = np.zeros((n_points,1))
    std_bev = np.zeros((n_points,1))
    for i, iso_code in enumerate( data_df['Country (ISO)']):
        country = ep.getISOCodeCountry(iso_code)
        mean_ice_g[i] = ep.getValues(scenario, 'CostOfCapital', 'mean').loc[(year, country, sale_status, application, 'ICE-G'), 'BaseValue']
        mean_ice_sf[i] = ep.getValues(scenario, 'CostOfCapital', 'mean').loc[(year, country, sale_status, application, 'ICE-SF'), 'BaseValue']
        mean_bev[i] = ep.getValues(scenario, 'CostOfCapital', 'mean').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
        std_ice_g[i] = ep.getValues(scenario, 'CostOfCapital', 'std').loc[(year, country, sale_status, application, 'ICE-G'), 'BaseValue']
        std_ice_sf[i] = ep.getValues(scenario, 'CostOfCapital', 'std').loc[(year, country, sale_status, application, 'ICE-SF'), 'BaseValue']
        std_bev[i] = ep.getValues(scenario, 'CostOfCapital', 'std').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']

    costOfCapital_ice_g = np.random.normal(loc=mean_ice_g, scale=std_ice_g, size=(n_points, n_draws))
    costOfCapital_ice_sf = np.random.normal(loc=mean_ice_sf, scale=std_ice_sf, size=(n_points, n_draws))
    costOfCapital_bev = np.random.normal(loc=mean_bev, scale=std_bev, size=(n_points, n_draws))

    # Near Zero or "Risk-Free" Cost of Capital (i.e. i = 0.00001%) [unitless]
    nearZeroCostOfCapital_ice_g = costOfCapital_ice_g/costOfCapital_ice_g*0.014
    nearZeroCostOfCapital_ice_sf = costOfCapital_ice_sf/costOfCapital_ice_sf*0.014
    nearZeroCostOfCapital_bev = costOfCapital_bev/costOfCapital_bev*0.014

    print("Computing presentValueFactor")
    # Present Value Factor [unitless]
    presentValueFactor_ice_g = calculatePresentValueFactor(costOfCapital_ice_g, vehicleLifetime_ice_g)
    presentValueFactor_ice_sf = calculatePresentValueFactor(costOfCapital_ice_sf, vehicleLifetime_ice_sf)
    presentValueFactor_bev = calculatePresentValueFactor(costOfCapital_bev, vehicleLifetime_bev)

    print("Computing capitalRecoveryFactor")
    # Capital Recovery Factor [unitless]
    capitalRecoveryFactor_ice_g = calculateCapitalRecoveryFactor(costOfCapital_ice_g, vehicleLifetime_ice_g)
    capitalRecoveryFactor_ice_sf = calculateCapitalRecoveryFactor(costOfCapital_ice_sf, vehicleLifetime_ice_sf)
    capitalRecoveryFactor_bev = calculateCapitalRecoveryFactor(costOfCapital_bev, vehicleLifetime_bev)

    # Near Zero Capital Recovery Factor [unitless]
    nearZeroCapitalRecoveryFactor_ice_g = calculateCapitalRecoveryFactor(nearZeroCostOfCapital_ice_g, vehicleLifetime_ice_g)
    nearZeroCapitalRecoveryFactor_ice_sf = calculateCapitalRecoveryFactor(nearZeroCostOfCapital_ice_sf, vehicleLifetime_ice_sf)
    nearZeroCapitalRecoveryFactor_bev = calculateCapitalRecoveryFactor(nearZeroCostOfCapital_bev, vehicleLifetime_bev)

    ############################################################################
    # ## Correlated parameters: [CostVehicleCAPEX, VehicleEnergyConsumption]
    #
    # # Results is a (5000, n_draws, 2) array where:
    # # - Each "i" index corresponds to one of the 5000 sets of parameters
    # # - Each "j" index in `n_draws` corresponds to one draw from the distribution
    # # - The last dimension stores the two correlated values
    #
    # print("Computing VehicleEnergyConsumption and CostVehicleCAPEX as correlated parameters")
    # # Vehicle Energy Consumption [L/km or kWh/km]
    # mode = ep.getValues(scenario, 'VehicleEnergyConsumption', 'mode').values.astype(float)
    # max = ep.getValues(scenario, 'VehicleEnergyConsumption', 'max').values.astype(float)
    # min = ep.getValues(scenario, 'VehicleEnergyConsumption', 'min').values.astype(float)
    # mean_1 = (min + mode + max)/3
    # std_1 = (np.power(min,2) + np.power(mode,2) + np.power(max,2) - min*mode - min*max - mode*max)/18
    #
    # # Cost Vehicle CAPEX [USD]
    # mean_2 = ep.getValues(scenario, 'CostVehicleCAPEX', 'mean').values.astype(float)
    # std_2 = ep.getValues(scenario, 'CostVehicleCAPEX', 'std').values.astype(float)
    #
    # # Concatinate the mean and std
    # mean_12 = np.concatenate((mean_1, mean_2), axis=1)
    # std_12 = np.concatenate((std_1, std_2), axis=1)
    #
    # # Correlation between the two parameters
    # correlation = -0.5
    #
    # # Generate the covariance matrices for each set of parameters
    # covariance_matrices = np.array([[[std_12[i, 0]**2, correlation * std_12[i, 0] * std_12[i, 1]],
    #                                  [correlation * std_12[i, 0] * std_12[i, 1], std_12[i, 1]**2]]
    #                                 for i in range(mean_1.size)])
    #
    # # Allocate space for the results
    # results = np.zeros((mean_1.size, n_draws, 2))
    #
    # # Generate samples
    # for i in range(mean_1.size):
    #     results[i] = np.random.multivariate_normal(mean_12[i], covariance_matrices[i], n_draws)
    #
    # vehicleEnergyConsumption = results[:,:,0]
    # costVehicleCAPEX = results[:,:,1]

    ############################################################################
    # Uncorrelated parameters: [CostVehicleCAPEX, VehicleEnergyConsumption]

    print("Computing VehicleEnergyConsumption")
    # Vehicle Energy Consumption [L/km or kWh/km]
    mode = ep.getValues(scenario, 'VehicleEnergyConsumption', 'mode').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    max = ep.getValues(scenario, 'VehicleEnergyConsumption', 'max').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    min = ep.getValues(scenario, 'VehicleEnergyConsumption', 'min').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    vehicleEnergyConsumption_ice_g = pert_samples(min, mode, max, (n_points,n_draws))
    mode = ep.getValues(scenario, 'VehicleEnergyConsumption', 'mode').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    max = ep.getValues(scenario, 'VehicleEnergyConsumption', 'max').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    min = ep.getValues(scenario, 'VehicleEnergyConsumption', 'min').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    vehicleEnergyConsumption_ice_sf = pert_samples(min, mode, max, (n_points,n_draws))
    mode = ep.getValues(scenario, 'VehicleEnergyConsumption', 'mode').loc[(year, 'Algeria', sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'VehicleEnergyConsumption', 'max').loc[(year, 'Algeria', sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'VehicleEnergyConsumption', 'min').loc[(year, 'Algeria', sale_status, application, 'BEV'), 'BaseValue']
    vehicleEnergyConsumption_bev = pert_samples(min, mode, max, (n_points,n_draws))

    print("Computing CostVehicleCAPEX")
    # Cost Vehicle CAPEX [USD]
    mean = ep.getValues(scenario, 'CostVehicleCAPEX', 'mean').values.astype(float)
    std = ep.getValues(scenario, 'CostVehicleCAPEX', 'std').values.astype(float)
    costVehicleCAPEX = np.random.normal(loc=mean, scale=std, size=(mean.size,n_draws))

    mean = ep.getValues(scenario, 'CostVehicleCAPEX', 'mean').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    std = ep.getValues(scenario, 'CostVehicleCAPEX', 'std').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    costVehicleCAPEX_ice_g = np.random.normal(loc=mean, scale=std, size=(n_points,n_draws))
    mean = ep.getValues(scenario, 'CostVehicleCAPEX', 'mean').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    std = ep.getValues(scenario, 'CostVehicleCAPEX', 'std').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    costVehicleCAPEX_ice_sf = np.random.normal(loc=mean, scale=std, size=(n_points,n_draws))
    mean = ep.getValues(scenario, 'CostVehicleCAPEX', 'mean').loc[(year, 'Algeria', sale_status, application, 'BEV'), 'BaseValue']
    std = ep.getValues(scenario, 'CostVehicleCAPEX', 'std').loc[(year, 'Algeria', sale_status, application, 'BEV'), 'BaseValue']
    costVehicleCAPEX_bev = np.random.normal(loc=mean, scale=std, size=(n_points,n_draws))


    ############################################################################

    print("Computing CostVehicleScrappage")
    # Cost Vehicle Scrappage [USD]
    a = ep.getValues(scenario, 'CostVehicleScrappage', 'a').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    b = ep.getValues(scenario, 'CostVehicleScrappage', 'b').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    x_0 = ep.getValues(scenario, 'CostVehicleScrappage', 'x0').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    y_0 = ep.getValues(scenario, 'CostVehicleScrappage', 'y0').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    # Calculate the total vehicle km travelled (over the full vehicle lifetime)
    total_km_travelled_ice_g = vehicleLifetime_ice_g * annualKmTravelled
    total_km_travelled_ice_sf = vehicleLifetime_ice_sf * annualKmTravelled
    total_km_travelled_bev = vehicleLifetime_bev * annualKmTravelled
    # Calculate the scrappage value % of retail value
    perc_retail_value_ice_g = a + (y_0 - a) * np.exp(-b * (total_km_travelled_ice_g - x_0))
    perc_retail_value_ice_sf = a + (y_0 - a) * np.exp(-b * (total_km_travelled_ice_sf - x_0))
    perc_retail_value_bev = a + (y_0 - a) * np.exp(-b * (total_km_travelled_bev - x_0))
    costVehicleScrappage_ice_g = costVehicleCAPEX_ice_g * perc_retail_value_ice_g
    costVehicleScrappage_ice_sf = costVehicleCAPEX_ice_sf * perc_retail_value_ice_sf
    costVehicleScrappage_bev = costVehicleCAPEX_bev * perc_retail_value_bev

    print("Computing CostVehicleOM")
    # Cost O&M [USD/year]
    mode = ep.getValues(scenario, 'CostVehicleOM', 'mode').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    max = ep.getValues(scenario, 'CostVehicleOM', 'max').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    min = ep.getValues(scenario, 'CostVehicleOM', 'min').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    costOM_ice_g = pert_samples(min, mode, max, (n_points,n_draws))
    mode = ep.getValues(scenario, 'CostVehicleOM', 'mode').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    max = ep.getValues(scenario, 'CostVehicleOM', 'max').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    min = ep.getValues(scenario, 'CostVehicleOM', 'min').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    costOM_ice_sf = pert_samples(min, mode, max, (n_points,n_draws))
    mode = ep.getValues(scenario, 'CostVehicleOM', 'mode').loc[(year, 'Algeria', sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'CostVehicleOM', 'max').loc[(year, 'Algeria', sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'CostVehicleOM', 'min').loc[(year, 'Algeria', sale_status, application, 'BEV'), 'BaseValue']
    costOM_bev = pert_samples(min, mode, max, (n_points,n_draws))

    print("Computing CostVehicleEnergy")
    # Cost Vehicle Energy [USD/year]        NOTE: CostVehicleEnergy is multiplied by vehicleEfficiency and annualKmTravelled to give cost in USD
    # ICE-G
    mean = ep.getValues(scenario, 'CostVehicleEnergy', 'mean').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    std = ep.getValues(scenario, 'CostVehicleEnergy', 'std').loc[(year, 'Algeria', sale_status, application, 'ICE-G'), 'BaseValue']
    # Create a truncated normal distribution to avoid negative energy costs
    lower, upper = 0, np.inf
    a, b = (lower - mean) / std, (upper - mean) / std
    trunc_normal_dist = truncnorm(a, b, loc=mean, scale=std)
    costVehicleEnergy_ice_g = trunc_normal_dist.rvs(size=(n_points,n_draws)) * vehicleEnergyConsumption_ice_g * annualKmTravelled
    # ICE-SF
    mean = ep.getValues(scenario, 'CostVehicleEnergy', 'mean').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    std = ep.getValues(scenario, 'CostVehicleEnergy', 'std').loc[(year, 'Algeria', sale_status, application, 'ICE-SF'), 'BaseValue']
    # Create a truncated normal distribution to avoid negative energy costs
    lower, upper = 0, np.inf
    a, b = (lower - mean) / std, (upper - mean) / std
    trunc_normal_dist = truncnorm(a, b, loc=mean, scale=std)
    costVehicleEnergy_ice_sf = trunc_normal_dist.rvs(size=(n_points,n_draws)) * vehicleEnergyConsumption_ice_sf * annualKmTravelled

    print("Computing (in a loop) CostVehicleLCOC")
    # LCOC [USD/kWh]
    lcoc = np.zeros((n_points,n_draws))
    for i, iso_code in enumerate(data_df['Country (ISO)']):
        if i == n_points/4:
            print(f'     Calculating point {i} of {n_points} (25% finished)')
        if i == n_points/2:
            print(f'     Calculating point {i} of {n_points} (50% finished)')
        if i == 3*n_points/4:
            print(f'     Calculating point {i} of {n_points} (75% finished)')
        country = ep.getISOCodeCountry(iso_code)

        sas_component_capacities = data_df[f'CompCaps {sale_status} {application}'][i]

        # Calculate LCOC
        lcoc[i,:] = calculateLCOC(scenario, year, country, application, sale_status, 1, n_draws, costOfCapital_bev[i,:], vehicleEnergyConsumption_bev[i,:], annualKmTravelled[i,:], vehicleLifetime_bev[i,:], sas_component_capacities)

    costVehicleEnergy_bev = lcoc * vehicleEnergyConsumption_bev * annualKmTravelled

    metric_selection = 'TCO'

    if metric_selection == 'TCO':
        ############################################################################
        ############################################################################
        """
        Calculation for the Total Cost of Ownership (TCO):

                1. computes TCO and intermediate results:

                    TCOVehicleCAPEX = (CostVehicleCAPEX * CRF) / AKT
                    TCOVehicleCAPEXCoC = (CostVehicleCAPEX * CRF) / AKT, where CRF->0
                    TCOVehicleScrappage = (CostVehicleScrappage * 1/(1+i)^N * CRF) / AKT
                    TCOOM = ((1/N) * (∑(n) CostOM/(1+i)^n) / AKT
                    TCOVehicleEnergy = ((1/N) * (∑(n) CostVehicleEnergy/(1+i)^n) / AKT

        """

        ## Calculate TCO and all intermediate results
        print("Computing TCO Values")


        print("Computing TCOVehicleCAPEX Values")
        # Calculate the total cost of vehicle CAPEX (TCOVehicleCAPEX)
        tco_vehicle_capex_ice_g = (costVehicleCAPEX_ice_g.copy() * capitalRecoveryFactor_ice_g) / annualKmTravelled
        tco_vehicle_capex_ice_sf = (costVehicleCAPEX_ice_sf.copy() * capitalRecoveryFactor_ice_sf) / annualKmTravelled
        tco_vehicle_capex_bev = (costVehicleCAPEX_bev.copy() * capitalRecoveryFactor_bev) / annualKmTravelled

        print("Computing TCOVehicleCAPEXCoC")
        # Calculate the total cost of vehicle CAPEX cost of capital (TCOVehicleCAPEXCoC)
        tco_vehicle_capex_coc_ice_g = tco_vehicle_capex_ice_g  - ((costVehicleCAPEX_ice_g.copy() * nearZeroCapitalRecoveryFactor_ice_g) / annualKmTravelled)
        tco_vehicle_capex_coc_ice_sf = tco_vehicle_capex_ice_sf  - ((costVehicleCAPEX_ice_sf.copy() * nearZeroCapitalRecoveryFactor_ice_sf) / annualKmTravelled)
        tco_vehicle_capex_coc_bev = tco_vehicle_capex_bev  - ((costVehicleCAPEX_bev.copy() * nearZeroCapitalRecoveryFactor_bev) / annualKmTravelled)

        print("Computing TCOVehicleScrappage")
        # Calculate the total cost of vehicle scrappage value (TCOVehicleScrappage)
        tco_vehicle_scrappage_ice_g = (-1) * (costVehicleScrappage_ice_g.copy() * presentValueFactor_ice_g * capitalRecoveryFactor_ice_g) / annualKmTravelled
        tco_vehicle_scrappage_ice_sf = (-1) * (costVehicleScrappage_ice_sf.copy() * presentValueFactor_ice_sf * capitalRecoveryFactor_ice_sf) / annualKmTravelled
        tco_vehicle_scrappage_bev = (-1) * (costVehicleScrappage_bev.copy() * presentValueFactor_bev * capitalRecoveryFactor_bev) / annualKmTravelled

        print("Computing TCOOM")
        # Calculate the total cost of O&M (TCOOM)
        tco_om_ice_g = costOM_ice_g.copy()
        tco_om_ice_sf = costOM_ice_sf.copy()
        tco_om_bev = costOM_bev.copy()
        tco_om_ice_g = matrixBroadcastingFunction(tco_om_ice_g, costOfCapital_ice_g, vehicleLifetime_ice_g, div_bool=True, cond_matrix_bool=[False, False])
        tco_om_ice_sf = matrixBroadcastingFunction(tco_om_ice_sf, costOfCapital_ice_sf, vehicleLifetime_ice_sf, div_bool=True, cond_matrix_bool=[False, False])
        tco_om_bev = matrixBroadcastingFunction(tco_om_bev, costOfCapital_bev, vehicleLifetime_bev, div_bool=True, cond_matrix_bool=[False, False])
        tco_om_ice_g /= annualKmTravelled
        tco_om_ice_sf /= annualKmTravelled
        tco_om_bev /= annualKmTravelled

        print("Computing TCOVehicleEnergy")
        # Calculate the total cost of vehicle energy (TCOVehicleEnergy)
        tco_vehicle_energy_ice_g = costVehicleEnergy_ice_g.copy()
        tco_vehicle_energy_ice_sf = costVehicleEnergy_ice_sf.copy()
        tco_vehicle_energy_bev = costVehicleEnergy_bev.copy()
        tco_vehicle_energy_ice_g = matrixBroadcastingFunction(tco_vehicle_energy_ice_g, costOfCapital_ice_g, vehicleLifetime_ice_g, div_bool=True, cond_matrix_bool=[False, False])
        tco_vehicle_energy_ice_sf = matrixBroadcastingFunction(tco_vehicle_energy_ice_sf, costOfCapital_ice_sf, vehicleLifetime_ice_sf, div_bool=True, cond_matrix_bool=[False, False])
        tco_vehicle_energy_bev = matrixBroadcastingFunction(tco_vehicle_energy_bev, costOfCapital_bev, vehicleLifetime_bev, div_bool=True, cond_matrix_bool=[False, False])
        tco_vehicle_energy_ice_g /= annualKmTravelled
        tco_vehicle_energy_ice_sf /= annualKmTravelled
        tco_vehicle_energy_bev /= annualKmTravelled

        print("Computing TCO")
        # Calculate the total cost of ownership
        tco_ice_g = tco_vehicle_capex_ice_g + tco_vehicle_scrappage_ice_g + tco_om_ice_g + tco_vehicle_energy_ice_g
        tco_ice_sf = tco_vehicle_capex_ice_sf + tco_vehicle_scrappage_ice_sf + tco_om_ice_sf + tco_vehicle_energy_ice_sf
        tco_bev = tco_vehicle_capex_bev + tco_vehicle_scrappage_bev + tco_om_bev + tco_vehicle_energy_bev


        ############################################################################
        print("Recording TCO Values")

        ## Silently record results if argument record is true (default), else return all outputs

        # Prepare output of this calculation (for recording or returning)
        ## Mean
        all_outputs_mean = pd.concat([
                        pd.Series(tco_vehicle_capex_ice_g.mean(axis=1)),
                        pd.Series(tco_vehicle_capex_coc_ice_g.mean(axis=1)),
                        pd.Series(tco_vehicle_scrappage_ice_g.mean(axis=1)),
                        pd.Series(tco_om_ice_g.mean(axis=1)),
                        pd.Series(tco_vehicle_energy_ice_g.mean(axis=1)),
                        pd.Series(tco_ice_g.mean(axis=1)),
                        pd.Series(tco_vehicle_capex_ice_sf.mean(axis=1)),
                        pd.Series(tco_vehicle_capex_coc_ice_sf.mean(axis=1)),
                        pd.Series(tco_vehicle_scrappage_ice_sf.mean(axis=1)),
                        pd.Series(tco_om_ice_sf.mean(axis=1)),
                        pd.Series(tco_vehicle_energy_ice_sf.mean(axis=1)),
                        pd.Series(tco_ice_sf.mean(axis=1)),
                        pd.Series(tco_vehicle_capex_bev.mean(axis=1)),
                        pd.Series(tco_vehicle_capex_coc_bev.mean(axis=1)),
                        pd.Series(tco_vehicle_scrappage_bev.mean(axis=1)),
                        pd.Series(tco_om_bev.mean(axis=1)),
                        pd.Series(tco_vehicle_energy_bev.mean(axis=1)),
                        pd.Series(tco_bev.mean(axis=1)),
                        pd.Series(lcoc.mean(axis=1)),
                        pd.Series(costOfCapital_ice_g.mean(axis=1)),
                        pd.Series(costOfCapital_bev.mean(axis=1))],
                        axis=1)
        all_outputs_mean.columns = [
        'TCOVehicleCAPEX_ICE_G', 'TCOVehicleCAPECCoC_ICE_G', 'TCOVehicleScrappage_ICE_G', 'TCOOM_ICE_G', 'TCOVehicleEnergy_ICE_G', 'TCO_ICE_G',
        'TCOVehicleCAPEX_ICE_SF', 'TCOVehicleCAPECCoC_ICE_SF', 'TCOVehicleScrappage_ICE_SF', 'TCOOM_ICE_SF', 'TCOVehicleEnergy_ICE_SF', 'TCO_ICE_SF',
        'TCOVehicleCAPEX_BEV', 'TCOVehicleCAPECCoC_BEV', 'TCOVehicleScrappage_BEV', 'TCOOM_BEV', 'TCOVehicleEnergy_BEV', 'TCO_BEV',
        'LCOC',
        'CoC_ICE_G', 'CoC_BEV'
                                    ]
        ## Standard Deviation
        all_outputs_std = pd.concat([
                        pd.Series(tco_vehicle_capex_ice_g.std(axis=1)),
                        pd.Series(tco_vehicle_capex_coc_ice_g.std(axis=1)),
                        pd.Series(tco_vehicle_scrappage_ice_g.std(axis=1)),
                        pd.Series(tco_om_ice_g.std(axis=1)),
                        pd.Series(tco_vehicle_energy_ice_g.std(axis=1)),
                        pd.Series(tco_ice_g.std(axis=1)),
                        pd.Series(tco_vehicle_capex_ice_sf.std(axis=1)),
                        pd.Series(tco_vehicle_capex_coc_ice_sf.std(axis=1)),
                        pd.Series(tco_vehicle_scrappage_ice_sf.std(axis=1)),
                        pd.Series(tco_om_ice_sf.std(axis=1)),
                        pd.Series(tco_vehicle_energy_ice_sf.std(axis=1)),
                        pd.Series(tco_ice_sf.std(axis=1)),
                        pd.Series(tco_vehicle_capex_bev.std(axis=1)),
                        pd.Series(tco_vehicle_capex_coc_bev.std(axis=1)),
                        pd.Series(tco_vehicle_scrappage_bev.std(axis=1)),
                        pd.Series(tco_om_bev.std(axis=1)),
                        pd.Series(tco_vehicle_energy_bev.std(axis=1)),
                        pd.Series(tco_bev.std(axis=1)),
                        pd.Series(lcoc.std(axis=1)),
                        pd.Series(costOfCapital_ice_g.std(axis=1)),
                        pd.Series(costOfCapital_bev.std(axis=1))],
                        axis=1)
        all_outputs_std.columns = [
        'TCOVehicleCAPEX_ICE_G', 'TCOVehicleCAPECCoC_ICE_G', 'TCOVehicleScrappage_ICE_G', 'TCOOM_ICE_G', 'TCOVehicleEnergy_ICE_G', 'TCO_ICE_G',
        'TCOVehicleCAPEX_ICE_SF', 'TCOVehicleCAPECCoC_ICE_SF', 'TCOVehicleScrappage_ICE_SF', 'TCOOM_ICE_SF', 'TCOVehicleEnergy_ICE_SF', 'TCO_ICE_SF',
        'TCOVehicleCAPEX_BEV', 'TCOVehicleCAPECCoC_BEV', 'TCOVehicleScrappage_BEV', 'TCOOM_BEV', 'TCOVehicleEnergy_BEV', 'TCO_BEV',
        'LCOC',
        'CoC_ICE_G', 'CoC_BEV'
                                    ]
        if record:
            # Record all results of this scenario in one file
            all_outputs_mean.to_pickle(PLOTS_PATH + f'_AfricaZoomPlot_AllOutputs_{scenario}_{year}_{sale_status}_{application}_mean.pkl')
            all_outputs_std.to_pickle(PLOTS_PATH + f'_AfricaZoomPlot_AllOutputs_{scenario}_{year}_{sale_status}_{application}_std.pkl')

        return all_outputs_mean, all_outputs_std

    elif metric_selection == 'LCOD':
        ############################################################################
        ############################################################################
        """
        Calculation for the Levelized Cost of Driving (LCOD):

            1. computes LCOD and intermediate results:

                LCOVehicleCAPEX = (CostVehicleCAPEX) / (∑(n) AKT/(1+i)^n)
                LCOOM = (∑(n) CostOM/(1+i)^n) / (∑(n) AKT/(1+i)^n)
                LCOVehicleEnergy = (∑(n) CostVehicleEnergy/(1+i)^n) / (∑(n) AKT/(1+i)^n)

        """

        ## Calculate the LCOD and all intermediate results
        print("Computing LCOD Values")

        print("Computing LCOAnnualKmTravelled")
        # Calculate the levelized annual kilometers travelled (LCOAnnualKmTravelled)
        lco_akt = annualKmTravelled.copy()
        lco_akt = matrixBroadcastingFunction(lco_akt, costOfCapital, vehicleLifetime, div_bool=False, cond_matrix_bool=[False, False])

        print("Computing LCOOM")
        # Calculate the levelized cost of O&M (LCOOM)
        lco_om = costOM.copy()
        lco_om = matrixBroadcastingFunction(lco_om, costOfCapital, vehicleLifetime, div_bool=False, cond_matrix_bool=[False, False])
        lco_om /= lco_akt

        print("Computing LCOVehicleEnergy")
        # Calculate the levelized cost of vehicle energy (LCOVehicleEnergy)
        lco_vehicle_energy = costVehicleEnergy.copy()
        lco_vehicle_energy = matrixBroadcastingFunction(lco_vehicle_energy, costOfCapital, vehicleLifetime, div_bool=False, cond_matrix_bool=[False, False])
        lco_vehicle_energy /= lco_akt

        print("Computing LCOVehicleCAPEX")
        # Calculate the levelized cost of vehicle CAPEX (LCOVehicleCAPEX)
        lco_vehicle_capex = costVehicleCAPEX.copy() / lco_akt

        print("Computing LCOVehicleScrappage")
        # Calculate the levelized cost of vehicle CAPEX (LCOVehicleCAPEX)
        lco_vehicle_scrappage = (-1) * costVehicleScrappage.copy() / lco_akt

        # Calculate the levelized cost of driving
        lcod = lco_vehicle_capex + lco_vehicle_scrappage + lco_om + lco_vehicle_energy

        ############################################################################

        print("Recording LCOD Values")

        ## Silently record results if argument record is true (default), else return all outputs

        # Prepare output of this calculation (for recording or returning)
        ## Mean
        all_outputs_mean = pd.concat([
                        pd.Series(lco_vehicle_capex.mean(axis=1)),
                        pd.Series(lco_vehicle_scrappage.mean(axis=1)),
                        pd.Series(lco_om.mean(axis=1)),
                        pd.Series(lco_vehicle_energy.mean(axis=1)),
                        pd.Series(lcod.mean(axis=1))],
                        axis=1)
        all_outputs_mean.columns = ep.getOutputs()
        ## Standard Deviation
        all_outputs_std = pd.concat([
                        pd.Series(lco_vehicle_capex.std(axis=1)),
                        pd.Series(lco_vehicle_scrappage.std(axis=1)),
                        pd.Series(lco_om.std(axis=1)),
                        pd.Series(lco_vehicle_energy.std(axis=1)),
                        pd.Series(lcod.std(axis=1))],
                        axis=1)
        all_outputs_std.columns = ep.getOutputs()

        if record:
            # Record all results of this scenario in one file for each type of result
            io.recordOutput(pd.Series(lco_vehicle_capex.mean(axis=1)), scenario, "TCOVehicleCAPEX", "mean")
            io.recordOutput(pd.Series(lco_vehicle_scrappage.mean(axis=1)), scenario, "TCOVehicleScrappage", "mean")
            io.recordOutput(pd.Series(lco_om.mean(axis=1)), scenario, "TCOOM", "mean")
            io.recordOutput(pd.Series(lco_vehicle_energy.mean(axis=1)), scenario, "TCOVehicleEnergy", "mean")
            io.recordOutput(pd.Series(lcod.mean(axis=1)), scenario, "TCO", "mean")
            io.recordOutput(pd.Series(lco_vehicle_capex.std(axis=1)), scenario, "TCOVehicleCAPEX", "std")
            io.recordOutput(pd.Series(lco_vehicle_scrappage.std(axis=1)), scenario, "TCOVehicleScrappage", "std")
            io.recordOutput(pd.Series(lco_om.std(axis=1)), scenario, "TCOOM", "std")
            io.recordOutput(pd.Series(lco_vehicle_energy.std(axis=1)), scenario, "TCOVehicleEnergy", "std")
            io.recordOutput(pd.Series(lcod.std(axis=1)), scenario, "TCO", "std")

            # Record all results of this scenario in one file
            io.recordScenario(all_outputs_mean, scenario, "mean")
            io.recordScenario(all_outputs_std, scenario, "std")

        else:
            return all_outputs_mean

def calculateLCOC(scenario, year, country, application, sale_status, n_points, n_draws, costOfCapital, vehicleEnergyConsumption, annualKmTravelled, vehicleLifetime, sas_component_capacities):
    """
    This function: calculates the LCOC for the BEV technology in all application
    segments and countries.

    1. Compute TCO and intermediate results:

        LCOC = ( ∑(n) ( CostSolarPVCAPEX*CapSolarPV + CostInverterCAPEX*CapInverter + CostStationaryBatteryCAPEX*CapStationaryBattery + CostSolarPVOM*CapSolarPV + CostInvStatBattOM*CapStationaryBattery ) / (1+i)^n ) / ∑(n) ( TotalChargingEnergy / (1+i)^n )
        LCOCSolarPVCAPEX = (∑(n) CostSolarPVCAPEX*CapSolarPV / (1+i)^n) / ∑(n) (TotalChargingEnergy / (1+i)^n)
        LCOCInverterCAPEX = (∑(n) CostInverterCAPEX*CapInverter / (1+i)^n) / ∑(n) (TotalChargingEnergy / (1+i)^n)
        LCOCStationaryBatteryCAPEX = (∑(n) CostStationaryBatteryCAPEX*CapStationaryBattery / (1+i)^n) / ∑(n) (TotalChargingEnergy / (1+i)^n)
        LCOCSolarPVOM = (∑(n) CostSolarPVOPEX*CapSolarPV / (1+i)^n) / ∑(n) (TotalChargingEnergy / (1+i)^n)
        LCOCInvStatBattOM = (∑(n) CostInvStatBattOPEX*CapStationaryBattery / (1+i)^n) / ∑(n) (TotalChargingEnergy / (1+i)^n)

    """

    #print("Computing LCOC : CostSolarPVCAPEX")
    # Cost Solar PV CAPEX [USD/kWp]
    mode = ep.getValues(scenario, 'CostSolarPVCAPEX', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'CostSolarPVCAPEX', 'max').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'CostSolarPVCAPEX', 'min').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    costSolarPVCAPEX = pert_samples(min, mode, max, (n_points,n_draws))[0]

    #print("Computing LCOC : CostInverterCAPEX")
    # Cost Inverter CAPEX [USD/kWp]
    mode = ep.getValues(scenario, 'CostInverterCAPEX', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'CostInverterCAPEX', 'max').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'CostInverterCAPEX', 'min').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    costInverterCAPEX = pert_samples(min, mode, max, (n_points,n_draws))[0]

    #print("Computing LCOC : CostStationaryBatteryCAPEX")
    # Cost Stationary Battery CAPEX [USD/kWh]
    mode = ep.getValues(scenario, 'CostStationaryBatteryCAPEX', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'CostStationaryBatteryCAPEX', 'max').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'CostStationaryBatteryCAPEX', 'min').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    costStationaryBatteryCAPEX = pert_samples(min, mode, max, (n_points,n_draws))[0]

    #print("Computing LCOC : CostBOSCAPEX")
    # Cost Balance of System CAPEX [USD/kWp]
    mode = ep.getValues(scenario, 'CostBOSCAPEX', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'CostBOSCAPEX', 'max').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'CostBOSCAPEX', 'min').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    costBOSCAPEX = pert_samples(min, mode, max, (n_points,n_draws))[0]

    #print("Computing LCOC : CostInstallationCAPEX")
    # Cost Installation CAPEX [USD]
    mode = ep.getValues(scenario, 'CostInstallationCAPEX', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'CostInstallationCAPEX', 'max').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'CostInstallationCAPEX', 'min').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    costInstallationCAPEX = pert_samples(min, mode, max, (n_points,n_draws))[0]

    #print("Computing LCOC : CostSolarPVOPEX")
    # Cost Solar PV OPEX [USD/kWp*y]
    mode = ep.getValues(scenario, 'CostSolarPVOPEX', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'CostSolarPVOPEX', 'max').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'CostSolarPVOPEX', 'min').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    costSolarPVOPEX = pert_samples(min, mode, max, (n_points,n_draws))[0]

    #print("Computing LCOC : CostInvStatBattOPEX")
    # Cost Inverter/Stationary Battery OPEX [USD/kWh*y]
    mode = ep.getValues(scenario, 'CostInvStatBattOPEX', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'CostInvStatBattOPEX', 'max').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'CostInvStatBattOPEX', 'min').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    costInvStatBattOPEX = pert_samples(min, mode, max, (n_points,n_draws))[0]

    #print("Computing LCOC : SolarPVLifetime")
    # Solar PV Lifetime
    mode = ep.getValues(scenario, 'SolarPVLifetime', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'SolarPVLifetime', 'max').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'SolarPVLifetime', 'min').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    solarPVLifetime = np.round(pert_samples(min, mode, max, (n_points,n_draws)))[0]

    #print("Computing LCOC : InverterLifetime")
    # Inverter Lifetime
    mode = ep.getValues(scenario, 'InverterLifetime', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'InverterLifetime', 'max').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'InverterLifetime', 'min').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    inverterLifetime = np.round(pert_samples(min, mode, max, (n_points,n_draws)))[0]

    #print("Computing LCOC : StationaryBatteryLifetime")
    # Stationary Battery Lifetime
    mode = ep.getValues(scenario, 'StationaryBatteryLifetime', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'StationaryBatteryLifetime', 'max').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'StationaryBatteryLifetime', 'min').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    stationaryBatteryLifetime = np.round(pert_samples(min, mode, max, (n_points,n_draws)))[0]

    #print("Computing LCOC : SASSystemLifetime")
    # SAS System Lifetime
    mode = ep.getValues(scenario, 'SASSystemLifetime', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    max = ep.getValues(scenario, 'SASSystemLifetime', 'max').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    min = ep.getValues(scenario, 'SASSystemLifetime', 'min').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    SASSystemLifetime = np.round(pert_samples(min, mode, max, (n_points,n_draws)))[0]

    #print("Computing LCOC : SolarPVCapacity")
    # Solar PV Capacity
    mode = sas_component_capacities[0]
    max = mode*1
    min = mode*1
    oversize_factor = ep.getValues(scenario, 'SolarPVOversizeFactor', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    solarPVCapacity = pert_samples(min, mode, max, (n_points,n_draws))[0]*(1 + oversize_factor)

    #print("Computing LCOC : InverterCapacity")
    # Inverter Capacity
    mode = sas_component_capacities[1]
    max = mode*1
    min = mode*1
    oversize_factor = ep.getValues(scenario, 'InverterOversizeFactor', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    inverterCapacity = pert_samples(min, mode, max, (n_points,n_draws))[0]*(1 + oversize_factor)

    #print("Computing LCOC : StationaryBatteryCapacity")
    # Stationary Battery Capacity
    mode = sas_component_capacities[2]
    max = mode*1
    min = mode*1
    oversize_factor = ep.getValues(scenario, 'StatBatteryOversizeFactor', 'mode').loc[(year, country, sale_status, application, 'BEV'), 'BaseValue']
    stationaryBatteryCapacity = pert_samples(min, mode, max, (n_points,n_draws))[0]*(1 + oversize_factor)


    ############################################################################
    ############################################################################
    # Compute LCOC components and the total LCOC

    #print("Computing LCOC : total charging energy summation")
    # Calculate the total charging energy summation function
    total_charging_energy_summation = vehicleEnergyConsumption*annualKmTravelled
    total_charging_energy_summation = arrayBroadcastingFunction(total_charging_energy_summation, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[False, False])

    #print("Computing LCOC : LCOCSolarPVCAPEX")
    # Calculate the levelized cost of CAPEX for the solar PV pannel (LCOCSolarPVCAPEX)
    lcoc_solar_pv_capex = costSolarPVCAPEX.copy()*solarPVCapacity
    cond_matrix = constructConditionMatrixFromArrays(SASSystemLifetime, solarPVLifetime)
    lcoc_solar_pv_capex = arrayBroadcastingFunction(lcoc_solar_pv_capex, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix])
    lcoc_solar_pv_capex /= total_charging_energy_summation

    #print("Computing LCOC : LCOCInverterCAPEX")
    # Calculate the levelized cost of CAPEX for the inverter (LCOCInverterCAPEX)
    lcoc_inverter_capex = costInverterCAPEX.copy()*inverterCapacity
    cond_matrix = constructConditionMatrixFromArrays(SASSystemLifetime, inverterLifetime)
    lcoc_inverter_capex = arrayBroadcastingFunction(lcoc_inverter_capex, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix])
    lcoc_inverter_capex /= total_charging_energy_summation

    #print("Computing LCOC : LCOCStationaryBatteryCAPEX")
    # Calculate the levelized cost of CAPEX for the stationary battery (LCOCStationaryBatteryCAPEX)
    lcoc_stationary_battery_capex = costStationaryBatteryCAPEX.copy()*stationaryBatteryCapacity
    cond_matrix = constructConditionMatrixFromArrays(SASSystemLifetime, stationaryBatteryLifetime)
    lcoc_stationary_battery_capex = arrayBroadcastingFunction(lcoc_stationary_battery_capex, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix])
    lcoc_stationary_battery_capex /= total_charging_energy_summation

    #print("Computing LCOC : LCOCBOSCAPEX")
    # Calculate the levelized cost of CAPEX for the balance of system (LCOCBOSCAPEX)
    lcoc_bos_capex = costBOSCAPEX.copy()*solarPVCapacity
    cond_matrix = constructConditionMatrixFromArrays(SASSystemLifetime, solarPVLifetime)
    lcoc_bos_capex = arrayBroadcastingFunction(lcoc_bos_capex, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix])
    lcoc_bos_capex /= total_charging_energy_summation

    #print("Computing LCOC : LCOCInstallationCAPEX")
    # Calculate the levelized cost of CAPEX for the installation (LCOCInstallationCAPEX)
    lcoc_installation_capex = costInstallationCAPEX.copy()*(costSolarPVCAPEX.copy() + costInverterCAPEX.copy() + costStationaryBatteryCAPEX.copy() + costBOSCAPEX.copy())
    cond_matrix = constructConditionMatrixFromArrays(SASSystemLifetime, solarPVLifetime)
    lcoc_installation_capex = arrayBroadcastingFunction(lcoc_installation_capex, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix])
    lcoc_installation_capex /= total_charging_energy_summation

    #print("Computing LCOC : LCOCSolarPVOM")
    # Calculate the levelized cost of O&M for the solar PV pannel (LCOCSolarPVOM)
    lcoc_solar_pv_om = costSolarPVOPEX.copy()*solarPVCapacity
    lcoc_solar_pv_om = arrayBroadcastingFunction(lcoc_solar_pv_om, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[False, False])
    lcoc_solar_pv_om /= total_charging_energy_summation

    #print("Computing LCOC : LCOCInvStatBattOM")
    # Calculate the levelized cost of O&M for the inverter and stationary battery (LCOCInvStatBattOM)
    lcoc_inv_stat_batt_om = costInvStatBattOPEX.copy()*stationaryBatteryCapacity
    lcoc_inv_stat_batt_om = arrayBroadcastingFunction(lcoc_inv_stat_batt_om, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[False, False])
    lcoc_inv_stat_batt_om /= total_charging_energy_summation

    # Calculate the levelized cost of charging
    lcoc = lcoc_solar_pv_capex + lcoc_inverter_capex + lcoc_stationary_battery_capex + lcoc_bos_capex + lcoc_installation_capex + lcoc_solar_pv_om + lcoc_inv_stat_batt_om

    return lcoc

def calculatePresentValueFactor(costOfCapital_matrix, vehicleLifetime_matrix):
    presentValueFactor = 1/np.power(1+costOfCapital_matrix, vehicleLifetime_matrix)

    return presentValueFactor

def calculateCapitalRecoveryFactor(costOfCapital_matrix, vehicleLifetime_matrix):
    capitalRecoveryFactor = (costOfCapital_matrix * np.power(1+costOfCapital_matrix, vehicleLifetime_matrix)) / (np.power(1+costOfCapital_matrix, vehicleLifetime_matrix) - 1)

    return capitalRecoveryFactor

def constructConditionMatrixFromArrays(system_lifetime_array, component_lifetime_array):
    """
    Constructs a condition matrix that indicates when components need to be replaced
    within the system lifetime, considering element-wise component lifetimes.

    system_lifetime_array : np.ndarray
        Array of the system lifetime values, shape (1, N_draws).
    component_lifetime_array : np.ndarray
        Array of the component lifetime values, shape (1, N_draws).

    Returns
    -------
    condition_matrix : np.ndarray
        A boolean matrix indicating where components need to be replaced, shape (N_draws, N_max).
    """
    # Extract N_draws from the array shape
    N_draws = system_lifetime_array.shape[0]

    # Determine the maximum system lifetime to define the range of n values
    N_max = int(system_lifetime_array.max())

    # Create an array representing each year from 1 to N_max with shape (1, N_max)
    n_values = np.arange(1, N_max + 1).reshape(1, N_max)

    # Reshape system_lifetime_array and component_lifetime_array to (N_draws, 1)
    sys_life_expanded = system_lifetime_array.reshape(N_draws, 1)
    comp_life_expanded = component_lifetime_array.reshape(N_draws, 1)

    # Generate the replacement years condition using vectorized operations
    replacement_years = ((n_values % comp_life_expanded == 0) & (n_values < sys_life_expanded)) | (n_values == 1)

    # Apply the condition by masking out the years beyond the system lifetime
    condition_matrix = np.where(n_values <= sys_life_expanded, replacement_years, False)

    return condition_matrix.reshape(N_draws, 1, N_max)

def matrixBroadcastingFunction(X_matrix, i_matrix, N_matrix, div_bool, cond_matrix_bool):
    """
    NOTE:
    X_matrix : opperating parameter (could be CostVehicleO&M or CostVehicleEnergy or VehicleAnnualKmTravelled)
    i_matrix : cost of capital
    N_matrix : vehicle lifetime
    condition_matrix: condition to determine if a value should be included in the summation
    """

    # Determine the maximum lifetime (N) to know the range of n values we need for the mask summation
    N_max = N_matrix.max()

    # Generating an array of n values from 1 to N_max
    n_values = np.arange(1, N_max + 1)  # 1-based to include N_max in the calculations

    # Compute (1 + i) to the power of n for each value of n
    # This results in a shape of (5832, 1000, N_max)
    powers = (1 + i_matrix[:, :, np.newaxis]) ** n_values

    # Calculate the inverse of these powers
    inv_powers = 1 / powers

    # Compute the result using a pre-calculated conditional matrix
    if cond_matrix_bool[0]:
        # Use the mask to apply the calculation only where valid and sum over the n-dimension (axis=2)
        result = (X_matrix[:, :, np.newaxis] * inv_powers * cond_matrix_bool[1]).sum(axis=2)
    # Compute the result using a mask matrix
    else:
        # Create a mask where each slice along the third axis represents whether n is <= N at each element
        mask = n_values <= N_matrix[:, :, np.newaxis]

        # Use the mask to apply the calculation only where valid and sum over the n-dimension (axis=2)
        result = (X_matrix[:, :, np.newaxis] * inv_powers * mask).sum(axis=2)

    if div_bool:
        # Don't forget to divide by the lifetime before returning the result
        result *= 1/N_matrix

    return result

def arrayBroadcastingFunction(X_array, i_array, N_array, div_bool, cond_matrix_bool):
    """
    Modified function to handle 2D array inputs for X_matrix, i_array, and N_array.

    Parameters:
    X_array : 2D array of shape (1, N_draws)
        Operating parameter (could be CostVehicleO&M, CostVehicleEnergy, or VehicleAnnualKmTravelled)
    i_array : 2D array of shape (1, N_draws)
        Array representing the cost of capital for each draw
    N_array : 2D array of shape (1, N_draws)
        Array representing the vehicle lifetime for each draw
    cond_matrix_bool : tuple (boolean, condition_matrix)
        If the boolean is True, a precomputed condition matrix is used; otherwise, a mask is generated dynamically.
    div_bool : boolean
        Indicates whether to divide the result by the vehicle lifetime.

    Returns:
    result : 2D array
        Computed result based on matrix broadcasting operations.
    """

    # Extract N_draws from the shape
    N_draws = X_array.shape[0]

    # Determine the maximum lifetime (N) to know the range of n values we need for the mask summation
    N_max = N_array.max()

    # Generate an array of n values from 1 to N_max
    n_values = np.arange(1, N_max + 1)  # 1-based to include N_max in the calculations

    # Reshape i_array and N_array and X_array to (N_draws, 1) to enable proper broadcasting
    i_array = i_array.reshape(N_draws, 1)
    N_array = N_array.reshape(N_draws, 1)
    X_array = X_array.reshape(N_draws, 1)

    # Compute (1 + i) to the power of n for each value of n (broadcast i_matrix across n)
    powers = (1 + i_array[:, np.newaxis]) ** n_values

    # Calculate the inverse of these powers
    inv_powers = 1 / powers

    # Compute the result using a pre-calculated conditional matrix or generate a mask
    if cond_matrix_bool[0]:
        # Use the mask to apply the calculation only where valid and sum over the n-dimension (axis=1)
        result = (X_array[:, np.newaxis] * inv_powers * cond_matrix_bool[1]).sum(axis=2)
    else:
        # Create a mask where each slice along the third axis represents whether n is <= N at each element
        mask = n_values <= N_array[:, np.newaxis]
        # Use the mask to apply the calculation only where valid and sum over the n-dimension (axis=1)
        result = (X_array[:, np.newaxis] * inv_powers * mask).sum(axis=2)

    if div_bool:
        # Divide by the vehicle lifetime before returning the result
        result *= 1 / N_array

    return result.T


"""
Plot Types:
"""
def lcocDataScatterPlotMap(model_scenario, year, application, sale_status, runTCOBool):
    # Step 1: Load the data (assuming it's a CSV file with 'Longitude', 'Latitude', and 'Cost' columns)
    # Adjust the file path and column names to match your dataset
    xls_SAS_Optimization_results = pd.ExcelFile(INPUT_PATH + "SOGOptimizationData.xlsx")
    SASOptimizationResults2025_data = pd.read_excel(xls_SAS_Optimization_results, '2025')
    SASOptimizationResults2030_data = pd.read_excel(xls_SAS_Optimization_results, '2030')
    SASOptimizationResults2040_data = pd.read_excel(xls_SAS_Optimization_results, '2040')

    data_df = eval(f'SASOptimizationResults{year}_data')

    # Clean the dataframe
    for column in data_df.columns:
        if 'CompCaps' in column:
            data_df[column] = data_df[column].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

    # Step 2: Calculate the TCO for the desired situation (i.e. model scenario, year, application, etc.)
    if runTCOBool:
        print('----------------------------------')
        print('----------------------------------')
        print('Computing TCO Data')
        print('----------------------------------')
        print('----------------------------------')
        all_outputs_mean, all_outputs_std = calculateTCO(data_df, model_scenario, year, application, sale_status, record=True)
    else:
        print('----------------------------------')
        print('----------------------------------')
        print('Loading TCO Data')
        print('----------------------------------')
        print('----------------------------------')
        # Load pre-saved files
        all_outputs_mean = pd.read_pickle(PLOTS_PATH + f'_AfricaZoomPlot_AllOutputs_{model_scenario}_{year}_{sale_status}_{application}_mean.pkl')
        all_outputs_std = pd.read_pickle(PLOTS_PATH + f'_AfricaZoomPlot_AllOutputs_{model_scenario}_{year}_{sale_status}_{application}_mean.pkl')

    print('Plotting LCOC Data Scatter Plot')

    # Step 2: Set up the map of Africa using Cartopy
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    # Initialize the custom colorbar parameters
    boundaries, cmap_colors, cbar_ticks, cbar_tick_labels = create_colorbar_params(-35, 35, 5, "red", "blue")#"#b35656", "#5589a7")
    # Create the custom colormap
    custom_cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm(boundaries, ncolors=custom_cmap.N, clip=True)

    # Set the map extent to focus on Africa (approximate coordinates for Africa)
    ax.set_extent([-25, 60, -35, 40], crs=ccrs.PlateCarree())

    shapefile = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
    reader = shpreader.Reader(shapefile)
    african_countries = [
        country.geometry for country in reader.records()
        if country.attributes['CONTINENT'].strip('\x00') == 'Africa'
    ]
    ax.add_geometries(african_countries, ccrs.PlateCarree(), edgecolor='darkgrey', facecolor='none', linewidth=1)


    # Step 4: Plot the 2558 points with color based on 'Cost'
    longitude_data = data_df['Longitude']
    latitude_data = data_df['Latitude']
    plot_data = all_outputs_mean['LCOC']

    # Use plt.scatter() to plot longitude, latitude, and color them based on the 'Cost' column
    sc = ax.scatter(longitude_data, latitude_data, c=plot_data, cmap='coolwarm', s=80, transform=ccrs.PlateCarree())
    # Plot over with white dots for the non modelled countries
    [ws_lat, ws_long] = getWesternSaharaPoints()
    sc1 = ax.scatter(ws_long, ws_lat, color='White', s=90, transform=ccrs.PlateCarree())
    [ws_lat, ws_long] = getDjiboutiPoints()
    sc1 = ax.scatter(ws_long, ws_lat, color='White', s=90, transform=ccrs.PlateCarree())
    [ws_lat, ws_long] = getSeychellesPoints()
    sc1 = ax.scatter(ws_long, ws_lat, color='White', s=90, transform=ccrs.PlateCarree())

    # Step 5: Add a colorbar
    ####### Standard #######
    #cbar = plt.colorbar(sc, orientation='vertical', pad=0.05, fraction=0.01)
    cbar = plt.colorbar(
                        sc,
                        ax=ax,
                        orientation='vertical',
                        fraction=0.035,  # Increase this for a longer bar
                        shrink=0.5,    # Expand the bar's length
                        aspect=20,     # Decrease this for a wider bar
                        pad=0.01       # Adjust spacing between bar and plot
                    )
    #cbar.set_label('[USD/kWh]')
    cbar.ax.xaxis.set_label_position('top')  # Move label to the bottom
    cbar.ax.set_xlabel('[USD/kWh]', labelpad=20, fontsize=13, loc='center')

    # Step 6: Remove the box around the figure
    ax.spines['geo'].set_visible(False)

    # Step 7: Adjust the figure spacing
    fig.subplots_adjust(left=0.035, right=0.915, top=0.93, bottom=0.015, wspace=0.2, hspace=0.2)

    # Show the plot
    plt.title(f'Levelized Cost of Charging Across Africa \n [{model_scenario}, {year}, {sale_status}, {application}]')

    # Save figue
    filename = PLOTS_PATH + f'0_AfricaZoomPlot_LCOC_{model_scenario}_{year}_{sale_status}_{application}.png'
    dpi = 300
    plt.savefig(filename, dpi=dpi)

    plt.show()


"""
Plot Calls:

Each model year can be run individually. Refer to externalParameters.py to adjust
the number of Monte Carlo draws desired.
"""

# 2025
lcocDataScatterPlotMap(model_scenario='BaseScenario', year=2025, application='4wheeler-small', sale_status='New', runTCOBool=True)
# # 2030
# lcocDataScatterPlotMap(model_scenario='BaseScenario', year=2030, application='4wheeler-small', sale_status='New', runTCOBool=True)
# # 2040
# lcocDataScatterPlotMap(model_scenario='BaseScenario', year=2040, application='4wheeler-small', sale_status='New', runTCOBool=True)



print('---------------------------------------------------------------')
print('CODE RUN TIME:')
print('--- ',np.around((time.time() - start_time), decimals=3), 'seconds ---')
print('--- ',np.around((time.time() - start_time)/60, decimals=3), 'minutes ---')
print('---------------------------------------------------------------')









#






#
