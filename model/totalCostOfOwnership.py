# FILE: totalCostOfOwnership.py
# PROJECT: Africa Passenger Vehicle TCO Model
# SCRIPT DESCRIPTION: This is the TCO Module that computes the total cost
# of ownership for all countries, applications and technologies, and time horizons


# Import statements
import pandas as pd
import numpy as np
import externalParameters as ep
import intermediateOutputs as io
from scipy.stats import truncnorm


def getParameterSet(parameter_codes):
    """
    This function
    takes: a Series of codes for the value selection of the model parameters

    returns: a multiindex DataFrame with columns containing the corresponding
    parameter values (i.e. one specific parameter set of a scenario)
    """

    # Initialize empty dataframe to take parameter values
    parameter_set = ep.getEmptyDataFrame('parameters')

    # Gather all parameter values corresponding to the passed parameter codes
    for parameter in ep.getParameters():

        # Get scenario code of current parameter
        curr_code = parameter_codes.loc[parameter]
        # Get corresponding parameter values
        curr_parameters = ep.getValues(parameter, curr_code)
        # Add parameter values to the parameter set

        parameter_set.loc[:,parameter] = curr_parameters.values

    return parameter_set


def calculateLCOC(scenario, n_draws, costOfCapital, vehicleEnergyConsumption, annualKmTravelled, vehicleLifetime):
    """
    This function: calculates the LCOC for the BEV technology in all application
    segments and countries.

    1. Compute LCOC and intermediate results:

        LCOC = ( ∑(n) ( CostSolarPVCAPEX*CapSolarPV + CostInverterCAPEX*CapInverter + CostStationaryBatteryCAPEX*CapStationaryBattery + CostSolarPVOM*CapSolarPV + CostInvStatBattOM*CapStationaryBattery ) / (1+i)^n ) / ∑(n) ( TotalChargingEnergy / (1+i)^n )
        LCOCSolarPVCAPEX = (∑(n) CostSolarPVCAPEX*CapSolarPV / (1+i)^n) / ∑(n) (TotalChargingEnergy / (1+i)^n)
        LCOCInverterCAPEX = (∑(n) CostInverterCAPEX*CapInverter / (1+i)^n) / ∑(n) (TotalChargingEnergy / (1+i)^n)
        LCOCStationaryBatteryCAPEX = (∑(n) CostStationaryBatteryCAPEX*CapStationaryBattery / (1+i)^n) / ∑(n) (TotalChargingEnergy / (1+i)^n)
        LCOCSolarPVOM = (∑(n) CostSolarPVOPEX*CapSolarPV / (1+i)^n) / ∑(n) (TotalChargingEnergy / (1+i)^n)
        LCOCInvStatBattOM = (∑(n) CostInvStatBattOPEX*CapStationaryBattery / (1+i)^n) / ∑(n) (TotalChargingEnergy / (1+i)^n)

    2. Compute SAS CAPEX and OPEX without discounting energy for use in TCO equation NOT as LCOC (ownership perspective)

        DiscountedSolarPVCAPEX = (∑(n) CostSolarPVCAPEX*CapSolarPV / (1+i)^n)
        DiscountedInverterCAPEX = (∑(n) CostInverterCAPEX*CapInverter / (1+i)^n)
        DiscountedStationaryBatteryCAPEX = (∑(n) CostStationaryBatteryCAPEX*CapStationaryBattery / (1+i)^n)
        SolarPVOM = CostSolarPVOPEX*CapSolarPV
        InvStatBattOM = CostInvStatBattOPEX*CapStationaryBattery

    """

    print("Computing LCOC : CostSolarPVCAPEX")
    # Cost Solar PV CAPEX [USD/kWp]
    mode = ep.getValues(scenario, 'CostSolarPVCAPEX', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'CostSolarPVCAPEX', 'max').values.astype(float)
    min = ep.getValues(scenario, 'CostSolarPVCAPEX', 'min').values.astype(float)
    costSolarPVCAPEX = pert_samples(min, mode, max, (mode.size,n_draws))

    print("Computing LCOC : CostInverterCAPEX")
    # Cost Inverter CAPEX [USD/kWp]
    mode = ep.getValues(scenario, 'CostInverterCAPEX', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'CostInverterCAPEX', 'max').values.astype(float)
    min = ep.getValues(scenario, 'CostInverterCAPEX', 'min').values.astype(float)
    costInverterCAPEX = pert_samples(min, mode, max, (mode.size,n_draws))

    print("Computing LCOC : CostStationaryBatteryCAPEX")
    # Cost Stationary Battery CAPEX [USD/kWh]
    mode = ep.getValues(scenario, 'CostStationaryBatteryCAPEX', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'CostStationaryBatteryCAPEX', 'max').values.astype(float)
    min = ep.getValues(scenario, 'CostStationaryBatteryCAPEX', 'min').values.astype(float)
    costStationaryBatteryCAPEX = pert_samples(min, mode, max, (mode.size,n_draws))

    print("Computing LCOC : CostBOSCAPEX")
    # Cost Balance of System CAPEX [USD/kWp]
    mode = ep.getValues(scenario, 'CostBOSCAPEX', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'CostBOSCAPEX', 'max').values.astype(float)
    min = ep.getValues(scenario, 'CostBOSCAPEX', 'min').values.astype(float)
    costBOSCAPEX = pert_samples(min, mode, max, (mode.size,n_draws))

    print("Computing LCOC : CostInstallationCAPEX")
    # Cost Installation CAPEX [USD]
    mode = ep.getValues(scenario, 'CostInstallationCAPEX', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'CostInstallationCAPEX', 'max').values.astype(float)
    min = ep.getValues(scenario, 'CostInstallationCAPEX', 'min').values.astype(float)
    costInstallationCAPEX = pert_samples(min, mode, max, (mode.size,n_draws))

    print("Computing LCOC : CostSolarPVOPEX")
    # Cost Solar PV OPEX [USD/kWp*y]
    mode = ep.getValues(scenario, 'CostSolarPVOPEX', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'CostSolarPVOPEX', 'max').values.astype(float)
    min = ep.getValues(scenario, 'CostSolarPVOPEX', 'min').values.astype(float)
    costSolarPVOPEX = pert_samples(min, mode, max, (mode.size,n_draws))

    print("Computing LCOC : CostInvStatBattOPEX")
    # Cost Inverter/Stationary Battery OPEX [USD/kWh*y]
    mode = ep.getValues(scenario, 'CostInvStatBattOPEX', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'CostInvStatBattOPEX', 'max').values.astype(float)
    min = ep.getValues(scenario, 'CostInvStatBattOPEX', 'min').values.astype(float)
    costInvStatBattOPEX = pert_samples(min, mode, max, (mode.size,n_draws))

    print("Computing LCOC : SolarPVLifetime")
    # Solar PV Lifetime
    mode = ep.getValues(scenario, 'SolarPVLifetime', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'SolarPVLifetime', 'max').values.astype(float)
    min = ep.getValues(scenario, 'SolarPVLifetime', 'min').values.astype(float)
    solarPVLifetime = np.round(pert_samples(min, mode, max, (mode.size,n_draws)))

    print("Computing LCOC : InverterLifetime")
    # Inverter Lifetime
    mode = ep.getValues(scenario, 'InverterLifetime', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'InverterLifetime', 'max').values.astype(float)
    min = ep.getValues(scenario, 'InverterLifetime', 'min').values.astype(float)
    inverterLifetime = np.round(pert_samples(min, mode, max, (mode.size,n_draws)))

    print("Computing LCOC : StationaryBatteryLifetime")
    # Stationary Battery Lifetime
    mode = ep.getValues(scenario, 'StationaryBatteryLifetime', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'StationaryBatteryLifetime', 'max').values.astype(float)
    min = ep.getValues(scenario, 'StationaryBatteryLifetime', 'min').values.astype(float)
    stationaryBatteryLifetime = np.round(pert_samples(min, mode, max, (mode.size,n_draws)))

    print("Computing LCOC : SASSystemLifetime")
    # SAS System Lifetime
    mode = ep.getValues(scenario, 'SASSystemLifetime', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'SASSystemLifetime', 'max').values.astype(float)
    min = ep.getValues(scenario, 'SASSystemLifetime', 'min').values.astype(float)
    SASSystemLifetime = np.round(pert_samples(min, mode, max, (mode.size,n_draws)))

    print("Computing LCOC : SolarPVCapacity")
    # Solar PV Capacity [kWp]
    mode = ep.getValues(scenario, 'SolarPVCapacity', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'SolarPVCapacity', 'max').values.astype(float)
    min = ep.getValues(scenario, 'SolarPVCapacity', 'min').values.astype(float)
    oversize_factor = ep.getValues(scenario, 'SolarPVOversizeFactor', 'mode').values.astype(float)
    solarPVCapacity = pert_samples(min, mode, max, (mode.size,n_draws))*(1 + oversize_factor)
    solarPVCapacity[np.isnan(solarPVCapacity)] = 0

    print("Computing LCOC : InverterCapacity")
    # Inverter Capacity [kWp]
    mode = ep.getValues(scenario, 'InverterCapacity', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'InverterCapacity', 'max').values.astype(float)
    min = ep.getValues(scenario, 'InverterCapacity', 'min').values.astype(float)
    oversize_factor = ep.getValues(scenario, 'InverterOversizeFactor', 'mode').values.astype(float)
    inverterCapacity = pert_samples(min, mode, max, (mode.size,n_draws))*(1 + oversize_factor)
    inverterCapacity[np.isnan(inverterCapacity)] = 0

    print("Computing LCOC : StationaryBatteryCapacity")
    # Stationary Battery Capacity [kWh]
    mode = ep.getValues(scenario, 'StationaryBatteryCapacity', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'StationaryBatteryCapacity', 'max').values.astype(float)
    min = ep.getValues(scenario, 'StationaryBatteryCapacity', 'min').values.astype(float)
    oversize_factor = ep.getValues(scenario, 'StatBatteryOversizeFactor', 'mode').values.astype(float)
    stationaryBatteryCapacity = pert_samples(min, mode, max, (mode.size,n_draws))*(1 + oversize_factor)
    stationaryBatteryCapacity[np.isnan(stationaryBatteryCapacity)] = 0


    ############################################################################
    ############################################################################
    # Compute LCOC components and the total LCOC

    print("Computing LCOC : total charging energy summation")
    # Calculate the total charging energy summation function
    total_charging_energy_summation = vehicleEnergyConsumption*annualKmTravelled
    total_charging_energy_summation = matrixBroadcastingFunction(total_charging_energy_summation, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[False, False])

    # Define the size of the storage matrix
    n_rows = total_charging_energy_summation.shape[0]
    n_cols = total_charging_energy_summation.shape[1]

    print("Computing LCOC : LCOCSolarPVCAPEX")
    # Calculate the levelized cost of CAPEX for the solar PV pannel (LCOCSolarPVCAPEX)
    solar_pv_capex = costSolarPVCAPEX.copy()*solarPVCapacity
    cond_matrix = construct_condition_matrix(SASSystemLifetime, solarPVLifetime, n_rows, n_cols)
    cond_matrix_tco = construct_condition_matrix(vehicleLifetime, solarPVLifetime, n_rows, n_cols)
    discounted_solar_pv_capex = matrixBroadcastingFunction(solar_pv_capex, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix])
    discounted_solar_pv_capex_tco = matrixBroadcastingFunction(solar_pv_capex, costOfCapital, vehicleLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix_tco])
    lcoc_solar_pv_capex = discounted_solar_pv_capex / total_charging_energy_summation

    print("Computing LCOC : LCOCInverterCAPEX")
    # Calculate the levelized cost of CAPEX for the inverter (LCOCInverterCAPEX)
    inverter_capex = costInverterCAPEX.copy()*inverterCapacity
    cond_matrix = construct_condition_matrix(SASSystemLifetime, inverterLifetime, n_rows, n_cols)
    cond_matrix_tco = construct_condition_matrix(vehicleLifetime, inverterLifetime, n_rows, n_cols)
    discounted_inverter_capex = matrixBroadcastingFunction(inverter_capex, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix])
    discounted_inverter_capex_tco = matrixBroadcastingFunction(inverter_capex, costOfCapital, vehicleLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix_tco])
    lcoc_inverter_capex = discounted_inverter_capex / total_charging_energy_summation

    print("Computing LCOC : LCOCStationaryBatteryCAPEX")
    # Calculate the levelized cost of CAPEX for the stationary battery (LCOCStationaryBatteryCAPEX)
    stationary_battery_capex = costStationaryBatteryCAPEX.copy()*stationaryBatteryCapacity
    cond_matrix = construct_condition_matrix(SASSystemLifetime, stationaryBatteryLifetime, n_rows, n_cols)
    cond_matrix_tco = construct_condition_matrix(vehicleLifetime, stationaryBatteryLifetime, n_rows, n_cols)
    discounted_stationary_battery_capex = matrixBroadcastingFunction(stationary_battery_capex, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix])
    discounted_stationary_battery_capex_tco = matrixBroadcastingFunction(stationary_battery_capex, costOfCapital, vehicleLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix_tco])
    lcoc_stationary_battery_capex = discounted_stationary_battery_capex / total_charging_energy_summation

    print("Computing LCOC : LCOCBOSCAPEX")
    # Calculate the levelized cost of CAPEX for the balance of system (LCOCBOSCAPEX)
    bos_capex = costBOSCAPEX.copy()*solarPVCapacity
    cond_matrix = construct_condition_matrix(SASSystemLifetime, solarPVLifetime, n_rows, n_cols)
    cond_matrix_tco = construct_condition_matrix(vehicleLifetime, solarPVLifetime, n_rows, n_cols)
    discounted_bos_capex = matrixBroadcastingFunction(bos_capex, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix])
    discounted_bos_capex_tco = matrixBroadcastingFunction(bos_capex, costOfCapital, vehicleLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix_tco])
    lcoc_bos_capex = discounted_bos_capex / total_charging_energy_summation

    print("Computing LCOC : LCOCInstallationCAPEX")
    # Calculate the levelized cost of CAPEX for the system installation (LCOCInstallationCAPEX)
    installation_capex = costInstallationCAPEX.copy()*(solar_pv_capex + inverter_capex + stationary_battery_capex + bos_capex)
    cond_matrix = construct_condition_matrix(SASSystemLifetime, solarPVLifetime, n_rows, n_cols)
    cond_matrix_tco = construct_condition_matrix(vehicleLifetime, solarPVLifetime, n_rows, n_cols)
    discounted_installation_capex = matrixBroadcastingFunction(installation_capex, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix])
    discounted_installation_capex_tco = matrixBroadcastingFunction(installation_capex, costOfCapital, vehicleLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix_tco])
    lcoc_installation_capex = discounted_installation_capex / total_charging_energy_summation

    print("Computing LCOC : LCOCSolarPVOM")
    # Calculate the levelized cost of O&M for the solar PV pannel (LCOCSolarPVOM)
    solar_pv_om = costSolarPVOPEX.copy()*solarPVCapacity    # [USD/year]
    lcoc_solar_pv_om = matrixBroadcastingFunction(solar_pv_om, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[False, False])
    lcoc_solar_pv_om /= total_charging_energy_summation

    print("Computing LCOC : LCOCInvStatBattOM")
    # Calculate the levelized cost of O&M for the inverter and stationary battery (LCOCInvStatBattOM)
    inv_stat_batt_om = costInvStatBattOPEX.copy()*stationaryBatteryCapacity    # [USD/year]
    lcoc_inv_stat_batt_om = matrixBroadcastingFunction(inv_stat_batt_om, costOfCapital, SASSystemLifetime, div_bool=False, cond_matrix_bool=[False, False])
    lcoc_inv_stat_batt_om /= total_charging_energy_summation

    # Calculate the levelized cost of charging
    lcoc = lcoc_solar_pv_capex + lcoc_inverter_capex + lcoc_stationary_battery_capex + lcoc_bos_capex + lcoc_installation_capex + lcoc_solar_pv_om + lcoc_inv_stat_batt_om

    return (lcoc, lcoc_solar_pv_capex, lcoc_inverter_capex, lcoc_stationary_battery_capex, lcoc_bos_capex, lcoc_installation_capex, lcoc_solar_pv_om, lcoc_inv_stat_batt_om,
            np.nan_to_num(discounted_solar_pv_capex_tco, nan=0.0), np.nan_to_num(discounted_inverter_capex_tco, nan=0.0), np.nan_to_num(discounted_stationary_battery_capex_tco, nan=0.0),
            np.nan_to_num(discounted_bos_capex_tco, nan=0.0), np.nan_to_num(discounted_installation_capex_tco, nan=0.0), solar_pv_om, inv_stat_batt_om)


def calculateTCO(scenario="NoScenario", record=True):
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

    ## Preparation

    # Set scenario
    setScenario(scenario)

    # Initialize the number of Monte Carlo draws
    n_draws = ep.getMonteCarloDraws()

    ## Calculate all distributed input parameters for the specific scenario
    print("Computing VehicleAnnualKmTravelled")
    # Annual Kilometers Travelled [km]
    mode = ep.getValues(scenario, 'VehicleAnnualKmTravelled', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'VehicleAnnualKmTravelled', 'max').values.astype(float)
    min = ep.getValues(scenario, 'VehicleAnnualKmTravelled', 'min').values.astype(float)
    annualKmTravelled = pert_samples(min, mode, max, (mode.size,n_draws))

    print("Computing VehicleLifetime")
    # Vehicle Lifetime [years]
    mode = ep.getValues(scenario, 'VehicleLifetime', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'VehicleLifetime', 'max').values.astype(float)
    min = ep.getValues(scenario, 'VehicleLifetime', 'min').values.astype(float)
    vehicleLifetime = np.round(pert_samples(min, mode, max, (mode.size,n_draws)))

    print("Computing CostOfCapital")
    # Cost of Capital [unitless]
    mean = ep.getValues(scenario, 'CostOfCapital', 'mean').values.astype(float)
    std = ep.getValues(scenario, 'CostOfCapital', 'std').values.astype(float)
    costOfCapital = np.random.normal(loc=mean, scale=std, size=(mean.size,n_draws))

    # Near Zero or "Risk-Free" Cost of Capital (i.e. i = 0.00001%) [unitless]
    nearZeroCostOfCapital = costOfCapital/costOfCapital*0.014

    print("Computing presentValueFactor")
    # Present Value Factor [unitless]
    presentValueFactor = calculatePresentValueFactor(costOfCapital, vehicleLifetime)

    print("Computing capitalRecoveryFactor")
    # Capital Recovery Factor [unitless]
    capitalRecoveryFactor = calculateCapitalRecoveryFactor(costOfCapital, vehicleLifetime)

    # Near Zero Capital Recovery Factor [unitless]
    nearZeroCapitalRecoveryFactor = calculateCapitalRecoveryFactor(nearZeroCostOfCapital, vehicleLifetime)

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
    mode = ep.getValues(scenario, 'VehicleEnergyConsumption', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'VehicleEnergyConsumption', 'max').values.astype(float)
    min = ep.getValues(scenario, 'VehicleEnergyConsumption', 'min').values.astype(float)
    vehicleEnergyConsumption = pert_samples(min, mode, max, (mode.size,n_draws))

    print("Computing CostVehicleCAPEX")
    # Cost Vehicle CAPEX [USD]
    mean = ep.getValues(scenario, 'CostVehicleCAPEX', 'mean').values.astype(float)
    std = ep.getValues(scenario, 'CostVehicleCAPEX', 'std').values.astype(float)
    costVehicleCAPEX = np.random.normal(loc=mean, scale=std, size=(mean.size,n_draws))

    # print("Computing CostVehicleBatteryCAPEX")
    # # Cost Vehicle Battery CAPEX  [USD/kWh]
    # mode = ep.getValues(scenario, 'CostVehicleBatteryCAPEX', 'mode').values.astype(float)
    # max = ep.getValues(scenario, 'CostVehicleBatteryCAPEX', 'max').values.astype(float)
    # min = ep.getValues(scenario, 'CostVehicleBatteryCAPEX', 'min').values.astype(float)
    # costVehicleBatteryCAPEX = pert_samples(min, mode, max, (mode.size,n_draws))

    # print("Computing VehicleBatteryCapacity")
    # # Vehicle Battery Capacity [kWh]
    # mode = ep.getValues(scenario, 'VehicleBatteryCapacity', 'mode').values.astype(float)
    # max = ep.getValues(scenario, 'VehicleBatteryCapacity', 'max').values.astype(float)
    # min = ep.getValues(scenario, 'VehicleBatteryCapacity', 'min').values.astype(float)
    # vehicleBatteryCapacity = pert_samples(min, mode, max, (mode.size,n_draws))

    ############################################################################

    print("Computing CostVehicleScrappage")
    # Cost Vehicle Scrappage [USD]
    a = ep.getValues(scenario, 'CostVehicleScrappage', 'a').values.astype(float)
    b = ep.getValues(scenario, 'CostVehicleScrappage', 'b').values.astype(float)
    x_0 = ep.getValues(scenario, 'CostVehicleScrappage', 'x0').values.astype(float)
    y_0 = ep.getValues(scenario, 'CostVehicleScrappage', 'y0').values.astype(float)
    # Calculate the total vehicle km travelled (over the full vehicle lifetime)
    total_km_travelled = vehicleLifetime * annualKmTravelled
    # Calculate the scrappage value % of retail value
    perc_retail_value = a + (y_0 - a) * np.exp(-b * (total_km_travelled - x_0))
    costVehicleScrappage = costVehicleCAPEX * perc_retail_value

    print("Computing CostVehicleOM")
    # Cost O&M [USD/year]
    mode = ep.getValues(scenario, 'CostVehicleOM', 'mode').values.astype(float)
    max = ep.getValues(scenario, 'CostVehicleOM', 'max').values.astype(float)
    min = ep.getValues(scenario, 'CostVehicleOM', 'min').values.astype(float)
    costOM = pert_samples(min, mode, max, (mode.size,n_draws))

    print("Computing CostVehicleEnergy")
    # Cost Vehicle Energy [USD/year]        NOTE: CostVehicleEnergy is multiplied by vehicleEfficiency and annualKmTravelled to give cost in USD
    mean = ep.getValues(scenario, 'CostVehicleEnergy', 'mean').values.astype(float)
    std = ep.getValues(scenario, 'CostVehicleEnergy', 'std').values.astype(float)
    # Create a truncated normal distribution to avoid negative energy costs
    lower, upper = 0, np.inf
    a, b = (lower - mean) / std, (upper - mean) / std
    trunc_normal_dist = truncnorm(a, b, loc=mean, scale=std)
    costVehicleEnergy = trunc_normal_dist.rvs(size=(mean.size,n_draws)) * vehicleEnergyConsumption * annualKmTravelled

    LCOC_Bool = True
    private_ownership_perspective = True

    if LCOC_Bool:
        print("Computing CostVehicleLCOC")
        (lcoc,
        lcoc_solar_pv_capex,
        lcoc_inverter_capex,
        lcoc_stationary_battery_capex,
        lcoc_bos_capex,
        lcoc_installation_capex,
        lcoc_solar_pv_om,
        lcoc_inv_stat_batt_om,
        discounted_solar_pv_capex_tco,
        discounted_inverter_capex_tco,
        discounted_stationary_battery_capex_tco,
        discounted_bos_capex_tco,
        discounted_installation_capex_tco,
        solar_pv_om,
        inv_stat_batt_om) = calculateLCOC(scenario, n_draws, costOfCapital, vehicleEnergyConsumption, annualKmTravelled, vehicleLifetime)

        if private_ownership_perspective:
            # Assume investor purchases the SAS outright
            costSASOPEX = solar_pv_om + inv_stat_batt_om    # [USD/year]
            costVehicleEnergy[2::3, :] = costSASOPEX[2::3, :]
        else:
            # Assume 3rd party owner of SAS selling electricity to vehicle investor at LCOC value
            costVehicleLCOC = lcoc * vehicleEnergyConsumption * annualKmTravelled
            # Replace every third row of Matrix1 with the corresponding row from     Matrix2
            costVehicleEnergy[2::3, :] = costVehicleLCOC[2::3, :]


    metric_selection = 'TCO'

    if metric_selection == 'TCO':
        ############################################################################
        ############################################################################
        """
        Calculation for the Total Cost of Ownership (TCO):

                1. computes TCO and intermediate results:

                    TCOVehicleCAPEX = (CostVehicleCAPEX * CRF) / AKT
                    TCOVehicleCAPEXCoC = (CostVehicleCAPEX * CRF) / AKT, where CRF->0
                    TCOSASCAPEX = (∑(n) (SolarPVCAPEX + InverterCAPEX + StatBattCAPEX)/(1+i)^n) / AKT
                    TCOVehicleScrappage = (CostVehicleScrappage * 1/(1+i)^N * CRF) / AKT
                    TCOOM = ((1/N) * (∑(n) CostOM/(1+i)^n) / AKT
                    TCOVehicleEnergy = ((1/N) * (∑(n) CostVehicleEnergy/(1+i)^n) / AKT

        """

        ## Calculate TCO and all intermediate results
        print("Computing TCO Values")

        # ########################################################################
        # ### VEHICLE BATTERY REPLACEMENT ###
        # print("Computing TCOVehicleBatteryCAPEX")
        # # Calculate the total cost of vehicle CAPEX (TCOVehicleCAPEX)
        # vehicle_battery_capex = (costVehicleBatteryCAPEX.copy() * vehicleBatteryCapacity)
        # # Define the size of the storage matrix
        # n_rows = vehicle_battery_capex.shape[0]
        # n_cols = vehicle_battery_capex.shape[1]
        # battery_replacement_matrix = (vehicleLifetime/vehicleLifetime)*10.0
        # cond_matrix = construct_condition_matrix(vehicleLifetime, battery_replacement_matrix, n_rows, n_cols)
        # # Set year zero investment to False as the vehicle CAPEX already includes this cost (i.e. only include replacement battery costs)
        # cond_matrix[:, :, 0] = False
        # discounted_vehicle_battery_capex = matrixBroadcastingFunction(vehicle_battery_capex, costOfCapital, vehicleLifetime, div_bool=False, cond_matrix_bool=[True, cond_matrix])
        # ########################################################################
        discounted_vehicle_battery_capex = 0

        print("Computing TCOVehicleCAPEX Values")
        # Calculate the total cost of vehicle CAPEX (TCOVehicleCAPEX)
        tco_vehicle_capex = ((costVehicleCAPEX.copy() + discounted_vehicle_battery_capex) * capitalRecoveryFactor) / annualKmTravelled

        print("Computing TCOVehicleCAPEXCoC")
        # Calculate the total cost of vehicle CAPEX cost of capital (TCOVehicleCAPEXCoC)
        tco_vehicle_capex_coc = tco_vehicle_capex  - (((costVehicleCAPEX.copy() + discounted_vehicle_battery_capex) * nearZeroCapitalRecoveryFactor) / annualKmTravelled)


        print("Computing TCOSASCAPEX")
        if private_ownership_perspective:
            # Calculate the total cost of stand alone system (SAS) CAPEX (TCOSASCAPEX)
            tco_sas_capex = ((discounted_solar_pv_capex_tco + discounted_inverter_capex_tco + discounted_stationary_battery_capex_tco + discounted_bos_capex_tco + discounted_installation_capex_tco) * capitalRecoveryFactor) / annualKmTravelled
        else:
            tco_sas_capex = tco_vehicle_capex*0.0 / annualKmTravelled

        print("Computing TCOVehicleScrappage")
        # Calculate the total cost of vehicle scrappage value (TCOVehicleScrappage)
        tco_vehicle_scrappage = (-1) * (costVehicleScrappage.copy() * presentValueFactor * capitalRecoveryFactor) / annualKmTravelled

        print("Computing TCOOM")
        # Calculate the total cost of O&M (TCOOM)
        tco_om = costOM.copy()
        tco_om = matrixBroadcastingFunction(tco_om, costOfCapital, vehicleLifetime, div_bool=True, cond_matrix_bool=[False, False])
        tco_om /= annualKmTravelled

        print("Computing TCOVehicleEnergy")
        # Calculate the total cost of vehicle energy (TCOVehicleEnergy)
        tco_vehicle_energy = costVehicleEnergy.copy()
        tco_vehicle_energy = matrixBroadcastingFunction(tco_vehicle_energy, costOfCapital, vehicleLifetime, div_bool=True, cond_matrix_bool=[False, False])
        tco_vehicle_energy /= annualKmTravelled

        print("Computing TCO")
        # Calculate the total cost of ownership
        #tco = tco_vehicle_capex + tco_vehicle_scrappage + tco_om + tco_vehicle_energy
        tco = tco_vehicle_capex + tco_sas_capex + tco_vehicle_scrappage + tco_om + tco_vehicle_energy


        ############################################################################
        print("Recording TCO Values")

        ## Silently record results if argument record is true (default), else return all outputs

        # Prepare output of this calculation (for recording or returning)
        ## Mean
        all_outputs_mean = pd.concat([
                        pd.Series(tco_vehicle_capex.mean(axis=1)),
                        pd.Series(tco_vehicle_capex_coc.mean(axis=1)),
                        pd.Series(tco_sas_capex.mean(axis=1)),
                        pd.Series(tco_vehicle_scrappage.mean(axis=1)),
                        pd.Series(tco_om.mean(axis=1)),
                        pd.Series(tco_vehicle_energy.mean(axis=1)),
                        pd.Series(tco.mean(axis=1)),
                        pd.Series(lcoc_solar_pv_capex.mean(axis=1)),
                        pd.Series(lcoc_inverter_capex.mean(axis=1)),
                        pd.Series(lcoc_stationary_battery_capex.mean(axis=1)),
                        pd.Series(lcoc_bos_capex.mean(axis=1)),
                        pd.Series(lcoc_installation_capex.mean(axis=1)),
                        pd.Series(lcoc_solar_pv_om.mean(axis=1)),
                        pd.Series(lcoc_inv_stat_batt_om.mean(axis=1)),
                        pd.Series(lcoc.mean(axis=1))],
                        axis=1)
        all_outputs_mean.columns = ep.getOutputs()
        ## Standard Deviation
        all_outputs_std = pd.concat([
                        pd.Series(tco_vehicle_capex.std(axis=1)),
                        pd.Series(tco_vehicle_capex_coc.std(axis=1)),
                        pd.Series(tco_sas_capex.std(axis=1)),
                        pd.Series(tco_vehicle_scrappage.std(axis=1)),
                        pd.Series(tco_om.std(axis=1)),
                        pd.Series(tco_vehicle_energy.std(axis=1)),
                        pd.Series(tco.std(axis=1)),
                        pd.Series(lcoc_solar_pv_capex.std(axis=1)),
                        pd.Series(lcoc_inverter_capex.std(axis=1)),
                        pd.Series(lcoc_stationary_battery_capex.std(axis=1)),
                        pd.Series(lcoc_bos_capex.std(axis=1)),
                        pd.Series(lcoc_installation_capex.std(axis=1)),
                        pd.Series(lcoc_solar_pv_om.std(axis=1)),
                        pd.Series(lcoc_inv_stat_batt_om.std(axis=1)),
                        pd.Series(lcoc.std(axis=1))],
                        axis=1)
        all_outputs_std.columns = ep.getOutputs()


        if record:
            # Record all results of this scenario in one file for each type of result
            io.recordOutput(pd.Series(tco_vehicle_capex.mean(axis=1)), scenario, "TCOVehicleCAPEX", "mean")
            io.recordOutput(pd.Series(tco_vehicle_capex_coc.mean(axis=1)), scenario, "TCOVehicleCAPEXCoC", "mean")
            io.recordOutput(pd.Series(tco_sas_capex.mean(axis=1)), scenario, "TCOSASCAPEX", "mean")
            io.recordOutput(pd.Series(tco_vehicle_scrappage.mean(axis=1)), scenario, "TCOVehicleScrappage", "mean")
            io.recordOutput(pd.Series(tco_om.mean(axis=1)), scenario, "TCOOM", "mean")
            io.recordOutput(pd.Series(tco_vehicle_energy.mean(axis=1)), scenario, "TCOVehicleEnergy", "mean")
            io.recordOutput(pd.Series(tco.mean(axis=1)), scenario, "TCO", "mean")
            io.recordOutput(pd.Series(lcoc_solar_pv_capex.mean(axis=1)), scenario, "LCOCSolarPVCAPEX", "mean")
            io.recordOutput(pd.Series(lcoc_inverter_capex.mean(axis=1)), scenario, "LCOCInverterCAPEX", "mean")
            io.recordOutput(pd.Series(lcoc_stationary_battery_capex.mean(axis=1)), scenario, "LCOCStatBatteryCAPEX", "mean")
            io.recordOutput(pd.Series(lcoc_bos_capex.mean(axis=1)), scenario, "LCOCBOSCAPEX", "mean")
            io.recordOutput(pd.Series(lcoc_installation_capex.mean(axis=1)), scenario, "LCOCInstallationCAPEX", "mean")
            io.recordOutput(pd.Series(lcoc_solar_pv_om.mean(axis=1)), scenario, "LCOCSolarPVOM", "mean")
            io.recordOutput(pd.Series(lcoc_inv_stat_batt_om.mean(axis=1)), scenario, "LCOCInvStatBattOM", "mean")
            io.recordOutput(pd.Series(lcoc.mean(axis=1)), scenario, "LCOC", "mean")
            io.recordOutput(pd.Series(tco_vehicle_capex.std(axis=1)), scenario, "TCOVehicleCAPEX", "std")
            io.recordOutput(pd.Series(tco_vehicle_capex_coc.std(axis=1)), scenario, "TCOVehicleCAPEXCoC", "std")
            io.recordOutput(pd.Series(tco_sas_capex.std(axis=1)), scenario, "TCOSASCAPEX", "std")
            io.recordOutput(pd.Series(tco_vehicle_scrappage.std(axis=1)), scenario, "TCOVehicleScrappage", "std")
            io.recordOutput(pd.Series(tco_om.std(axis=1)), scenario, "TCOOM", "std")
            io.recordOutput(pd.Series(tco_vehicle_energy.std(axis=1)), scenario, "TCOVehicleEnergy", "std")
            io.recordOutput(pd.Series(tco.std(axis=1)), scenario, "TCO", "std")
            io.recordOutput(pd.Series(lcoc_solar_pv_capex.std(axis=1)), scenario, "LCOCSolarPVCAPEX", "std")
            io.recordOutput(pd.Series(lcoc_inverter_capex.std(axis=1)), scenario, "LCOCInverterCAPEX", "std")
            io.recordOutput(pd.Series(lcoc_stationary_battery_capex.std(axis=1)), scenario, "LCOCStatBatteryCAPEX", "std")
            io.recordOutput(pd.Series(lcoc_bos_capex.std(axis=1)), scenario, "LCOCBOSCAPEX", "std")
            io.recordOutput(pd.Series(lcoc_installation_capex.std(axis=1)), scenario, "LCOCInstallationCAPEX", "std")
            io.recordOutput(pd.Series(lcoc_solar_pv_om.std(axis=1)), scenario, "LCOCSolarPVOM", "std")
            io.recordOutput(pd.Series(lcoc_inv_stat_batt_om.std(axis=1)), scenario, "LCOCInvStatBattOM", "std")
            io.recordOutput(pd.Series(lcoc.std(axis=1)), scenario, "LCOC", "std")

            # Record all results of this scenario in one file
            io.recordScenario(all_outputs_mean, scenario, "mean")
            io.recordScenario(all_outputs_std, scenario, "std")

        else:
            return all_outputs_mean
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


def calculatePresentValueFactor(costOfCapital_matrix, vehicleLifetime_matrix):
    presentValueFactor = 1/np.power(1+costOfCapital_matrix, vehicleLifetime_matrix)

    return presentValueFactor


def calculateCapitalRecoveryFactor(costOfCapital_matrix, vehicleLifetime_matrix):
    capitalRecoveryFactor = (costOfCapital_matrix * np.power(1+costOfCapital_matrix, vehicleLifetime_matrix)) / (np.power(1+costOfCapital_matrix, vehicleLifetime_matrix) - 1)

    return capitalRecoveryFactor


def construct_condition_matrix(system_lifetime_matrix, component_lifetime_matrix, n_rows, n_cols):
    """
    Constructs a condition matrix that indicates when components need to be replaced
    within the system lifetime, considering element-wise component lifetimes.

    system_lifetime_matrix : np.ndarray
        Matrix of the system lifetime values, shape (n_rows, n_cols).
    component_lifetime_matrix : np.ndarray
        Matrix of the component lifetime values, shape (n_rows, n_cols).
    n_rows, n_cols : int
        Dimensions of the matrices.

    Returns
    -------
    condition_matrix : np.ndarray
        A boolean matrix indicating where components need to be replaced.
    """

    # Determine the maximum system lifetime to define the range of n values
    N_max = int(system_lifetime_matrix.max())

    # Create an array representing each year from 1 to N_max with shape (1, 1, N_max)
    n_values = np.arange(1, N_max + 1).reshape(1, 1, N_max)

    # Broadcast system_lifetime_matrix and component_lifetime_matrix to shape (n_rows, n_cols, N_max)
    sys_life_expanded = np.broadcast_to(system_lifetime_matrix[:, :, np.newaxis], (n_rows, n_cols, N_max))
    comp_life_expanded = np.broadcast_to(component_lifetime_matrix[:, :, np.newaxis], (n_rows, n_cols, N_max))

    # Generate the replacement years condition using vectorized operations
    replacement_years = ((n_values % comp_life_expanded == 0) & (n_values < sys_life_expanded)) | (n_values == 1)

    # Apply the condition by masking out the years beyond the system lifetime
    condition_matrix = np.where(n_values <= sys_life_expanded, replacement_years, False)

    return condition_matrix


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


def pert_samples(min, mode, max, size):
    """
    Generates PERT-distributed samples or returns constant values if min, mode, and max are the same.

    Parameters:
    min (ndarray): Array of minimum values.
    mode (ndarray): Array of mode values.
    max (ndarray): Array of maximum values.
    size (tuple): Tuple specifying the output shape (N, num_samples).

    Returns:
    result (ndarray): A matrix of shape (N, num_samples) containing the generated samples.
    """

    # Case 1: If all values are the same, return an array of that value
    if np.array_equal(min, mode) and np.array_equal(mode, max):
        return np.full(size, mode)

    # Case 2: Normal PERT distribution
    alpha = (4 * mode + max - 5 * min) / (max - min)
    beta = (5 * max - min - 4 * mode) / (max - min)

    return np.random.beta(alpha, beta, size=size) * (max - min) + min


def conductSensitivityAnalysis():

    # Create codes for base parameter set
    base = pd.DataFrame(ep.getScenarios().loc["BaseScenario",:]).T


    # Create codes for parameter sets of all up-sensitivities
    sens_up = pd.DataFrame(data="B", index=ep.getParameters(), columns=ep.getParameters())
    np.fill_diagonal(sens_up.values, "SU")
    # # include a case of all electricity components compounded
    # sens_up = sens_up.append(pd.DataFrame(data="B", index=["CostElectricity"], columns=ep.getParameters()))
    # sens_up.loc["CostElectricity",["CostElectricityE", "CostElectricityN", "CostElectricityT"]] = "SU"

    # Create codes for parameter sets of all down-sensitivities
    sens_down = pd.DataFrame(data="B", index=ep.getParameters(), columns=ep.getParameters())
    np.fill_diagonal(sens_down.values, "SD")
    # # include a case of all electricity components compounded
    # sens_down = sens_down.append(pd.DataFrame(data="B", index=["CostElectricity"], columns=ep.getParameters()))
    # sens_down.loc["CostElectricity",["CostElectricityE", "CostElectricityN", "CostElectricityT"]] = "SD"

    # Deal with the parameter link between SolarPVLifetime and SASSystemLifetime
    sens_down['SolarPVLifetime']['SASSystemLifetime'] = "SD"
    sens_up['SolarPVLifetime']['SASSystemLifetime'] = "SU"
    sens_down = sens_down.drop('SASSystemLifetime', axis=0)
    sens_up = sens_up.drop('SASSystemLifetime', axis=0)

    # Initialize a list of additional sensitivity parameters that are technology specific (i.e. VehicleChargingEnergy - BEV)
    defined_columns_list = ep.getParameters()

    # Calculate base values for each option (but only for TCO, no intermediate outputs)
    print(" Scenario currently being computed: Base Scenario")
    curr_tco = calculateTCO("BaseScenario", record=False)
    # Record base values for each option
    values_base = ep.getEmptyDataFrame('defined_col_list', 'None', ['BaseValues'])
    values_base['BaseValues'] = curr_tco['TCO'].values

    # Initialize data frame to store up-sensitivity values
    values_up = ep.getEmptyDataFrame('defined_col_list', 'None', defined_columns_list)

    # Calculate up-sensitivity values for each option (but only for TCO, no intermediate outputs)
    for row in sens_up.index:
        print(" Scenario currently being computed: Parameter Sensitivities UP - " + row)

        # Add this current parameter set to the scenarios df
        curr_scenario_name = "Sensitivities UP - " + row
        curr_parameter_codes = sens_up.loc[row,:]
        ep.addScenario(curr_scenario_name, curr_parameter_codes)

        curr_tco = calculateTCO(curr_scenario_name, record=False)
        values_up[row] = curr_tco['TCO'].values

    # Initialize data frame to store down-sensitivity values
    values_down = ep.getEmptyDataFrame('defined_col_list', 'None', defined_columns_list)

    # Calculate and aggregate up-sensitivity values for each option (but only for LCOC, no intermediate outputs)
    for row in sens_down.index:
        print(" Scenario currently being computed: Parameter Sensitivities DOWN - " + row)

        # Add this current parameter set to the scenarios df
        curr_scenario_name = "Sensitivities DOWN - " + row
        curr_parameter_codes = sens_down.loc[row,:]
        ep.addScenario(curr_scenario_name, curr_parameter_codes)

        curr_tco = calculateTCO(curr_scenario_name, record=False)
        values_down[row] = curr_tco['TCO'].values

    # Record base, up and down values (e.g. for tornado charts)
     # to pkl
    values_base.to_pickle(ep.OUTPUT_PATH + "SensitivityAnalysisBaseValues" + ".pkl")
    values_up.to_pickle(ep.OUTPUT_PATH + "SensitivityAnalysisUpValues" + ".pkl")
    values_down.to_pickle(ep.OUTPUT_PATH + "SensitivityAnalysisDownValues" + ".pkl")
     # to xlsx
    values_base.to_excel(ep.OUTPUT_PATH + "_SensitivityAnalysisBaseValues" + ".xlsx", sheet_name = "SensitivityAnalysisBaseValues")
    values_up.to_excel(ep.OUTPUT_PATH + "_SensitivityAnalysisUpValues" + ".xlsx", sheet_name = "SensitivityAnalysisUpValues")
    values_down.to_excel(ep.OUTPUT_PATH + "_SensitivityAnalysisDownValues" + ".xlsx", sheet_name = "SensitivityAnalysisDownValues")


def setScenario(scenario):
    global SCENARIO
    SCENARIO = scenario


def getScenario():
    return SCENARIO


def printScenario():
    print(" Scenario currently being computed: " + str(SCENARIO))
