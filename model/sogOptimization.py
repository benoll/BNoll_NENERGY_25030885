# FILE: sogOptimization.py
# PROJECT: Africa Passenger Vehicle TCO Model
# SCRIPT DESCRIPTION: This is the module that optimizes the sizing of the solar
# off-grid charging system

# V11.I IMPORTS

# V11.I.1 Import Packages

import numpy as np
import pandas as pd
import random
import csv
from scipy.optimize import minimize


print("")
print(">>> RunSOGOptimization >>>")
print("")


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
"""


# V11.I.2 Import Input Data Tables

input_Countries = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'Countries')
input_Years = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'Years')
input_VehicleEnergyConsumption = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'VehicleEnergyConsumption')
input_VehicleAnnualKmTravelled = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'VehicleAnnualKmTravelled')
input_CostOfCapital = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'CostOfCapital')
input_CostSolarPVCAPEX = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'CostSolarPVCAPEX')
input_CostInverterCAPEX = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'CostInverterCAPEX')
input_CostStationaryBatteryCAPEX = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'CostStationaryBatteryCAPEX')
input_CostBOSCAPEX = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'CostBOSCAPEX')
input_CostSolarPVOPEX = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'CostSolarPVOPEX')
input_CostInvStatBattOPEX = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'CostInvStatBattOPEX')
input_SolarPVLifetime = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'SolarPVLifetime')
input_InverterLifetime = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'InverterLifetime')
input_StationaryBatteryLifetime = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'StationaryBatteryLifetime')
input_SASSystemLifetime = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'SASSystemLifetime')
input_SASSystemLoss = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'SASSystemLoss')
input_SolarPVOversizeFactor = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'SolarPVOversizeFactor')
input_SASDemandPattern = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'SASDemandPattern')


# NEWLY ADDED BY BESSIE
input_SASReliability = pd.read_excel(INPUT_PATH + "AfricaTCOData.xlsx", sheet_name = 'SASReliability')



# V11.II. BASIC DEFINITIONS AND DECLARATIONS

# V11.II.1 Scenario
"""
NOTE: Optimization is run one year at a time.
"""
year = input_Years.iloc[0, 0] # NOTE: either 2025 = [0, 0], 2030 [1, 0], or 2040 [2, 0]
reliability = 0.9

# V11.II.2 Open CSV
"""
NOTE: Here we import the corrected solar irradiation data. This is not a direct output from getSolarIrradiationData.py,
though that could also be run if desired.

Uncorrected file name: "GeneratedSolarIrradiationData.csv"
"""
with open(INPUT_PATH + "CORRECTED_GeneratedSolarIrradiationData.csv") as csvfile:
    si_geo_data = list(csv.reader(csvfile, delimiter = ","))
nch = 8760 # number of calendar hours


# V11.II.3 Define Results Matrix

lcoc_results = np.zeros((len(si_geo_data), 12), object)

lcoc_results[0, :] = ("Index", "Country (ISO)", "Latitude", "Longitude", "30a LCOC", "CompCaps per kWh Demand",
                      "CompCaps E2W small", "CompCaps E2W large", "CompCaps E4W small",
                      "CompCaps E4W medium", "CompCaps E4W large", "CompCaps e-minibus")


# V11.II.4 Input Variables 2025

if year == 2025:

    system_loss = input_SASSystemLoss.iloc[0, 1]  # System loss of energy (proportion)
    mg_lifetime = int(input_SASSystemLifetime.iloc[0, 1])  # <= 100

    countries = np.zeros(54, dtype=object)
    waccs = np.zeros(54, float)

    for i in range(0, 54):
        countries[i] = input_Countries.iloc[i, 1]
        waccs[i] = input_CostOfCapital.iloc[(i + 108), 3]

    pv_om_cost_perpower = input_CostSolarPVOPEX.iloc[0, 1]
    batinv_om_cost = input_CostInvStatBattOPEX.iloc[0, 1]

    pv_investment_cost = input_CostSolarPVCAPEX.iloc[0, 1]
    inverter_investment_cost = input_CostInverterCAPEX.iloc[0, 1]
    bat_investment_cost = input_CostStationaryBatteryCAPEX.iloc[0, 1]
    other_investment_cost = input_CostBOSCAPEX.iloc[0, 1]

    pv_expected_lifetime = input_SolarPVLifetime.iloc[0, 1]
    inverter_expected_lifetime = input_InverterLifetime.iloc[0, 1]
    bat_expected_lifetime = input_StationaryBatteryLifetime.iloc[0, 1]

    new_e2w_small_energyperkm = input_VehicleEnergyConsumption.iloc[12, 4]
    new_e2w_large_energyperkm = input_VehicleEnergyConsumption.iloc[13, 4]
    new_e4w_small_energyperkm = input_VehicleEnergyConsumption.iloc[14, 4]
    new_e4w_medium_energyperkm = input_VehicleEnergyConsumption.iloc[15, 4]
    new_e4w_large_energyperkm = input_VehicleEnergyConsumption.iloc[16, 4]
    new_eminibus_energyperkm = input_VehicleEnergyConsumption.iloc[17, 4]

    e2w_small_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[0, 2] / 365
    e2w_large_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[1, 2] / 365
    e4w_small_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[2, 2] / 365
    e4w_medium_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[3, 2] / 365
    e4w_large_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[4, 2] / 365
    eminibus_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[5, 2] / 365


# V11.II.5 Input Variables 2030

elif year == 2030:

    system_loss = input_SASSystemLoss.iloc[0, 2]  # System loss of energy (proportion)
    mg_lifetime = int(input_SASSystemLifetime.iloc[0, 2])  # <= 100

    countries = np.zeros(54, dtype=object)
    waccs = np.zeros(54, float)
    for i in range(0, 54):
        countries[i] = input_Countries.iloc[i, 1]
        waccs[i] = input_CostOfCapital.iloc[(i + 108), 4]

    pv_om_cost_perpower = input_CostSolarPVOPEX.iloc[0, 2]
    batinv_om_cost = input_CostInvStatBattOPEX.iloc[0, 2]

    pv_investment_cost = input_CostSolarPVCAPEX.iloc[0, 2]
    inverter_investment_cost = input_CostInverterCAPEX.iloc[0, 2]
    bat_investment_cost = input_CostStationaryBatteryCAPEX.iloc[0, 2]
    other_investment_cost = input_CostBOSCAPEX.iloc[0, 2]

    pv_expected_lifetime = input_SolarPVLifetime.iloc[0, 2]
    inverter_expected_lifetime = input_InverterLifetime.iloc[0, 2]
    bat_expected_lifetime = input_StationaryBatteryLifetime.iloc[0, 2]

    new_e2w_small_energyperkm = input_VehicleEnergyConsumption.iloc[12, 5]
    new_e2w_large_energyperkm = input_VehicleEnergyConsumption.iloc[13, 5]
    new_e4w_small_energyperkm = input_VehicleEnergyConsumption.iloc[14, 5]
    new_e4w_medium_energyperkm = input_VehicleEnergyConsumption.iloc[15, 5]
    new_e4w_large_energyperkm = input_VehicleEnergyConsumption.iloc[16, 5]
    new_eminibus_energyperkm = input_VehicleEnergyConsumption.iloc[17, 5]

    e2w_small_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[0, 3] / 365
    e2w_large_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[1, 3] / 365
    e4w_small_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[2, 3] / 365
    e4w_medium_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[3, 3] / 365
    e4w_large_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[4, 3] / 365
    eminibus_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[5, 3] / 365


# V11.II.6 Input Variables 2040

elif year == 2040:

    system_loss = input_SASSystemLoss.iloc[0, 3]  # System loss of energy (proportion)
    mg_lifetime = int(input_SASSystemLifetime.iloc[0, 3])  # <= 100

    countries = np.zeros(54, dtype=object)
    waccs = np.zeros(54, float)
    for i in range(0, 54):
        countries[i] = input_Countries.iloc[i, 1]
        waccs[i] = input_CostOfCapital.iloc[(i + 108), 5]

    pv_om_cost_perpower = input_CostSolarPVOPEX.iloc[0, 3]
    batinv_om_cost = input_CostInvStatBattOPEX.iloc[0, 3]

    pv_investment_cost = input_CostSolarPVCAPEX.iloc[0, 3]
    inverter_investment_cost = input_CostInverterCAPEX.iloc[0, 3]
    bat_investment_cost = input_CostStationaryBatteryCAPEX.iloc[0, 3]
    other_investment_cost = input_CostBOSCAPEX.iloc[0, 3]

    pv_expected_lifetime = input_SolarPVLifetime.iloc[0, 3]
    inverter_expected_lifetime = input_InverterLifetime.iloc[0, 3]
    bat_expected_lifetime = input_StationaryBatteryLifetime.iloc[0, 3]

    new_e2w_small_energyperkm = input_VehicleEnergyConsumption.iloc[12, 6]
    new_e2w_large_energyperkm = input_VehicleEnergyConsumption.iloc[13, 6]
    new_e4w_small_energyperkm = input_VehicleEnergyConsumption.iloc[14, 6]
    new_e4w_medium_energyperkm = input_VehicleEnergyConsumption.iloc[15, 6]
    new_e4w_large_energyperkm = input_VehicleEnergyConsumption.iloc[16, 6]
    new_eminibus_energyperkm = input_VehicleEnergyConsumption.iloc[17, 6]

    e2w_small_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[0, 4] / 365
    e2w_large_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[1, 4] / 365
    e4w_small_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[2, 4] / 365
    e4w_medium_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[3, 4] / 365
    e4w_large_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[4, 4] / 365
    eminibus_daily_driven_distance = input_VehicleAnnualKmTravelled.iloc[5, 4] / 365


# V11.II.7 Define Demand-Arrays (incl. Time Zones)

energy_demand_UTC = np.zeros(24, float)  # = 0 (Universal Time Coordinated)
energy_demand_CVT = np.zeros(24, float)  # - 1 (Cabo Verde Time)
energy_demand_WAT = np.zeros(24, float)  # + 1 (Western African Time)
energy_demand_CAT = np.zeros(24, float)  # + 2 (Central African Time)
energy_demand_EAT = np.zeros(24, float)  # + 3 (Eastern African Time)
energy_demand_MUT = np.zeros(24, float)  # + 4 (Mauritius Time)
for i in range(0, 24):
    energy_demand_UTC[i] = input_SASDemandPattern.iloc[i, 1]
    energy_demand_CVT[i] = input_SASDemandPattern.iloc[i + 1, 1]  # - 1
    energy_demand_WAT[i] = input_SASDemandPattern.iloc[i + 23, 1]  # + 1
    energy_demand_CAT[i] = input_SASDemandPattern.iloc[i + 22, 1]  # + 2
    energy_demand_EAT[i] = input_SASDemandPattern.iloc[i + 21, 1]  # + 3
    energy_demand_MUT[i] = input_SASDemandPattern.iloc[i + 20, 1]  # + 4




# V11.III FUNCTIONS


# V11.III.1 Reliability Calculation Function

def reliability_calc(core_spv_cap, core_inv_cap, core_bat_cap, pv_output_raw_hourly, energy_demand_xhour): # Inputs

    # This function is later used again in the function for the constraint in the optimization. It calculates the
    # reliability based on the solar irradiation and the potential component capacities. The component capacities are
    # optimized.

    # PV Outputs
    pv_output_hourly = np.zeros(nch, float)
    for i in range(0, nch):
        pv_output_hourly[i] = pv_output_raw_hourly[i]
        if pv_output_hourly[i] * (1 - system_loss) * core_spv_cap <= core_inv_cap:
            pv_output_hourly[i] = pv_output_hourly[i] * (1 - system_loss) * core_spv_cap
        else:
            pv_output_hourly[i] = core_inv_cap

    pv_only_net_energy_hourly = pv_output_hourly - energy_demand_xhour

    # PV + Battery
    bat_energy = np.zeros(nch, float)  # Current battery energy balance, values = [0:core_bat_cap]
    bat_energy[0] = core_bat_cap  # Battery is fully charged at the start of the year
    bat_demand = np.zeros(nch, float)  # Unmet demand by pv -> could be provided by bat
    bat_charge = np.zeros(nch, float)  # Undemanded pv energy -> charge bat
    bat_discharge = np.zeros(nch, float)  # Demanded energy provided by bat
    pv_bat_net_energy_hourly = np.zeros(nch, float)  # Energy balance with bat and pv
    for i in range(0, nch):
        if pv_only_net_energy_hourly[i] < 0:
            bat_demand[i] = pv_only_net_energy_hourly[i] * (-1)
        elif pv_only_net_energy_hourly[i] > 0 and pv_only_net_energy_hourly[i] <= core_inv_cap:
            bat_charge[i] = pv_only_net_energy_hourly[i]
        elif pv_only_net_energy_hourly[i] > 0 and pv_only_net_energy_hourly[i] > core_inv_cap:
            bat_charge[i] = core_inv_cap
    for i in range(1, nch):
        if bat_energy[i - 1] + bat_charge[i - 1] - bat_demand[i - 1] < 0 and bat_demand[i - 1] <= core_inv_cap:
            bat_energy[i] = 0
        elif bat_energy[i - 1] + bat_charge[i - 1] - bat_demand[i - 1] < 0 and bat_demand[i - 1] > core_inv_cap:
            bat_energy[i] = bat_energy[i - 1] + bat_charge[i - 1] - core_inv_cap
            if bat_energy[i] < 0:
                bat_energy[i] = 0
        elif bat_energy[i - 1] + bat_charge[i - 1] - bat_demand[i - 1] > core_bat_cap and bat_demand[
            i - 1] <= core_inv_cap:
            bat_energy[i] = core_bat_cap
        elif bat_energy[i - 1] + bat_charge[i - 1] - bat_demand[i - 1] > core_bat_cap and bat_demand[
            i - 1] > core_inv_cap:
            bat_energy[i] = bat_energy[i - 1] + bat_charge[i - 1] - core_inv_cap
            if bat_energy[i] > core_bat_cap:
                bat_energy[i] = core_bat_cap
        elif bat_demand[i - 1] > core_inv_cap:
            bat_energy[i] = bat_energy[i - 1] + bat_charge[i - 1] - core_inv_cap
            if bat_energy[i] < 0:
                bat_energy[i] = 0
            if bat_energy[i] > core_bat_cap:
                bat_energy[i] = core_bat_cap
        else:
            bat_energy[i] = bat_energy[i - 1] + bat_charge[i - 1] - bat_demand[i - 1]
    for i in range(0, nch):
        if bat_demand[i] <= bat_energy[i] and bat_demand[i] <= core_inv_cap:
            bat_discharge[i] = bat_demand[i]
        elif bat_demand[i] <= bat_energy[i] and bat_demand[i] > core_inv_cap:
            bat_discharge[i] = core_inv_cap
        elif bat_demand[i] > bat_energy[i] and bat_demand[i] <= core_inv_cap:
            bat_discharge[i] = bat_energy[i]
        elif bat_demand[i] > bat_energy[i] and bat_demand[i] > core_inv_cap:
            if bat_energy[i] <= core_inv_cap:
                bat_discharge[i] = bat_energy[i]
            else:
                bat_discharge[i] = core_inv_cap
    pv_bat_net_energy_hourly = pv_output_hourly + bat_discharge - energy_demand_xhour

    # Unmet Demand
    unmet_demand_hourly = np.zeros(nch, float)
    for i in range(0, nch):
        if pv_bat_net_energy_hourly[i] <= 0:
            unmet_demand_hourly[i] = pv_bat_net_energy_hourly[i] * (-1)
    # Return
    return sum(unmet_demand_hourly)


# V11.III.2 Optimization Funtion

def optimization(pv_output_raw_hourly, reliability, init, energy_demand_xhour): # Inputs

    # Objective Function (LCOC Calculation)
    def objective(x):
        opti_spv_cap = x[0]
        opti_inv_cap = x[1]
        opti_bat_cap = x[2]
        uac_spv = np.zeros(mg_lifetime + 1, float)
        uac_inv = np.zeros(mg_lifetime + 1, float)
        uac_bat = np.zeros(mg_lifetime + 1, float)

        uac_spv[0] = opti_spv_cap * (pv_investment_cost + other_investment_cost)
        uac_inv[0] = opti_inv_cap * inverter_investment_cost
        uac_bat[0] = opti_bat_cap * bat_investment_cost

        for i in range(1, mg_lifetime + 1):
            if i % pv_expected_lifetime == 0:
                uac_spv[i] = ((opti_spv_cap * sum(pv_output_raw_hourly) * pv_om_cost_perpower) +
                            (opti_spv_cap * (pv_investment_cost + other_investment_cost))) / ((1 + wacc) ** i)
            else:
                uac_spv[i] = (opti_spv_cap * sum(pv_output_raw_hourly) * pv_om_cost_perpower) / ((1 + wacc) ** i)
            if i % inverter_expected_lifetime == 0:
                uac_inv[i] = (opti_inv_cap * inverter_investment_cost) / ((1 + wacc) ** i)
            if i % bat_expected_lifetime == 0:
                uac_bat[i] = (((opti_bat_cap * bat_investment_cost) + (opti_bat_cap * batinv_om_cost)) /
                            ((1 + wacc) ** i))
            else:
                uac_bat[i] = (opti_bat_cap * batinv_om_cost) / ((1 + wacc) ** i)
        return sum(uac_spv) + sum(uac_inv) + sum(uac_bat)


    # Constraint Function
    def constraint1(x): # Reliability Calculation
        unmet_demand = reliability_calc(x[0], x[1], x[2], pv_output_raw_hourly, energy_demand_xhour)
        return (sum(energy_demand_xhour) * (1 - reliability)) - unmet_demand


    # Minimization
    b = (0, None)
    bnds = (b, b, b)
    con1 = {'type': 'ineq', 'fun': constraint1}
    cons = [con1]
    result = minimize(objective, x0 = init, method = 'COBYLA', bounds = bnds, constraints = cons,
                      options = {'maxiter': 1, 'tol': 1}) #1e-10 #maxiter: 10000

    # Total provided Energy
    total_provided_energy = np.zeros(mg_lifetime + 1, float)
    for i in range(1, mg_lifetime + 1):
        total_provided_energy[i] = (sum(energy_demand_xhour) * reliability) / ((1 + wacc) ** i)

    # Return
    return (result.fun / sum(total_provided_energy)), result.x, result.success


# V11.III.3 LCOC Calculation Function

def mg_lcoc_vehicle(energy_demand): # Inputs

    # Recall Optimization Results
    init_x_values = [sum(energy_demand[0:24]) / 3, sum(energy_demand[0:24]) / 6, sum(energy_demand[0:24]) / 2] # Highly sensitive
    mg_opti_results[0] = reliability
    objective_value, x_values, success = optimization(pv_output_raw_hourly, reliability, init_x_values, energy_demand)
    while success == False or x_values[1] > x_values[0]:
        recalc = random.uniform((-1), 1)
        objective_value, x_values, success = optimization(pv_output_raw_hourly, mg_opti_results[0],
                                                          ([init_x_values[0] + recalc, init_x_values[1] - recalc,
                                                            init_x_values[2] + recalc]), energy_demand)
        objective_value = objective_value
        x_values = x_values

    # Return
    return objective_value, x_values




# V11.IV EXECUTION

print()
print(">>> OPTIMIZATION AND LCOC CALCULATION >>>")
print()

for dp in range(1, len(si_geo_data)): # dp = data point

    print("Starting loop for high-res point x")

    # V11.IV.1 Recall Country-Specific Geographical Data

    for i in range(0, 54):
        if si_geo_data[dp][1] == countries[i]:
            wacc = waccs[i]
            country = countries[i]

    si_array = si_geo_data[dp][4].split()
    si_array = [float(value) for value in si_array]
    pv_output_raw_hourly = np.zeros(nch, float)  # PV output in kW/kWₚ, w/out system loss
    for i in range(0, nch):
        pv_output_raw_hourly[i] = si_array[i] / 1000 # SI array data was in W but the new array is in kW


    # V11.IV.2 Use right Demand-Array depending on Time Zone

    if (country == "CV"):
        energy_demand_TZ = energy_demand_CVT
    elif (country == "MR" or "ML" or "SN" or "GM" or "GW" or "GN" or "SL" or "LR" or "CI" or "GH" or "TG" or "BF" or
            "ST"):
        energy_demand_TZ = energy_demand_UTC
    elif (country == "MA" or "DZ" or "TN" or "NE" or "TD" or "BJ" or "NG" or "CM" or "CF" or "GQ" or "GA" or "CG" or
            "CD" or "AO"):
        energy_demand_TZ = energy_demand_WAT
    elif (country == "LY" or "EG" or "SD" or "SS" or "RW" or "BI" or "ZM" or "MW" or "MZ" or "NB" or "BW" or "ZW" or
            "ZA" or "LS" or "SZ"):
        energy_demand_TZ = energy_demand_CAT
    elif (country == "ER" or "ET" or "DJ" or "SO" or "UG" or "KE" or "TZ" or "KM" or "MG"):
        energy_demand_TZ = energy_demand_EAT
    elif (country == "SC" or "MU"):
        energy_demand_TZ = energy_demand_MUT


    # V11.IV.3 Calculate complete Demand-Array

    energy_demand_TZ_xhour = np.zeros(nch, float)
    for i in range(0, 24):
        for j in range(0, 365):
            energy_demand_TZ_xhour[i + j * 24] = energy_demand_TZ[i]


    # V11.IV.4 Recall Results Recall LCOCs for different vehicle types

    print(si_geo_data[dp][0], country, si_geo_data[dp][3], si_geo_data[dp][2], "  ", round((dp/2558) * 100, 2), "%")

    mg_opti_results = np.zeros(5)

    mg_lcoc_perkWhdemand, compcaps_perkWhdemand = mg_lcoc_vehicle(energy_demand_TZ_xhour)

    compcaps_new_e2w_small = compcaps_perkWhdemand * e2w_small_daily_driven_distance * new_e2w_small_energyperkm
    compcaps_new_e2w_large = compcaps_perkWhdemand * e2w_large_daily_driven_distance * new_e2w_large_energyperkm
    compcaps_new_e4w_small = compcaps_perkWhdemand * e4w_small_daily_driven_distance * new_e4w_small_energyperkm
    compcaps_new_e4w_medium = compcaps_perkWhdemand * e4w_medium_daily_driven_distance * new_e4w_medium_energyperkm
    compcaps_new_e4w_large = compcaps_perkWhdemand * e4w_large_daily_driven_distance * new_e4w_large_energyperkm
    compcaps_new_eminibus = compcaps_perkWhdemand * eminibus_daily_driven_distance * new_eminibus_energyperkm


    # V11.VI.5 Save Results in Matrix

    lcoc_results[(dp), :] = [dp, country, si_geo_data[dp][2], si_geo_data[dp][3],
                             mg_lcoc_perkWhdemand, np.array(compcaps_perkWhdemand),
                             np.array(compcaps_new_e2w_small), np.array(compcaps_new_e2w_large),
                             np.array(compcaps_new_e4w_small), np.array(compcaps_new_e4w_medium),
                             np.array(compcaps_new_e4w_large), np.array(compcaps_new_eminibus)]



# V11.V SAVE RESULTS IN CSV AND PRINT THEM IN CONSOLE

df = pd.DataFrame(lcoc_results[1:], columns = lcoc_results[0])
pd.set_option('display.max_columns', None)
np.set_printoptions(formatter={'float': '{:.4f}'.format})
print(df)
df = df.applymap(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
csv_string = df.to_csv(index=False)
df.to_csv(INPUT_PATH + "calculatedLCOCs_2040_rel90_TRY.csv", index=False)

# print(" ")
# print("Final Results")
# print()
# print()
# print("CSV:")
# print()
# print(csv_string)
