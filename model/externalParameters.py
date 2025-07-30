# FILE: externalParameters.py
# PROJECT: Africa Passenger Vehicle TCO Model
# SCRIPT DESCRIPTION: This is the parameter storage module where all input parameters as well as other model
# constants are read and stored.


# Import statements
import pandas as pd
import numpy as np
import ast
import math


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
Change path to other data folder here:
"""
global OTHER_PATH
OTHER_PATH = r'/Users/benoll/Africa TCO Model_vSub/data/_other/'

"""
******************************************************************************
"""


##############################################################
# EXCEL FILE IMPORTS:
##############################################################

# Read file:
xls_Africa_TCO_model = pd.ExcelFile(INPUT_PATH + "AfricaTCOData.xlsx")
xls_SAS_Optimization_results = pd.ExcelFile(INPUT_PATH + "SOGOptimizationData.xlsx")

# Read variable lists from file:
years_df = pd.read_excel(xls_Africa_TCO_model, 'Years')
countries_df = pd.read_excel(xls_Africa_TCO_model, 'Countries', keep_default_na=False)
sale_status_df = pd.read_excel(xls_Africa_TCO_model, 'SaleStatus')
applications_df = pd.read_excel(xls_Africa_TCO_model, 'Applications')
technologies_df = pd.read_excel(xls_Africa_TCO_model, 'Technologies')
parameters_df = pd.read_excel(xls_Africa_TCO_model, 'Parameters', index_col=[0])
input_param_dependencies_df = pd.read_excel(xls_Africa_TCO_model, 'InputParamDependencies', index_col=[0])
outputs_df = pd.read_excel(xls_Africa_TCO_model, 'Outputs')
scenario_codes_df = pd.read_excel(xls_Africa_TCO_model, 'ScenarioCodes', index_col=[0])

# Read list of scenarios from file:
scenarios_df = pd.read_excel(xls_Africa_TCO_model, '_Scenarios', index_col=[0])

# Read TCO parameter data from file
CostVehicleCAPEX_data = pd.read_excel(xls_Africa_TCO_model, 'CostVehicleCAPEX', index_col=[0,1,2,3])
CostVehicleScrappage_data = pd.read_excel(xls_Africa_TCO_model, 'CostVehicleScrappage', index_col=[0,1])
CostVehicleOM_data = pd.read_excel(xls_Africa_TCO_model, 'CostVehicleOM', index_col=[0,1,2,3])
CostGasoline_data = pd.read_excel(xls_Africa_TCO_model, 'CostGasoline', index_col=[0,1])
CostSyntheticFuel_data = pd.read_excel(xls_Africa_TCO_model, 'CostSyntheticFuel', index_col=[0,1])
VehicleEnergyConsumption_data = pd.read_excel(xls_Africa_TCO_model, 'VehicleEnergyConsumption', index_col=[0,1,2,3])
VehicleLifetime_data = pd.read_excel(xls_Africa_TCO_model, 'VehicleLifetime', index_col=[0,1,2,3])
VehicleAnnualKmTravelled_data = pd.read_excel(xls_Africa_TCO_model, 'VehicleAnnualKmTravelled', index_col=[0,1])
CostChargingEnergy_data = pd.read_excel(xls_Africa_TCO_model, 'CostChargingEnergy', index_col=[0,1,2,3])
CostOfCapital_data = pd.read_excel(xls_Africa_TCO_model, 'CostOfCapital', index_col=[0,1,2])

# Read LCOC parameter data from file
CostSolarPVCAPEX_data = pd.read_excel(xls_Africa_TCO_model, 'CostSolarPVCAPEX', index_col=[0])
CostInverterCAPEX_data = pd.read_excel(xls_Africa_TCO_model, 'CostInverterCAPEX', index_col=[0])
CostStationaryBatteryCAPEX_data = pd.read_excel(xls_Africa_TCO_model, 'CostStationaryBatteryCAPEX', index_col=[0])
CostBOSCAPEX_data = pd.read_excel(xls_Africa_TCO_model, 'CostBOSCAPEX', index_col=[0])
CostInstallationCAPEX_data = pd.read_excel(xls_Africa_TCO_model, 'CostInstallationCAPEX', index_col=[0])
CostSolarPVOPEX_data = pd.read_excel(xls_Africa_TCO_model, 'CostSolarPVOPEX', index_col=[0])
CostInvStatBattOPEX_data = pd.read_excel(xls_Africa_TCO_model, 'CostInvStatBattOPEX', index_col=[0])
SolarPVLifetime_data = pd.read_excel(xls_Africa_TCO_model, 'SolarPVLifetime', index_col=[0])
InverterLifetime_data = pd.read_excel(xls_Africa_TCO_model, 'InverterLifetime', index_col=[0])
StationaryBatteryLifetime_data = pd.read_excel(xls_Africa_TCO_model, 'StationaryBatteryLifetime', index_col=[0])
SASSystemLifetime_data = pd.read_excel(xls_Africa_TCO_model, 'SASSystemLifetime', index_col=[0])
SolarPVOversizeFactor_data = pd.read_excel(xls_Africa_TCO_model, 'SolarPVOversizeFactor', index_col=[0])
InverterOversizeFactor_data = pd.read_excel(xls_Africa_TCO_model, 'InverterOversizeFactor', index_col=[0])
StatBatteryOversizeFactor_data = pd.read_excel(xls_Africa_TCO_model, 'StatBatteryOversizeFactor', index_col=[0])

# Read LCA parameter data from file
BEVManufacturingEmissions_data = pd.read_excel(xls_Africa_TCO_model, 'BEVManufacturingEmissions', index_col=[0,1])
ICEManufacturingEmissions_data = pd.read_excel(xls_Africa_TCO_model, 'ICEManufacturingEmissions', index_col=[0,1])
FosFuelProdEmissions_data = pd.read_excel(xls_Africa_TCO_model, 'FosFuelProdEmissions', index_col=[0,1])
SynFuelProdEmissions_data = pd.read_excel(xls_Africa_TCO_model, 'SynFuelProdEmissions', index_col=[0,1])
ICECombustionEmissions_data = pd.read_excel(xls_Africa_TCO_model, 'ICECombustionEmissions', index_col=[0,1])
SASManufacturingEmissions_data = pd.read_excel(xls_Africa_TCO_model, 'SASManufacturingEmissions', index_col=[0,1])
BatteryEnergyDensity_data = pd.read_excel(xls_Africa_TCO_model, 'BatteryEnergyDensity', index_col=[0,1])

# Read the SAS Optimization data from the file
SASOptimizationResults2025_data = pd.read_excel(xls_SAS_Optimization_results, '2025')
SASOptimizationResults2030_data = pd.read_excel(xls_SAS_Optimization_results, '2030')
SASOptimizationResults2040_data = pd.read_excel(xls_SAS_Optimization_results, '2040')



################################################################################
def initialize(organize_data_bool, record_data_bool):
    """
    This function takes in the input Excel file and initializes all constants, parameters, and distributions
    """

    if organize_data_bool:
        # Organize the input data for proper code run
        organizeDataFrames(record_data_bool)
        # Organize SAS component capacities from external optimization
        organizeSASOptimizationResults(record_data_bool)
    else:
        # Load pre-organized data as pkl files
        loadDataFrames()

################################################################################

def organizeDataFrames(record_data_bool):
    ## Initialize DataFrame variable
    print("Reorganizing input data...")

    for parameter in getParameters():
        # Skip over the SAS capacity parameters as these are dealt with separately in a next step
        if parameter in ['SolarPVCapacity', 'InverterCapacity','StationaryBatteryCapacity']:
            continue

        print(parameter)

        if parameter == 'CostVehicleEnergy':
            # NOTE: there is no one to one dataframe match for the CostVehicleEnergy parameter
            # Get the number of unique statistical values for this parameter
            stat_values_array = CostGasoline_data.index.get_level_values('Statistical Parameter').unique().values

            # Loop through each statistical value and create a dataframe
            for stat_value in stat_values_array:
                print(stat_value)
                # Initialize a new empty organized dataframe
                curr_param_organized_df = getEmptyDataFrame('scenario_codes', getParameterNumScenarios(parameter))

                # Loop through this organized dataframe and assign appropriate values from the original dataframe
                for index, row in curr_param_organized_df.iterrows():
                    for col in curr_param_organized_df.columns:
                        year = index[0]
                        country = index[1]
                        sale_status = index[2]
                        application = index[3]
                        technology = index[4]
                        statistical_parameter = stat_value
                        scenario_code = col

                        if technology == 'ICE-G':
                            vehicle_energy_param = 'CostGasoline'
                            curr_param_original_df = CostGasoline_data
                        elif technology == 'ICE-SF':
                            vehicle_energy_param = 'CostSyntheticFuel'
                            curr_param_original_df = CostSyntheticFuel_data
                        elif technology == 'BEV':
                            vehicle_energy_param = 'CostChargingEnergy'
                            curr_param_original_df = CostChargingEnergy_data


                        ## Initialize the dimensional dependencies for this parameter
                        dimensional_dependence_array = input_param_dependencies_df.loc[vehicle_energy_param,:].values
                        # Take out nan Values
                        dimensional_dependence_array = dimensional_dependence_array[dimensional_dependence_array != '(None)']
                        # Replace spaces with underscores
                        dimensional_dependence_array = np.char.replace(np.array([dimensional_dependence_array], dtype=str), ' ', '_')
                        # Make all characters lowercase
                        dimensional_dependence_array = np.char.lower(dimensional_dependence_array)


                        ## Create a new array with the corresponding dimensional dependency values
                        original_data_index = [0] * dimensional_dependence_array[0].size
                        # Loop over each string in dimensional_dependence_array
                        for i, value in enumerate(dimensional_dependence_array[0]):
                            original_data_index[i] = eval(value)
                        # Convert list to a tuple for proper indexing
                        original_data_index = tuple(original_data_index)
                        original_data_col = scenario_code + "-" + str(year)

                        # Assign the value
                        curr_param_organized_df.at[index,col] = curr_param_original_df.loc[original_data_index, original_data_col]


                # Add this newly created dataframe to the list of global variables
                curr_param_organized_df_name = parameter + "_DF_" + stat_value
                globals()[curr_param_organized_df_name] = curr_param_organized_df

                if record_data_bool:
                    # Save dataframe as a pkl file
                    curr_param_organized_df.to_pickle(OUTPUT_PATH + "/Input Data PKL Files/" + curr_param_organized_df_name + "_" + str(getMonteCarloDraws()) + ".pkl")
                    curr_param_organized_df.to_excel(OUTPUT_PATH + "/Input Data PKL Files/" + curr_param_organized_df_name + "_" + str(getMonteCarloDraws()) + ".xlsx")

        else:
            # Get the original parameter data from the excel readin
            curr_param_original_df = eval(parameter + "_data")
            # Get the number of unique statistical values for this parameter
            stat_values_array = curr_param_original_df.index.get_level_values('Statistical Parameter').unique().values
            # Loop through each statistical value and create a dataframe
            for stat_value in stat_values_array:
                print(stat_value)
                # Initialize a new empty organized dataframe
                curr_param_organized_df = getEmptyDataFrame('scenario_codes', getParameterNumScenarios(parameter))


                # Loop through this organized dataframe and assign appropriate values from the original dataframe
                for index, row in curr_param_organized_df.iterrows():
                    for col in curr_param_organized_df.columns:
                        year = index[0]
                        country = index[1]
                        sale_status = index[2]
                        application = index[3]
                        technology = index[4]
                        statistical_parameter = stat_value
                        scenario_code = col


                        ## Initialize the dimensional dependencies for this parameter
                        dimensional_dependence_array = input_param_dependencies_df.loc[parameter,:].values
                        # Take out nan Values
                        dimensional_dependence_array = dimensional_dependence_array[dimensional_dependence_array != '(None)']
                        # Replace spaces with underscores
                        dimensional_dependence_array = np.char.replace(np.array([dimensional_dependence_array], dtype=str), ' ', '_')
                        # Make all characters lowercase
                        dimensional_dependence_array = np.char.lower(dimensional_dependence_array)


                        ## Create a new array with the corresponding dimensional dependency values
                        original_data_index = [0] * dimensional_dependence_array[0].size
                        # Loop over each string in dimensional_dependence_array
                        for i, value in enumerate(dimensional_dependence_array[0]):
                            original_data_index[i] = eval(value)
                        # Convert list to a tuple for proper indexing
                        original_data_index = tuple(original_data_index)
                        original_data_col = scenario_code + "-" + str(year)

                        # if parameter == 'CostVehicleScrappage':
                        #     print(curr_param_organized_df)
                        #     print(index)
                        #     print(col)
                        #     print(curr_param_original_df)
                        #     print(original_data_index)
                        #     print(original_data_col)
                        # #print(stopp)

                        # Assign the value
                        curr_param_organized_df.at[index,col] = curr_param_original_df.loc[original_data_index, original_data_col]


                # Add this newly created dataframe to the list of global variables
                curr_param_organized_df_name = parameter + "_DF_" + stat_value
                globals()[curr_param_organized_df_name] = curr_param_organized_df

                if record_data_bool:
                    # Save dataframe as a pkl file
                    curr_param_organized_df.to_pickle(OUTPUT_PATH + "/Input Data PKL Files/" + curr_param_organized_df_name + "_" + str(getMonteCarloDraws()) + ".pkl")
                    curr_param_organized_df.to_excel(OUTPUT_PATH + "/Input Data PKL Files/" + curr_param_organized_df_name + "_" + str(getMonteCarloDraws()) + ".xlsx")


def average_lists(list_of_lists):
    # Function to average corresponding elements in lists
    return [np.mean(elements) for elements in zip(*list_of_lists)]


def parse_and_convert(list_str):
    try:
        # Attempt to parse the string using ast.literal_eval
        parsed_list = ast.literal_eval(list_str)
        # Check if the result is a list and contains numerical values
        if isinstance(parsed_list, list):
            return [round(float(element), 4) for element in parsed_list]
        else:
            print(f"Unexpected format: {list_str}")
            return []
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing {list_str}: {e}")
        return []


def organizeSASOptimizationResults(record_data_bool):
    ## Initialize DataFrame variable
    print("Organizing pre-run SAS optimization data...")

    # Combine SAS Optimization results for all years
    SASOptimizationResults_data = pd.DataFrame([])
    for i, year in enumerate(getYears()):
        if i == 0:
            SASOptimizationResults_data = eval("SASOptimizationResults" + str(year) + "_data")
        else:
            df1 = SASOptimizationResults_data
            df2 = eval("SASOptimizationResults" + str(year) + "_data")
            SASOptimizationResults_data = pd.concat([df1, df2], ignore_index=True)

    SASOptimizationResults_data.set_index(['Year', 'Country (ISO)'], inplace=True)

    # Clean the dataframe
    for column in SASOptimizationResults_data.columns:
        if 'CompCaps' in column:
            SASOptimizationResults_data[column] = SASOptimizationResults_data[column].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

    # Loop over each parameter and reorganize into a new dataframe
    for parameter in ['SolarPVCapacity', 'InverterCapacity','StationaryBatteryCapacity']:
        print(parameter)
        # Get the original parameter data from the excel readin
        curr_param_original_df = SASOptimizationResults_data
        # Get the number of unique statistical values for this parameter
        stat_values_array = ['mode']
        # Loop through each statistical value and create a dataframe
        for stat_value in stat_values_array:
            print(stat_value)
            # Initialize a new empty organized dataframe
            curr_param_organized_df = getEmptyDataFrame('scenario_codes', getParameterNumScenarios(parameter))


            # Loop through this organized dataframe and assign appropriate values from the original dataframe (NOTE: only do this for the 'BaseValue' column; set SensitivityUp/Down later)
            for index, row in curr_param_organized_df.iterrows():
                col = 'BaseValue'
                year = index[0]
                country = index[1]
                sale_status = index[2]
                application = index[3]
                technology = index[4]
                statistical_parameter = stat_value
                scenario_code = col

                # If the technology is not a BEV, do not compute
                if technology == 'BEV':

                    # Get the country ISO code
                    iso_code = getCountryISOCode(country)

                    # # Make an exception for Algeria (DZ)
                    # if iso_code == 'DZ':
                    #     iso_code = 'MA' # Morocco
                    # Make an exception for Namibia (DZ)
                    if iso_code == 'NA':
                        iso_code = 'NB' # Renamed code

                    # Initialize the matching column name
                    column_name = f'CompCaps {sale_status} {application}'

                    # Compute the average and select the current year and country
                    average_results = SASOptimizationResults_data.groupby(['Year','Country (ISO)']).agg({column_name: lambda x: average_lists(x)}).loc[(year,iso_code), :]

                    # Select the parameter specific average value and assign it to the organized dataframe
                    if parameter == 'SolarPVCapacity':
                        # Assign the value
                        curr_param_organized_df.at[index,col] = average_results[0][0]
                    elif parameter == 'InverterCapacity':
                        # Assign the value
                        curr_param_organized_df.at[index,col] = average_results[0][1]
                    elif parameter == 'StationaryBatteryCapacity':
                        # Assign the value
                        curr_param_organized_df.at[index,col] = average_results[0][2]

            # Adjust the SensitivityUp and SensitivityDown column
            curr_param_organized_df['SensitivityUp'] = curr_param_organized_df['BaseValue']*1.2
            curr_param_organized_df['SensitivityDown'] = curr_param_organized_df['BaseValue']*0.8

            # Add this newly created dataframe to the list of global variables
            curr_param_organized_df_name = parameter + "_DF_" + stat_value
            globals()[curr_param_organized_df_name] = curr_param_organized_df


            # Then do the same for the min and max as +/- 10% of the mode
            ## MAX ##
            print('max')
            curr_param_organized_df_MAX = curr_param_organized_df*1.1
            # Add this newly created dataframe to the list of global variables
            curr_param_organized_df_name_MAX = parameter + "_DF_max"
            globals()[curr_param_organized_df_name_MAX] = curr_param_organized_df_MAX


            ## MIN ##
            print('min')
            curr_param_organized_df_MIN = curr_param_organized_df*0.9
            # Add this newly created dataframe to the list of global variables
            curr_param_organized_df_name_MIN = parameter + "_DF_min"
            globals()[curr_param_organized_df_name_MIN] = curr_param_organized_df_MIN

            if record_data_bool:
                # Save dataframe as a pkl file
                curr_param_organized_df.to_pickle(OUTPUT_PATH + "/Input Data PKL Files/" + curr_param_organized_df_name + "_" + str(getMonteCarloDraws()) + ".pkl")
                curr_param_organized_df.to_excel(OUTPUT_PATH + "/Input Data PKL Files/" + curr_param_organized_df_name + "_" + str(getMonteCarloDraws()) + ".xlsx")
                # Save dataframe as a pkl file
                curr_param_organized_df_MAX.to_pickle(OUTPUT_PATH + "/Input Data PKL Files/" + curr_param_organized_df_name_MAX + "_" + str(getMonteCarloDraws()) + ".pkl")
                curr_param_organized_df_MAX.to_excel(OUTPUT_PATH + "/Input Data PKL Files/" + curr_param_organized_df_name_MAX + "_" + str(getMonteCarloDraws()) + ".xlsx")
                # Save dataframe as a pkl file
                curr_param_organized_df_MIN.to_pickle(OUTPUT_PATH + "/Input Data PKL Files/" + curr_param_organized_df_name_MIN + "_" + str(getMonteCarloDraws()) + ".pkl")
                curr_param_organized_df_MIN.to_excel(OUTPUT_PATH + "/Input Data PKL Files/" + curr_param_organized_df_name_MIN + "_" + str(getMonteCarloDraws()) + ".xlsx")


def loadDataFrames():
    ## Initialize DataFrame variable
    print("Loading pre-organized input data...")

    for parameter in getParameters():

        statistical_parameters_array = ['mean', 'std', 'mode', 'min', 'max', 'a', 'b', 'x0', 'y0']
        for stat_param in statistical_parameters_array:
            try:
                # Load the pkl file
                curr_param_pkl_name = parameter + "_DF_" + stat_param
                curr_param_pkl = pd.read_pickle(OUTPUT_PATH + "/Input Data PKL Files/" + curr_param_pkl_name + "_" + str(getMonteCarloDraws()) + ".pkl")

                # Initialize the dataframe as a global variable
                globals()[curr_param_pkl_name] = curr_param_pkl
            except:
                #print(f"Did not load: {parameter}, {stat_param}")
                continue


def getEmptyStructuredDataframe():
    return globals()["CostVehicleCAPEX_DF_mean"]


def getMonteCarloDraws():
    # Set the number of Monte Carlo draws
    N_DRAWS = 1000

    return N_DRAWS


def getYears():
    return years_df.copy().values[:,0]


def getCountries():
    return countries_df.copy().values[:,0]


def getCountryISOCode(country):
    return countries_df.loc[countries_df['Countries'] == country, 'ISO Code'].values[0]


def getISOCodeCountry(iso_code):
    return countries_df.loc[countries_df['ISO Code'] == iso_code, 'Countries'].values[0]


def getSaleStatus():
    return sale_status_df.copy().values[:,0]


def getApplications():
    return applications_df.copy().values[:,0]


def getTechnologies():
    return technologies_df.copy().values[:,0]


def getParameters():
    return parameters_df.index.get_level_values('Parameters').unique().values


def getParameterNumScenarios(parameter):
    return parameters_df.loc[parameter,'Num Scenarios']


def getParameterDistributionType(parameter):
    return parameters_df.loc[parameter,'Distribution Type']


def getOutputs():
    return outputs_df.copy().values[:,0]


def getOutputStatisticalParameters():
    return ['mean', 'std']


def getScenarioCodes():
    return scenario_codes_df.copy()


def getScenarios():
    return scenarios_df.copy()


def addScenario(scenario_name, parameter_codes):
    existing_data_df = globals()['scenarios_df']

    # Create a new dataframe with the proper index
    new_data_df = pd.DataFrame(parameter_codes)
    new_data_df_transpose = new_data_df.T
    new_data_df_transpose.index = [scenario_name]

    # Then concatinate the new data with the old data
    df_combined = pd.concat([existing_data_df, new_data_df_transpose])

    # Reassign the global variable
    globals()['scenarios_df'] = df_combined


def getDimensions():
    return dimensions_df.copy()


def getInputPath():
    return INPUT_PATH


def getOutputPath():
    return OUTPUT_PATH


def getValues(scenario, parameter, statistical_parameter):
    """
    This function returns the parameter values for all countries and options
    of a specific parameter and scenario code (i.e. one column from data input)
    """

    # Initialize the correct dataframe
    curr_values_df = globals()[f'{parameter}_DF_{statistical_parameter}']

    # Initialize the parameter code(s) for the passed scenario
    parameter_code = scenarios_df.loc[scenario, parameter]

    # Initialize the correct column for
    column = getScenarioCodes().loc[parameter_code,:]

    return curr_values_df.loc[:,column]


def setOptimizationParameterValues(scenario, parameter, statistical_parameter, new_df):
    """
    This function sets the appropriate parameter df for the passed scenario for
    the model optimization script.
    """

    # Initialize the correct dataframe
    curr_values_df = globals()[f'{parameter}_DF_{statistical_parameter}']

    # Alter the dataframe
    curr_values_df[scenario] = new_df

    # Assign the altered dataframe back to the global variables list
    globals()[f'{parameter}_DF_{statistical_parameter}'] = curr_values_df


def getLCAValues(parameter, application, scenario, year):
    # Evaluate the dataframe
    df = eval(parameter + "_data")

    return df.loc[(application, scenario), year]


def computeLCA(technology, country, application, year, ssp_scenario, model_scenario):
    """
    This function computes the LCA (tCO2eq/km) for a vehicle technology in a certain application and year
    for a specified SSP scenario.

    Units returned: tCO2eq/km
    """

    sale_status = 'New'

    # Get the average annual km travelled (km/year) NOTE: parameter is independent of country
    akt = getValues(model_scenario, 'VehicleAnnualKmTravelled', 'mode').loc[(year, country, sale_status, application, technology), :].values[0]
    # Get the average lifetime of the vehicle (years) NOTE: parameter is independent of country
    vehicle_lifetime = getValues(model_scenario, 'VehicleLifetime', 'mode').loc[(year, country, sale_status, application, technology), :].values[0]
    # Compute the average total km travelled over full lifetime of the vehicle (km)
    total_km = akt*vehicle_lifetime

    # Conversion Factor
    diesel_kg_to_liter = 0.85

    if technology == 'ICE-G':
        # ----------VEHICLE PRODUCTION EMISSIONS-------------------
        # (kgCO2eq/km)
        vehicle_production_emissions_low = getLCAValues('ICEManufacturingEmissions', application, 'SSP2-RCP2.6', year) / total_km
        vehicle_production_emissions_high = getLCAValues('ICEManufacturingEmissions', application, 'SSP2-RCP6', year) / total_km
        vehicle_production_emissions_mean = (vehicle_production_emissions_low + vehicle_production_emissions_high) / 2


        # ----------VEHICLE USE EMISSIONS-------------------
        # Get the fossil fuel production emissions (kgCO2eq/L)
        ff_prod_emissions_factor_high = getLCAValues('FosFuelProdEmissions', application, 'Max', year)*(1/1000)*diesel_kg_to_liter
        ff_prod_emissions_factor_low = getLCAValues('FosFuelProdEmissions', application, 'Min', year)*(1/1000)*diesel_kg_to_liter
        # Get the ICE vehicle combustion emissions (kgCO2eq/L)
        vehicle_comb_emissions_factor_mid = getLCAValues('ICECombustionEmissions', application, 'Mean', year)*(1/1000)*diesel_kg_to_liter
        # Get the average fuel consumption (L/km)
        fuel_consumption = getValues(model_scenario, 'VehicleEnergyConsumption', 'mode').loc[(year, country, sale_status, application, technology), :].values[0]
        # Compute the vehicle use emissions (kgCO2eq/km)
        vehicle_use_emissions_low = (ff_prod_emissions_factor_high + vehicle_comb_emissions_factor_mid)*fuel_consumption
        vehicle_use_emissions_high = (ff_prod_emissions_factor_low + vehicle_comb_emissions_factor_mid)*fuel_consumption
        vehicle_use_emissions_mean = (vehicle_use_emissions_high + vehicle_use_emissions_low) / 2

        # ----------TOTAL EMISSIONS-------------------
        # (kgCO2eq/km)
        total_emissions_low = vehicle_production_emissions_low + vehicle_use_emissions_low
        total_emissions_high = vehicle_production_emissions_high + vehicle_use_emissions_high
        total_emissions_mean = vehicle_production_emissions_mean + vehicle_use_emissions_mean


    elif technology == 'BEV':
        # ----------VEHICLE PRODUCTION EMISSIONS-------------------
        # (kgCO2eq/km)
        vehicle_production_emissions_high = getLCAValues('BEVManufacturingEmissions', application, 'SSP2-RCP6', year) / total_km
        vehicle_production_emissions_low = getLCAValues('BEVManufacturingEmissions', application, 'SSP2-RCP2.6', year) / total_km
        vehicle_production_emissions_mean = (vehicle_production_emissions_high + vehicle_production_emissions_low) / 2

        # ----------SAS PRODUCTION EMISSIONS-------------------
        # SOLAR PV
        # Get the solar PV panel manufacturing emissions (kgCO2eq/m2)
        solarPV_emissions_low = getLCAValues('SASManufacturingEmissions', 'Solar PV', 'SSP2-RCP2.6', year)
        solarPV_emissions_high = getLCAValues('SASManufacturingEmissions', 'Solar PV', 'SSP2-RCP6', year)
        # Conversion factor for a m2 of solar pv to kW (m2/kW) [NOTE: assume 1000 W/m2]
        pv_m2_to_kW_conversion = 1
        # Get the solar PV capacity oversize factor
        solarPV_oversize_factor = getValues(model_scenario, 'SolarPVOversizeFactor', 'mode').loc[(year, country, sale_status, application, technology), :].values[0]
        # Get the solar PV capacity (kW)
        solar_pv_capacity = getValues(model_scenario, 'SolarPVCapacity', 'mode').loc[(year, country, sale_status, application, technology), :].values[0] * (1 + solarPV_oversize_factor)
        # Calculate solar PV emissions (kgCO2eq/km)
        solar_pv_emissions_low = solarPV_emissions_low*pv_m2_to_kW_conversion*solar_pv_capacity / total_km
        solar_pv_emissions_high = solarPV_emissions_high*pv_m2_to_kW_conversion*solar_pv_capacity / total_km

        # INVERTER
        # Determine the number of inverter replacements
        inverter_lifetime = getValues(model_scenario, 'InverterLifetime', 'mode').loc[(year, country, sale_status, application, technology), :].values[0]
        inverter_replacement_factor = math.ceil(vehicle_lifetime/inverter_lifetime)
        # Get the inverter manufacturing emissions (kgCO2eq/km)
        inverter_emissions_low = (getLCAValues('SASManufacturingEmissions', 'Inverter', 'SSP2-RCP2.6', year)*inverter_replacement_factor) / total_km
        inverter_emissions_high = (getLCAValues('SASManufacturingEmissions', 'Inverter', 'SSP2-RCP6', year)*inverter_replacement_factor) / total_km

        # STATIONARY BATTERY
        # Get the stationary battery manufacturing emissions (kgCO2eq/kg)
        stationary_battery_emissions_factor_low = getLCAValues('SASManufacturingEmissions', 'Stationary Battery', 'SSP2-RCP2.6', year)
        stationary_battery_emissions_factor_high = getLCAValues('SASManufacturingEmissions', 'Stationary Battery', 'SSP2-RCP6', year)
        # Get the stationary battery energy density (kWh/kg)
        stationary_battery_energy_density_low = getLCAValues('BatteryEnergyDensity', 'Stationary Battery', 'Low', year)
        stationary_battery_energy_density_high = getLCAValues('BatteryEnergyDensity', 'Stationary Battery', 'High', year)
        # Get the stationary battery capacity oversize factor
        stationary_battery_oversize_factor = getValues('BaseScenario', 'StatBatteryOversizeFactor', 'mode').loc[(year, country, sale_status, application, technology), :].values[0]
        # Get the stationary battery capacity (kWh)
        stationary_battery_capacity = getValues(model_scenario, 'StationaryBatteryCapacity', 'mode').loc[(year, country, sale_status, application, technology), :].values[0] * (1 + stationary_battery_oversize_factor)
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


    elif technology == 'ICE-SF':
        # ----------VEHICLE PRODUCTION EMISSIONS-------------------
        # (kgCO2eq/km)
        vehicle_production_emissions_low = getLCAValues('ICEManufacturingEmissions', application, 'SSP2-RCP2.6', year) / total_km
        vehicle_production_emissions_high = getLCAValues('ICEManufacturingEmissions', application, 'SSP2-RCP6', year) / total_km
        vehicle_production_emissions_mean = (vehicle_production_emissions_low + vehicle_production_emissions_high) / 2

        # ----------VEHICLE USE EMISSIONS-------------------
        # Get the fossil fuel production emissions (kgCO2eq/L)
        sf_prod_emissions_factor_high = getLCAValues('SynFuelProdEmissions', application, 'Max', year)*(1/1000)*diesel_kg_to_liter
        sf_prod_emissions_factor_low = getLCAValues('SynFuelProdEmissions', application, 'Min', year)*(1/1000)*diesel_kg_to_liter
        # Get the average fuel consumption (L/km)
        fuel_consumption = getValues(model_scenario, 'VehicleEnergyConsumption', 'mode').loc[(year, country, sale_status, application, technology), :].values[0]
        # Compute the vehicle use emissions (kgCO2eq/km)
        vehicle_use_emissions_low = sf_prod_emissions_factor_high*fuel_consumption
        vehicle_use_emissions_high = sf_prod_emissions_factor_low*fuel_consumption
        vehicle_use_emissions_mean = (vehicle_use_emissions_high + vehicle_use_emissions_low) / 2

        # ----------TOTAL EMISSIONS-------------------
        # (kgCO2eq/km)
        total_emissions_low = vehicle_production_emissions_low + vehicle_use_emissions_low
        total_emissions_high = vehicle_production_emissions_high + vehicle_use_emissions_high
        total_emissions_mean = vehicle_production_emissions_mean + vehicle_use_emissions_mean


    return total_emissions_mean


def getEmptyDataFrame(columns, param_num_scenarios="None", defined_col_list="NoList"):
    """
    Establish a MultiIndexed DataFrame for storage of model outputs.

    The format for storing data will look as follows:

    YEAR        COUNTRY     SALE STATUS     APPLICATION     TECHNOLOGY      VALUES
    2025        AL          New             2wheeler-small  ICE-G           []     []     []     []
    .           .           .               .               .               .      .      .      .
    2025        NI          New             4wheeler-small  ICE-G           []     []     []     []
    2025        NI          New             4wheeler-large  ICE-SF          []     []     []     []
    .           .           .               .               .               .      .      .      .
    2030        KN          Used            minibus         BEV             []     []     []     []
    2030        KN          Used            minibus         BEV             []     []     []     []
    .           .           .               .               .               .      .      .      .
    2040        ZW          Used            2wheeler-medium ICE-G           []     []     []     []
    .           .           .               .               .               .      .      .      .
    .           .           .               .               .               .      .      .      .
    .           .           .               .               .               .      .      .      .

    Depending on the passed "columns" parameter, the column names will change.
    --> If the "columns" parameter is 'scenarios', the returned storage
    dataFrame will have scenario names as column names.
    --> If the "columns" parameter is 'outputs' then the returend storage
    frame will look the same except the column names will be the different
    intermediate outputs (within a scenario).
    --> If the "columns" parameter is 'parameters' then the returend storage
    frame will againg look the same except the column names will be the
    different parameter names.


    returns: df_empty (MultiIndexed DataFrame)
    """

    # Initialize model dimensions
    # Initialize the interval lengths and arrays for each dimension in the dataframe (year, country, sale status, application, technology)
    year_interval = getCountries().size * getSaleStatus().size * getApplications().size * getTechnologies().size
    year_array = np.array([], dtype=int)
    country_interval = getSaleStatus().size * getApplications().size * getTechnologies().size
    country_array = np.array([], dtype=str)
    sale_status_interval = getApplications().size * getTechnologies().size
    sale_status_array = np.array([], dtype=str)
    application_interval = getTechnologies().size
    application_array = np.array([], dtype=str)
    technology_array = np.array([], dtype=str)

    for currYear in getYears():
        # Add to the countries array
        curr_year_array = np.full(year_interval, currYear)
        year_array = np.append(year_array, curr_year_array)

        for currCountry in getCountries():
            # Add to the countries array
            curr_country_array = np.full(country_interval, currCountry)
            country_array = np.append(country_array, curr_country_array)

            # Loop through the different sale status types
            for currSaleStatus in getSaleStatus():
                # Add to the geography array
                curr_sale_status_array = np.full(sale_status_interval, currSaleStatus)
                sale_status_array = np.append(sale_status_array, curr_sale_status_array)

                # Loop through the applications array
                for currApplication in getApplications():
                    # Add to the application array
                    curr_application_array = np.full(application_interval, currApplication)
                    application_array = np.append(application_array, curr_application_array)

                    # Add to the technology array
                    technology_array = np.append(technology_array, getTechnologies())

    # Concatenate the previously created columns
    data = np.column_stack((year_array, country_array, sale_status_array, application_array,technology_array))

    # Construct a DataFrame from the larger 2D data
    df = pd.DataFrame(data, columns=['Year','Country','Sale Status','Application','Technology'])

    # add in either the scenarios or intermediate outputs as additional columns
    if columns == 'scenarios':
        # add in columns for each scenario
        for curr_scenario in getScenarios().index:
                # NOTE: can also use df.insert() function which allows column insertion at a certain specified index
                df[curr_scenario] = np.zeros((df.shape[0], 1))

        # Now create the MultiIndex DataFrame
        df.set_index(['Year','Country','Sale Status','Application','Technology'], inplace=True)

        # Convert the DataFrame to dtype float64
        df = df.astype(object)

         # Assign df to the empty data frame variable
        df_empty = df

    elif columns == 'outputs':
        # add in columns for each output
        for curr_output in getOutputs():
                # NOTE: can also use df.insert() function which allows column insertion at a certain specified index
                df[curr_output] = np.zeros((df.shape[0], 1))

        # Now create the MultiIndex DataFrame
        df.set_index(['Year','Country','Sale Status','Application','Technology'], inplace=True)

        # Convert the DataFrame to dtype float64
        df = df.astype(object)

         # Assign df to the empty data frame variable
        df_empty = df

    elif columns == 'scenario_codes':
        # add in columns for each scenario code
        # Get the correct number of scenario codes as passed into this function
        scenario_codes_array = getScenarioCodes().values[0:param_num_scenarios]
        for curr_scenario_code in scenario_codes_array:
                # NOTE: can also use df.insert() function which allows column insertion at a certain specified index
                df[curr_scenario_code] = np.zeros((df.shape[0], 1))

        # Now create the MultiIndex DataFrame
        df.set_index(['Year','Country','Sale Status','Application','Technology'], inplace=True)

        # Convert the DataFrame to dtype float64
        df = df.astype(object)

         # Assign df to the empty data frame variable
        df_empty = df

    elif columns == 'parameters':
        # add in columns for each parameter
        for curr_parameter in getParameters():
             # NOTE: can also use df.insert() function which allows column insertion at a certain specified index
            df[curr_parameter] = np.zeros((df.shape[0], 1))

        # Now create the MultiIndex DataFrame
        df.set_index(['Year','Country','Sale Status','Application','Technology'], inplace=True)

        # Convert the DataFrame to dtype float64
        df = df.astype(object)

         # Assign df to the empty data frame variable
        df_empty = df

    elif columns == 'country_outputs':
        # use countries as index
        df = getCountries().index.to_frame(index=False)

        # add in columns for each output
        for curr_output in getOutputs().values:
             # NOTE: can also use df.insert() function which allows column insertion at a certain specified index
            df[curr_output] = np.zeros((df.shape[0], 1))

        # Now create the MultiIndex DataFrame
        df.set_index(['Countries'], inplace=True)

        # Convert the DataFrame to dtype float64
        df = df.astype(object)

         # Assign df to the empty data frame variable
        df_empty = df

    elif columns == 'defined_col_list':
        # use the passed list of column names as index
        columns = defined_col_list

        # add in columns for each scenario
        for curr_col in columns:
                # NOTE: can also use df.insert() function which allows column insertion at a certain specified index
                df[curr_col] = np.zeros((df.shape[0], 1))

        # Now create the MultiIndex DataFrame
        df.set_index(['Year','Country','Sale Status','Application','Technology'], inplace=True)

        # Convert the DataFrame to dtype float64
        df = df.astype(object)

         # Assign df to the empty data frame variable
        df_empty = df

    return df_empty
