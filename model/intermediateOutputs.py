# FILE: intermediateOutputs.py
# PROJECT: LCOC Model
# MODULE DESCRIPTION: This is the intermediate storage output module.
# Intermediate results will be collected/managed and stored in this module.


# Import statements
import pandas as pd
import numpy as np
import externalParameters as ep


def initialize():
    # Initialize (global) empty storage frames for all types of outputs
    for output in ep.getOutputs():
        for output_stat_param in ep.getOutputStatisticalParameters():
            curr_var = output + "_DF_" + output_stat_param
            globals()[curr_var] = ep.getEmptyDataFrame('scenarios')     # NOTE: for all outputs we want the scenarios as columns


    # Initialize (global) empty storage frames for all scenarios
    for scenario in ep.getScenarios().index:
        for output_stat_param in ep.getOutputStatisticalParameters():
            curr_var = scenario + "_DF_" + output_stat_param
            globals()[curr_var] = ep.getEmptyDataFrame('outputs')       # NOTE: for all scenario dataframes we want outputs as columns




def recordScenario(curr_outputs, scenario, output_stat_param):
    curr_df = globals()[scenario + "_DF_" + output_stat_param]
    curr_df[:] = curr_outputs.values


def recordOutput(curr_output, scenario, result_type, output_stat_param):
    curr_df = globals()[result_type + "_DF_" + output_stat_param]
    curr_df[scenario] = curr_output.values




def storeScenarios():
    for scenario in ep.getScenarios().index:
        for output_stat_param in ep.getOutputStatisticalParameters():
            curr_str = scenario + "_DF_" + output_stat_param
            curr_df = eval(curr_str)
            curr_df.to_pickle(ep.OUTPUT_PATH + curr_str + ".pkl")
            curr_df.to_excel(ep.OUTPUT_PATH + "_Scenario_" + curr_str + ".xlsx", sheet_name = curr_str)

def storeOutputs():     # NOTE: this stores all TCO output parameters (as individual files) for each scenario
    for output in ep.getOutputs():
        for output_stat_param in ep.getOutputStatisticalParameters():
            curr_str = output + "_DF_" + output_stat_param
            curr_df = eval(curr_str)
            curr_df.to_pickle(ep.OUTPUT_PATH + curr_str + ".pkl")
            curr_df.to_excel(ep.OUTPUT_PATH + "_" + curr_str + ".xlsx", sheet_name = curr_str)




def readScenario(scenario):
    curr_scenario = pd.read_pickle(ep.OUTPUT_PATH + scenario + ".pkl")
    return curr_scenario

def readOutput(output):
    curr_output = pd.read_pickle(ep.OUTPUT_PATH + output + ".pkl")
    return curr_output
