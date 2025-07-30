# FILE: main.py
# PROJECT: Africa Passenger Vehicle TCO Model
# SCRIPT DESCRIPTION: This is the main module where the model should be run.


"""
SYNTAX NOTES:

global variables:   GLOBAL_VARIABLES

function variables: function_variables

function definitions: functionDefinition

"""


## IMPORT PACKAGES
import numpy as np
import pandas as pd
import time
import externalParameters as ep
import intermediateOutputs as io
import totalCostOfOwnership as tco
from importlib import reload
from tqdm import tqdm

start_time = time.time()

print("")
print(">>> RunMainTCOModel >>>")
print("")


## INITIALIZE MODULES

print("Initializing model...")

# Initialize the external parameters
ep.initialize(organize_data_bool=True, record_data_bool=True)

# Initialize the storage matrices for desired intermediate outputs
io.initialize()



## COMPUTE AFRICA TCO AND STORE RESULTS

print("Computing TCO...")


# Get list of scenarios and corresponding parameter codes
scenarios_df = ep.getScenarios()

# Run model over all scenarios
for scenario in scenarios_df.index:

    # Set global variable to the current scenario
    tco.setScenario(scenario)
    tco.printScenario()

    # Compute and record TCO in current scenario for all countries, sale status, applications, technologies
    tco.calculateTCO(scenario)


## SAVE TCO RESULTS

print("Storing TCO results...")

# Save all outputs of a scenario to a pickle and an excel file (for all scenarios)
io.storeScenarios()

# Save all scenarios for an output to a pickle and an excel file (for all outputs)
io.storeOutputs()




print('---------------------------------------------------------------')
print('CODE RUN TIME:')
print('--- ',np.around((time.time() - start_time), decimals=3), 'seconds ---')
print('--- ',np.around((time.time() - start_time)/60, decimals=3), 'minutes ---')
print('---------------------------------------------------------------')
