# FILE: getSolarIrradiationData.py
# PROJECT: Africa Passenger Vehicle TCO Model
# SCRIPT DESCRIPTION: This is the module that sources solar irradiation data at
# a high resolution from the library pvlib.

import pandas as pd
import pvlib
import requests
import random
import time


print("")
print(">>> GetSolarIrradiationData >>>")
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


input_GeoData = pd.read_excel(INPUT_PATH + 'GeoData_IDXCountryCoords.xlsx', sheet_name='GeoData_IDXCountryCoords',
                              keep_default_na=False)
df = pd.DataFrame(columns=["Index", "Country (ISO)", "Latitude", "Longitude", "P(PV) [W/kW]"])


def get_loc_SI(latitude, longitude):
    while True:
        try:
            pv_output, meta, inputs = pvlib.iotools.get_pvgis_hourly(
                latitude=latitude, longitude=longitude, start=2020, end=2020, raddatabase="PVGIS-ERA5",
                components=True, outputformat='json', usehorizon=False, userhorizon=None, pvcalculation=True,
                peakpower=1, pvtechchoice='crystSi', mountingplace='free', loss=0, trackingtype=0,
                optimal_surface_tilt=False, optimalangles=True, url='https://re.jrc.ec.europa.eu/api/v5_2/',
                map_variables=True, timeout=30)
            return pv_output.iloc[:, 0].values
        except requests.HTTPError as e:
            print(e)
            if "Location over the sea" in str(e):
                print("X SEA", latitude, longitude)
                return "Sea", "Sea"
            else:
                print("X ERR", round(latitude, 2), round(longitude, 2))
                latitude += random.uniform(-0.05, 0.05)
                longitude += random.uniform(-0.05, 0.05)
                print("X TRY", round(latitude, 2), round(longitude, 2))
        except Exception as e:
            return "Error", "Error"


for i in range(700, 710): #0, 2560):
    pv_output = get_loc_SI(input_GeoData.iloc[i, 2], input_GeoData.iloc[i, 3])
    new_row = {"Index": input_GeoData.iloc[i, 0],
               "Country (ISO)": input_GeoData.iloc[i, 1],
               "Latitude": input_GeoData.iloc[i, 2],
               "Longitude": input_GeoData.iloc[i, 3],
               "P(PV) [W/kW]": pv_output}  # Das Numpy-Array wird in eine Liste konvertiert
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print(input_GeoData.iloc[i, 0], input_GeoData.iloc[i, 1], input_GeoData.iloc[i, 2], input_GeoData.iloc[i, 3])


def write_csv_with_pv_output(df, filename):
    df["P(PV) [W/kW]"] = df["P(PV) [W/kW]"].apply(lambda x: " ".join(map(str, x)))
    df.to_csv(filename, index=False)


write_csv_with_pv_output(df, INPUT_PATH + "GeneratedSolarIrradiationData.csv")
