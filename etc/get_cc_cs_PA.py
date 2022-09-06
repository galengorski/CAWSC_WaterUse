import os, sys

sys.path.insert(0, r"C:\work\water_use\CAWSC_WaterUse")

import data_collector

import time
import shapefile

"""
Just donwload data...
"""


# ----------------------------
# Census data collector
# ----------------------------
if __name__ == "__main__":
    apikey = "9aab719a1add05f9d46187b0a96cf32233300533"

    shp_file = r"C:\work\water_use\dataset\gis\PA_sites.shp"
    apikey = r"C:\work\water_use\dataset\others\apikey"
    fieldname = "WSA_AGG_ID".lower()
    outws = r"C:\work\water_use\dataset\dailywu\pa"

    dc = data_collector.DataCollector(
        service_area_shapefile=shp_file,
        apikey=apikey,
        fieldName=fieldname,
        output_folder=outws,
    )
    import time

    start = time.time()
    dc.get_training_data_for_shapefile(
        thread_num=20,
        climateFilter=["pet", "pr", "tmmn", "tmmx"],
        years=[1990, 2000, 2001, 2003] + list(range(2004, 2020)),
    )
    endd = time.time()

    print(endd - start)
