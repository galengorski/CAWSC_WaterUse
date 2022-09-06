import os, sys

sys.path.append(r"C:\work\water_use\census-data-collector")

from censusdc.utils import CensusTimeSeries
import time
import shapefile

if __name__ == "__main__":
    # ----------------------------
    # Test Census data collector
    # ----------------------------

    if 1:

        shp_file = r".\data\OH_WGS.shp"  # (1) (WGS projection)
        apikey = "9aab719a1add05f9d46187b0a96cf32233300533"  # (2)

        start_time = time.time()
        ts = CensusTimeSeries(shp_file, apikey, field="plant_id")
        shp = shapefile.Reader(shp_file)
        polygon = shp.shape(0)
        years = list(range(2010, 2018))  # 1990 - 2018 but remove 2005-2009
        df = ts.get_timeseries(
            "drwp",
            polygons=polygon,
            multithread=True,
            thread_pool=20,
            years=years,
        )
        df.to_csv("drwp_pop.csv")
        time_end = time.time()
        print(time_end - start_time)
