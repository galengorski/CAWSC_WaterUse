from climate_data_collector import GridMETDataCollector as gmet
import time

if __name__ == "__main__":

    # climateFilter = ['etr', 'pet', 'pr', 'sph', 'srad', 'tmmn', 'tmmx', 'vs']
    climateFilter = ["pet", "tmmn", "tmmx"]
    shpfile = r"D:\Workspace\projects\machine_learning\data\dataset\gis\GUall_repair_featproj.shp"
    zoneField = "GNIS_ID"
    huc2 = 13
    filter_field = {"HUC2": str(int(huc2)).zfill(2)}
    # intialize the data collector
    gmetDC = gmet.DataCollector(
        folder=r"D:\Workspace\projects\machine_learning\data\dataset\training_data\13"
    )
    start_time = time.time()
    climateData = gmetDC.get_data(
        shpfile,
        zoneField,
        climate_filter=climateFilter,
        year_filter="2000-2015",
        multiprocessing=True,
        chunksize=20,
        filter_field=filter_field,
    )
    end_time = time.time()
    print(end_time - start_time)
