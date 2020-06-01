from climate_data_collector import GridMETDataCollector as gmet
import time
if __name__ == "__main__":

    #climateFilter = ['etr', 'pet', 'pr', 'sph', 'srad', 'tmmn', 'tmmx', 'vs']
    climateFilter = ['pet', 'tmmn', 'tmmx']
    shpfile =  r"D:\Workspace\projects\machine_learning\data\gis\huc13_proj.shp"
    zoneField = 'GNIS_ID'
    # intialize the data collector
    gmetDC = gmet.DataCollector(folder = r"D:\Workspace\projects\machine_learning\data\dataset\training_data\hun2_13")
    start_time = time.time()
    climateData = gmetDC.get_data(shpfile, zoneField, climate_filter=climateFilter,
                    year_filter='2000-2015', multiprocessing= True )
    end_time = time.time()
    print(end_time-start_time)