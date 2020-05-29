import GridMETDataCollector as gmet
import time
import os

if __name__ == '__main__':
    # start_time = time.time()

    # input shapefile folder
    # wgs84 shapefiles
    shpFolder = os.path.join('GIS', 'WaterUseAreas', 'WGS84')
    ohioZones = os.path.join(shpFolder, 'OH_WGS84.shp')
    # field with the zone names
    zoneField = 'PLANT_ID'

    # climate types to be processed
    climateFilter = ['etr', 'pet', 'pr', 'sph', 'srad', 'tmmn', 'tmmx', 'vs']

    # intialize the data collector
    gmetDC = gmet.DataCollector()

    # process a single shapefile
    # s = time.time()
    gmetDC.get_data(ohioZones, zoneField, climate_filter=climateFilter,
                    year_filter='2000-2015', multiprocessing=True)
    # print('time', time.time() - s)

    # loop through mutlipel shapefiles and process
    shps = {'OH_WGS84.shp': 'PLANT_ID', 'ABCWUA_WGS84.shp': 'CN'}
    for shp, zoneField in shps.items():

        zoneShp = os.path.join(shpFolder, shp)

        climateData = gmetDC.get_data(zoneShp, zoneField, climate_filter=['pet'],
                    year_filter='2000-2015', multiprocessing=True)
