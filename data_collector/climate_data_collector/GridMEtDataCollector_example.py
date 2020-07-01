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
    ohioZonesField = 'PLANT_ID'

    # climate types to be processed
    climateFilter = ['etr', 'pet', 'pr', 'sph', 'srad', 'tmmn', 'tmmx', 'vs']
    # climateFilter = ['etr', 'pet']
    # climateFilter= ['etr']


    # process a single shapefile

    # intialize the data collector
    gmetDC = gmet.DataCollector(folder='GIS')


    # test = gmetDC.get_data(ohioZones, zoneField, climate_filter=climateFilter,
    #                 year_filter='2000-2005', multiprocessing=True, cpu_count=None,
    #                 save_to_csv=True, chunksize=2, filter_field={'PLANT_ID': ['HCWP']})

    # print(test)


    # shpFile = os.path.join(shpFolder, 'GU_HUC2_13_WGS84.shp')
    # shpFile = os.path.join(shpFolder, 'GU_ALL_WGS84.shp')
    shpFile = os.path.join(shpFolder, 'GU_ALL_FROM_AYMAN_WGS84.shp')
    # shpFile = os.path.join(shpFolder, 'GU_ALL_FROM_AYMAN_WGS84_HUC2_13.shp')
    zoneField = 'GNIS_ID'
    filterField = {'HUC2': '11'}
    # testing (20 chunk) (chunk 6)
    # filterField = {'GNIS_ID': ['2409116', '2409120', '2409140', '2409141', '2409152', '2409159', '2409169', '2409174', '2409176', '2409235', '2409238', '2409254', '2409256', '2409267', '2409269', '2409275', '2409276', '2409277', '2409278', '2409282']}
    # #
    # count = 1
    # while count < 10:
    s = time.time()

    climateData = gmetDC.get_data(shpFile, zoneField, climate_filter=climateFilter,
                    year_filter='2000-2015', multiprocessing=True, cpu_count=None,
                                  save_to_csv=True, chunksize=20, filter_field=filterField)
    # print('COMPLETE COUNT', count)
    # count += 1

    # print(climateData['pet'])

    print('time', time.time() - s)
    print()
#
    # # loop through mutlipel shapefiles and process
    # shps = {'OH_WGS84.shp': 'PLANT_ID', 'ABCWUA_WGS84.shp': 'CN'}
    # for shp, zoneField in shps.items():
    #
    #     zoneShp = os.path.join(shpFolder, shp)
    #
    #     climateData = gmetDC.get_data(zoneShp, zoneField, climate_filter=['pet'],
    #                 year_filter='2000-2015', multiprocessing=True)
