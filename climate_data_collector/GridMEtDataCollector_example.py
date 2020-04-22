import GridMETDataCollector as gmet
import time

if __name__ == '__main__':
    start_time = time.perf_counter()

    # input shapfile
    zoneShpPath = r'.\data\test_data\ABCWUA_GridMETPrj.shp'
    # field with the zone names
    zoneField = 'FieldTest'

    # get all dates till current date
    # climateData = gmet.get_data(zoneShpPath, zone_field=zoneField)

    yearFilter = '2010-2015'
    climateFilter = ['pet', 'pr']
    # get 2010-2015 using year_filter
    climateData = gmet.get_data(zoneShpPath, zone_field=zoneField,
            year_filter=yearFilter, climate_filter=climateFilter,
                                multiprocessing=False)

    # print(climateData.keys())
    # print(climateData)
    end_time = time.perf_counter()
    print('run time', end_time - start_time)



