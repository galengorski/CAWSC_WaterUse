import GridMETDataCollector3 as gmet
import time
import os
import numpy as np


if __name__ == '__main__':
    #shapelist =['pajaro', 'idaho', 'mississippi', 'colorado']
    shapelist = ['colorado']
    # zonefields = ['OBJECTID', 'WaterRight', 'SITENO', 'PARCEL_ID']
    for shape_ind, shape in enumerate(shapelist):
        start_time = time.time()
        years = ['2015', '2020']
        # shpfile = os.path.join('..', 'projected_shapes', shape + '_Project1.shp')
        shpfile = os.path.join('..', 'projected_shapes', shape[0:2] + '_new.shp')
        # zoneField = zonefields[shape_ind]
        zoneField = 'pilot_id'

        # climate types to be processed
        climateFilter = ['etr', # Reference ET
                         'pet', # Potential ET
                         'pr', # Precipitation
                         'sph', # Specific Humidity
                         'srad', # Shortwave Radiation
                         'tmmn', # Minimum Temperature
                         'tmmx', # Maximum Temperature
                         'vs', # Wind Velocity
                            ]

        for year in years:
            print('\nprocessing ' + shape + ' ' + year + '...')
            # intialize the data collector
            gmetDC = gmet.DataCollector()
            gmetDC.get_data(shpfile, zoneField, climate_filter=climateFilter,
                            year_filter=year, multiprocessing=True, save_to_csv=True, out_folder=os.path.join('..', 'output', shape, year))

        print('\n' + shape + ' took ', np.ceil((time.time() - start_time) / 60) , ' minutes')