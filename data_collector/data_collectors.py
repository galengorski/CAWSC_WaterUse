import sys, os
from census_data_collector.censusdc.utils import CensusTimeSeries, create_filter
from climate_data_collector import GridMETDataCollector as gmet
import shapefile
import time

class DataCollector(object):
    def __init__(self, service_area_shapefile = None, apikey = None, fieldName = '',
                 output_folder = ''):
        self.service_area_shapefile = service_area_shapefile
        with open(apikey) as api:
            apikey = api.readline().strip()
        self.apikey = apikey
        self.fieldName = fieldName
        self.output_folder = output_folder

    def get_training_data(self, huc2 = 3, thread_num=8, years = [2000,2001]):
        """

        :param huc2:
        :param thread_num:
        :param years:
        :return:
        """

        start_time = time.time()
        shp = self.service_area_shapefile
        cfilter = create_filter(shp,
                                criteria={'HUC2': str(int(huc2)).zfill(2)},
                                return_field=self.fieldName)


        ts = CensusTimeSeries(shp, self.apikey, field=self.fieldName, filter=cfilter)

        # we can also use the default "internal" flag to intesect with the polygons
        # that are associated with our filter and have already been loaded
        years_list = [2000,2001,2002,2003,2004,2011,2012,2013]
        for pid in cfilter:
            df = ts.get_timeseries(pid, polygons="internal", multithread=True,
                                   thread_pool=8, verbose=1, years=years_list)

            df.to_csv(os.path.join(self.output_folder,"census_huc2_{}.csv".format(pid)),
                      index=False)

            df2 = CensusTimeSeries.interpolate(df, skip_years=range(2005, 2010),
                                               min_extrapolate=2000,
                                               max_extrapolate=2015,
                                               kind='slinear',
                                               discretization='daily')

            df2.to_csv(os.path.join(self.output_folder, "{}_interp.csv".format(pid)),
                       index=False)
        pass

    def get_prediction_data(self, huc2 = 13, apikey = '', thread_num=8, years = [2000,2001]):
        pass

if __name__ == "__main__":
    shp = r"D:\Workspace\projects\machine_learning\data\GUall_no_lakes_erased_v1_1_geo_huc2assigned\GUall_no_lakes_erased_v1_1_geo_huc2assigned.shp"
    apikey = r"D:\Workspace\projects\machine_learning\scripts\apikey"
    outws = r"D:\Workspace\projects\machine_learning\data\dataset"
    dc = DataCollector(service_area_shapefile=shp, apikey=apikey, fieldName='GNIS_ID', output_folder= outws)

    dc.get_training_data(huc2=13, thread_num=4, years = [2000, 2010])


def get_training_data(huc2=13, service_area_id='1852627', apikey='1234', thread_num=8, start_year=2015, end_year=2016):
    """

    """
    # (1) read in service area shapefile attribute table
    shp_file = r"shapefiles\serviceArea_1852627.shp"


    # (3) format data/add data for census collector
    shp = shapefile.Reader(shp_file)
    polygon = shp.shape(0)
    ts = CensusTimeSeries(shp_file, apikey, field="GNIS_ID")

    # (4) Loop over service areas & download census data (right now we're just testing 1 service area)
    #for serv_area in service_area_list:
    serv_area = service_area_id
    df = ts.get_timeseries(serv_area, verbose=2, polygons=polygon, multithread=True,
                           thread_pool=thread_num, years=list(range(start_year, end_year)))
    print(df)
    df.to_csv('1852627_census.csv')

    # (4) Download climate data
    yearFilter = str(start_year)+'-'+str(end_year)
    climateFilter = ['pet', 'pr', 'tmmx']
    zoneShpPath = shp_file
    climateData = gmet.get_data(zoneShpPath, zone_field="GNIS_ID", year_filter=yearFilter, climate_filter=climateFilter,
                                multiprocessing=True)
    print(climateData)

    # (5) Link data to Water use data
    # pass


# ------------------------------------
# Download HUC12 data
# -----------------------------------

# test function
get_training_data(huc2=13, service_area_id='1852627', apikey='1234',
                  start_year=2010, end_year=2019)
