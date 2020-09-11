import sys, os
from .census_data_collector.censusdc.utils import CensusTimeSeries, create_filter, TigerWebMapServer
from .climate_data_collector import GridMETDataCollector as gmet
from .climate_data_collector.utils import *
import shapefile
import time
import pandas as pd
import datetime

class DataCollector(object):
    def __init__(self, service_area_shapefile = None, apikey = None, fieldName = '',
                 output_folder = ''):
        self.service_area_shapefile = service_area_shapefile
        with open(apikey) as api:
            apikey = api.readline().strip()
        self.apikey = apikey
        self.fieldName = fieldName
        self.output_folder = output_folder

    def isnumeric(self, s):
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            return False

    def isdatetime(self, s):
        if isinstance(s, (datetime.date, datetime.datetime)):
            return True
        else:
            return False

    def isbytes(self, s):
        if isinstance(s, bytes):
            return True
        else:
            return False

    def get_shapefile_df(self, shp):
        """
        Get attribute table
        :return:
        """
        with shapefile.Reader(shp) as foo:
            header = [i[0].lower() for i in foo.fields[1:]]
            data = {i: [] for i in header}

            for record in foo.records():
                for ix, v in enumerate(record):
                    if self.isnumeric(v):
                        data[header[ix]].append(v)
                    elif self.isdatetime(v):
                        data[header[ix]].append(v)
                    else:
                        if self.isbytes(v):
                            v = v.decode()
                        data[header[ix]].append(v.lower())

        df = pd.DataFrame.from_dict(data)
        return df

    def get_training_data_for_shapefile(self, thread_num=1, years=None, climateFilter=None):

        start_time = time.time()
        shp = self.service_area_shapefile
        dfS = self.get_shapefile_df(shp)

        if 1:  ## census data
            cfilter = None
            ts = CensusTimeSeries(shp, self.apikey, field=self.fieldName.lower(), filter=cfilter)
            fidw = open(os.path.join(self.output_folder, 'failed_pid_{}.dat'.format('None')), 'w')
            avail_years = [year for year in TigerWebMapServer.base.keys()]

            pull_years = []
            for yr in years:
                if yr in avail_years:
                    pull_years.append(yr)

            for ipid, pid in enumerate(dfS[self.fieldName.lower() ].values):
                try:
                    print("====== Downloading {}. Total Downloaded is {}".format(pid, int(
                        100 * float(ipid) / len(dfS[self.fieldName.lower() ].values))))

                    df = ts.get_timeseries(pid, polygons="internal", multithread=True,
                                           thread_pool=thread_num, verbose=2, years=pull_years)  # , years = years_list

                    df2 = CensusTimeSeries.interpolate(df, skip_years=range(2005, 2010),
                                                       min_extrapolate=1990,
                                                       max_extrapolate=2020,
                                                       kind='slinear',
                                                       discretization='daily')

                    df2.to_csv(os.path.join(self.output_folder, "cs_{}.csv".format(pid)),
                               index=False)
                except:
                    fidw.write('{}\n'.format(pid))
            fidw.close()

        ## download climate data

        shpfile = self.service_area_shapefile
        zoneField = self.fieldName
        # filter_field = {'HUC2': str(int(huc2)).zfill(2)}
        year_filter = '{},{}'.format(min(years), max(years))

        gmetDC = gmet.DataCollector(folder=self.output_folder)
        start_time = time.time()

        climateData = gmetDC.get_data(shpfile, zoneField, climate_filter=climateFilter,
                                      year_filter=year_filter, multiprocessing=True, chunksize=20,
                                      filter_field=None)

        end_time = time.time()
        print(end_time - start_time)

    def get_training_data(self, huc2 = None, thread_num =  1, years = None, climateFilter = None):
        """

        :param huc2:
        :param thread_num:
        :param years:
        :return:
        """

        start_time = time.time()
        shp = self.service_area_shapefile
        if 1:

            cfilter = create_filter(shp,
                                    criteria={'HUC2': str(int(huc2)).zfill(2)},
                                    return_field=self.fieldName)


            ts = CensusTimeSeries(shp, self.apikey, field=self.fieldName, filter=cfilter)
            fidw = open(os.path.join(self.output_folder, 'failed_pid_{}.dat'.format(huc2)), 'w')
            avail_years = [year for year in TigerWebMapServer.base.keys()]

            pull_years = []
            for yr in years:
                if yr in avail_years:
                    pull_years.append(yr)

            for ipid, pid in enumerate(cfilter):
                try:
                    print("====== Downloading {}. Total Downloaded is {}".format( pid, int(100 * float(ipid)/len(cfilter))))

                    df = ts.get_timeseries(pid, polygons="internal", multithread=True,
                                           thread_pool= thread_num, verbose=2, years = pull_years)#, years = years_list

                    df2 = CensusTimeSeries.interpolate(df, skip_years=range(2005, 2010),
                                                       min_extrapolate=2000,
                                                       max_extrapolate=2015,
                                                       kind='slinear',
                                                       discretization='daily')

                    df2.to_csv(os.path.join(self.output_folder, "cs_{}.csv".format(pid)),
                               index=False)
                except:
                    fidw.write('{}\n'.format(pid))
            fidw.close()

        ## download climate data
        self.get_climate(huc2 = huc2, thread_num= thread_num, years = years, climateFilter = climateFilter)

    def get_climate(self, huc2 = None, thread_num =  1, years = None,  climateFilter = None):

        # climateFilter = ['etr', 'pet', 'pr', 'sph', 'srad', 'tmmn', 'tmmx', 'vs']

        shpfile = self.service_area_shapefile
        zoneField = self.fieldName
        filter_field = {'HUC2': str(int(huc2)).zfill(2)}
        if years is None:
            years = [2000, 2020]
        year_filter = '{},{}'.format(min(years),max(years))

        gmetDC = gmet.DataCollector(folder=self.output_folder)

        start_time = time.time()
        climateData = gmetDC.get_data(shpfile, zoneField, climate_filter=climateFilter,
                                      year_filter=year_filter, multiprocessing=True, chunksize=20,
                                      filter_field=filter_field)
        end_time = time.time()
        print(end_time - start_time)
        pass


    def get_prediction_data(self, huc2 = 13, apikey = '', thread_num=8, years = [2000,2001]):
        pass

if __name__ == "__main__":
    shp = r"C:\work\water_use\dataset\gis\service_area_6_25_2020_proj.shp"
    apikey = r"C:\work\water_use\dataset\others\apikey"
    outws = r"C:\work\water_use\dataset\training_data\13"
    fieldname = 'WSA_AGG_ID'.lower()
    dc = DataCollector(service_area_shapefile=shp, apikey=apikey, fieldName=fieldname, output_folder= outws)
    import time
    start = time.time()
    dc.get_training_data(huc2=13, thread_num=20, climateFilter = ['pet', 'tmmn', 'tmmx'], years= [2000,2018])
    t_end = time.time()
    print("-------------------> download time is {}".format(t_end-start))
