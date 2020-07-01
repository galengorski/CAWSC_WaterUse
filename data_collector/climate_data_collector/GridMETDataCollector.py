import pycurl
import sys
import os
import numpy as np
import xarray as xr
import pandas as pd
import time
from osgeo import ogr
import rtree
import datetime as dt
import math
import multiprocessing as mp
from functools import reduce
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
# file_handler = logging.FileHandler('GridMETDataCollector.log', mode='w')
file_handler = logging.FileHandler('GridMETDataCollector.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class _DataDict(dict):

    def __init__(self):
        super(_DataDict, self).__init__()

    def add_val(self, key1, key2, val):
        if key1 not in self:
            self[key1] = {key2: val}
        elif key2 not in self[key1]:
            self[key1][key2] = val
        else:
            self[key1][key2] += val


class DataCollector(object):

    def __init__(self, folder=None, verbose=True):

        # build gridMET grid
        self._met_grid = _Grid(verbose=verbose)
        # dictionary to store met data
        self._met_dict = {}
        # folder of gridMET nc files
        self.out_folder = folder
        self._verbose = verbose

        self._params = {
            'etr': {
                'nc': 'agg_met_etr_1979_CurrentYear_CONUS',
                'var': 'daily_mean_reference_evapotranspiration_alfalfa',
                'col': 'etr_mm'},
            'pet': {
                'nc': 'agg_met_pet_1979_CurrentYear_CONUS',
                'var': 'daily_mean_reference_evapotranspiration_grass',
                'col': 'eto_mm'},
            'pr': {
                'nc': 'agg_met_pr_1979_CurrentYear_CONUS',
                'var': 'precipitation_amount',
                'col': 'prcp_mm'},
            'sph': {
                'nc': 'agg_met_sph_1979_CurrentYear_CONUS',
                'var': 'daily_mean_specific_humidity',
                'col': 'q_kgkg'},
            'srad': {
                'nc': 'agg_met_srad_1979_CurrentYear_CONUS',
                'var': 'daily_mean_shortwave_radiation_at_surface',
                'col': 'srad_wm2'},
            'vs': {
                'nc': 'agg_met_vs_1979_CurrentYear_CONUS',
                'var': 'daily_mean_wind_speed',
                'col': 'u10_ms'},
            'tmmx': {
                'nc': 'agg_met_tmmx_1979_CurrentYear_CONUS',
                'var': 'daily_maximum_temperature',
                'col': 'tmax_k'},
            'tmmn': {
                'nc': 'agg_met_tmmn_1979_CurrentYear_CONUS',
                'var': 'daily_minimum_temperature',
                'col': 'tmin_k'},
        }

    @property
    def met_dict(self):

        """
           Method to get the grid met data dictionary
           Returns
           -------
               dict : {met name: met dataframe}
           """
        return self._met_dict

    def divide_chunks(self, l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def get_data(self, shp, zone_field, year_filter=None, climate_filter=None,
             multiprocessing=False, chunksize=None, save_to_csv=False, cpu_count=None,
                 filter_field=None):

        # get shapefile projection epsg code
        shp_epsg = _get_spatial_ref(shp)
        if shp_epsg != 4326:  # WGS84 GridMET prj
            logger.error('PROJECTION ERROR: The shapefile projection must be WGS84')
            raise Exception('PROJECTION ERROR: The shapefile '
                            'projection must be WGS84')

        # download grid met data
        self._downloader(climate_filter=climate_filter)
        if chunksize is not None:
            save_to_csv = True

        shp_geo_list, shp_attr_list = _read_shapefile(shp, zone_field, filter_field=filter_field)
        # shp_geo, shp_attr = _read_shapefile(shp, zone_field)
        # print('shape attr list', shp_attr_list)
        if self._verbose:
            print('Processing climate data')
        # logger.info('Processing climate data')
        if chunksize is not None:
            # logger.info('Dividing Chunks')
            shp_geo_list = list(self.divide_chunks(shp_geo_list, chunksize ))
            shp_attr_list = list(self.divide_chunks(shp_attr_list, chunksize))

            for ichunk in range(len(shp_geo_list)):

                shp_geo = shp_geo_list[ichunk]
                shp_attr = shp_attr_list[ichunk]

                # intersect the shp with the met grid
                # returns a weights dictionary
                weights = self._met_grid.intersect(shp_geo, zones=shp_attr)

                # process the downloaded data
                climate_dict = self._process(weights, year_filter=year_filter,
                                 multiprocessing=multiprocessing, cpu_count=cpu_count)
                if save_to_csv:
                    self.clim_to_csv(climate_dict)
                # else:
                #     return climate_dict # doesn't work

        else:

            # intersect the shp with the met grid
            # returns a weights dictionary
            weights = self._met_grid.intersect(shp_geo_list, zones=shp_attr_list)

            # process the downloaded data
            climate_dict = self._process(weights, year_filter=year_filter,
                                         multiprocessing=multiprocessing,
                                         cpu_count=cpu_count)

            if save_to_csv:
                self.clim_to_csv(climate_dict)
            else:
                return climate_dict

    def clim_to_csv(self, climate_dict):

        for item in climate_dict.items():
            df = climate_dict[item[0]]
            for col in df.columns:
                ws = self.out_folder
                fn = '{}_{}.csv'.format(item[0], col)
                fn = os.path.join(ws, fn)
                df_ = pd.DataFrame(climate_dict[item[0]][col])
                df_.to_csv(fn)

    def _downloader(self, climate_filter=None):

        pd.options.display.float_format = '{:,.10f}'.format
        # set up url and paramters
        opendap_url = 'http://thredds.northwestknowledge.net:8080/thredds/dodsC'
        # elev_nc = '{}/{}'.format(
        #     opendap_url, '/MET/elev/metdata_elevationdata.nc#fillmismatch')

        # configure climate filter
        if climate_filter is not None:
            if isinstance(climate_filter, str):
                climate_filter = [climate_filter]
            elif isinstance(climate_filter, list):
                pass
            else:
                raise Exception('climate_filter is not str or list of str')

            for name in climate_filter:
                if name not in self._params:
                    name = name.lower()
                    raise Exception('Name {} is a valid GridMET climate '
                                    'variable'.format(name))
        else:
            climate_filter = self._params.keys()

        # check whats in the climate dictionary
        # add whats needed
        missing_met = [met for met in climate_filter if met not in
                         self._met_dict]

        for met_name in missing_met:
            logger.info('downloading data for {}'.format(met_name))
            if self._verbose:
                print('downloading data for ', met_name)
            # Pulling the full time series then filtering later seems faster than selecting here
            met_nc = '{}/{}.nc#fillmismatch'.format(opendap_url, self._params[met_name]['nc'])
            # fname = '{}.nc'.format(met_name)
            ds = xr.open_dataset(met_nc)
            # print(ds)
            # print('saving to netcdf')
            #     mode = 'w'
            #     if os.path.exists(fname):
            #         mode = 'a'
            #     ds.to_netcdf('{}.nc'.format(met_name), mode=mode)
            # print('1 year download time', time.time() - t1)
            self._met_dict[met_name] = ds

    #
    def _process(self, weights, year_filter=None, multiprocessing=False, cpu_count=None):
        # logger.info('Processing climate data')
        # if self._verbose:
        #     print('print processing climate data...')

        start_date, end_date = self._build_dates(year_filter)
        # convert weight dict to df
        weight_df = pd.DataFrame(weights)

        # get the gridmet cells that have been intersected
        # fid_time = time.time()
        fids = list(weight_df.index.values)
        # rows = []
        # cols = []
        fid_dict = {}
        for fid in fids:
            row, col = self._get_rowcol(fid, 1386) # number of cols in gridmet
            # fid_dict[(row, col)] = fid
            fid_dict[fid] = row, col
            # rows.append(row)
            # cols.append(col)
        # print(fid_dict)
        # print('fid time', time.time() - fid_time)
        climate_dict = {}
        for met_name in self._met_dict:

            # if self._verbose:
            #     print('Processing ', met_name)
            # logger.info('Processing {}'.format(met_name))

            data = self._process_cells(met_name, fid_dict, start_date, end_date,
                                     multiprocessing, cpu_count)

            climate_df = self._post_proc(data, weight_df)
            climate_dict[met_name] = climate_df

        return climate_dict

    def _process_cells(self, met_name, fid_dict, start_date, end_date,
                     multiprocessing, cpu_count):
        data_input = [
            [self._params[met_name]['var'], fid,
             self._met_dict[
                 met_name], fid_dict[fid][0], fid_dict[fid][1], start_date, end_date] for fid in fid_dict]

        if multiprocessing:
            # logger.info('Multiprocessing')
            # mp_proc_time = time.time()
            if cpu_count is None:
                cpu_count = mp.cpu_count()
            pool = mp.Pool(processes=cpu_count)
            data = pool.map(self._to_df, data_input)
            pool.close()
            pool.join()
            # print('mp_proc_time', time.time() - mp_proc_time)
        else:
            # normal_proc_time = time.time()
            data = []
            for in_data in data_input:
                fid, df = self._to_df(in_data)
                if df is not None:
                    data.append(self._to_df(in_data))
                    data.append((fid, df))

        return data

    def _post_proc(self, data, weight_df):
        # post_start = time.time()
        if self._verbose:
            print('post processing...')
        # get the area names
        area_keys = list(weight_df.columns.values)
        # c1 = time.time()
        dfs = []
        for fid, df in data:
            # loop through the areas
            for area in area_keys:
                # creating new area field climate val * frac area for fid and area
                weight = weight_df.loc[fid, area]
                df['{}_val'.format(area)] = df['VAL'] * weight_df.loc[
                    fid, area]
                df['{}_weight'.format(area)] = np.where(df.VAL.isnull(),
                                                        np.nan,
                                                        weight)
            df.drop('VAL', axis=1, inplace=True)
            dfs.append(df)
        # print('inner loop post proc time', time.time() - c1)
        # dfs = data.keys
        d = reduce(lambda x, y: x.add(y, fill_value=0), dfs)
        for area in area_keys:
            val_col = '{}_val'.format(area)
            weight_col = '{}_weight'.format(area)
            d[area] = d[val_col] / d[weight_col]
            d.drop([val_col, weight_col], axis=1, inplace=True)

        # print(d)
        # print('post proc time ', time.time() - post_start)
        return d

    # @staticmethod
    # def _check_retry(data):
    #     retry_fids = [item[0] for item in data if item[1] is None]
    #     # if there are failed cells redownload climate and retry
    #     return retry_fids

    @staticmethod
    def _build_dates(year_filter):
        # sets up year filter
        if year_filter:
            selection = set()
            invalid = set()
            # tokens are comma separated values
            tokens = [x.strip() for x in year_filter.split(',')]
            for i in tokens:
                try:
                    # typically tokens are plain old integers
                    selection.add(int(i))
                except:
                    # if not, then it might be a range
                    try:
                        token = [int(k.strip()) for k in i.split('-')]
                        if len(token) > 1:
                            token.sort()
                            # we have items seperated by a dash
                            # try to build a valid range
                            first = token[0]
                            last = token[len(token) - 1]
                            for x in range(first, last + 1):
                                selection.add(x)
                    except:
                        invalid.add(i)

            year_list = sorted(list(selection))
            date_list = pd.date_range(
                dt.datetime.strptime('{}-01-01'.format(min(year_list)),
                                     '%Y-%m-%d'),
                dt.datetime.strptime('{}-12-31'.format(max(year_list)),
                                     '%Y-%m-%d'))
        else:
            # Create List of all dates
            # determine end date of data collection
            current_date = dt.datetime.today()

            end_date = dt.date(current_date.year, current_date.month,
                               current_date.day) - dt.timedelta(days=1)
            date_list = pd.date_range(dt.datetime.strptime('1979-01-01',
                                                           '%Y-%m-%d'),
                                      end_date)

        start_date = min(date_list)
        end_date = max(date_list)

        return start_date, end_date

    @staticmethod
    def _get_rowcol(index, numofcols):
        row = math.floor(index / numofcols)
        col = index % numofcols
        return row, col

    @staticmethod
    def _to_df(in_data):

        met_name, fid, ds, row, col, start_date, end_date = in_data
        out_df = None
        try_num = 1
        while out_df is None:
            if try_num > 300:
                logger.error('Max Retry for fid {}'.format(fid))
                raise Exception('MAX RETRY')

            try:

                selection = ds.sel(day=slice(start_date, end_date)).isel({'lat':
                    row, 'lon': col}).drop(['crs', 'lat', 'lon']).rename({met_name:'VAL'})
                out_df = selection.to_dataframe()
            except Exception as e:
                print(e)
                out_df = None
                logger.error('To DataFrame Export Error: '
                             '{}, FID:{}, ROW:{}, COL:{}, StartDate:{}, EndDate:'
                             '{}\n'.format(met_name, fid, row,
                            col, start_date, end_date), exc_info=True)
                print('RETRYING {}'.format(try_num))
                logger.info('Retrying export to df FID {}...Retry #{}'.format(fid, try_num))
            try_num += 1

        return fid, out_df


class _Grid(object):

    def __init__(self, verbose=True):

        data_folder = os.path.join(os.path.dirname(__file__), 'data')
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        grid_shp = os.path.join(os.path.dirname(__file__), 'data', 'GridMET.shp')
        index_file = os.path.join(os.path.dirname(__file__), 'data', 'MET_INDEX')
        index_path = index_file + '.idx'

        if not os.path.exists(grid_shp):
            logger.info('creating GridMET Polygons {}'.format(grid_shp))
            if verbose:
                print('creating GridMet Polygons ', grid_shp)
            # start = time.time()
            self.grid_polys = self.netcdf_to_polygons(grid_shp)
            # print('build met shape grid time', time.time() - start)

        else:
            logger.info('loading GridMEt polygons from {}'.format(grid_shp))
            if verbose:
                print('GridMet polygons exist ', grid_shp)
            self.grid_polys = _read_shapefile(grid_shp)[0]
        if not os.path.exists(index_path):
            logger.info('creating spatial index {}'.format(index_file))
            if verbose:
                print('creating spatial index ', index_file)
            self._spatial_index = self.create_spatial_index(index_file)
        else:
            logger.info('loading spatial index from {}'.format(index_file))
            if verbose:
                print('spatial index exists', index_file)
            self._spatial_index = rtree.index.Index(index_file, interleaved=False)

    def create_spatial_index(self, index_file):
        # create spatial index
        index = rtree.index.Index(index_file, interleaved=False)
        for fid in range(0, len(self.grid_polys)):
            poly = self.grid_polys[fid]
            xmin, xmax, ymin, ymax = poly.GetEnvelope()
            index.insert(fid, (xmin, xmax, ymin, ymax))
        return index

    def intersect(self, in_geo, zones=None, ot_shp=None):
        # logger.info('INTERSECTING')
        weight_dict = _DataDict()
        if ot_shp is not None:
            # export to shape
            driver = ogr.GetDriverByName('ESRI Shapefile')
            out_file = driver.CreateDataSource(ot_shp)
            out_layer = out_file.CreateLayer('intersection',
                                             geom_type=ogr.wkbPolygon)
            feature_dfn = out_layer.GetLayerDefn()

        cell_area = None
        for fid2 in range(0, len(in_geo)):
            geo2 = in_geo[fid2]
            if not geo2.IsValid():
                logger.warning("Geo FID: {} not valid".format(fid2))
                continue
            if self._spatial_index is not None:
                xmin, xmax, ymin, ymax = geo2.GetEnvelope()
                for fid1 in list(self._spatial_index.intersection((xmin, xmax, ymin,
                                                      ymax))):

                    geo1 = self.grid_polys[fid1]

                    if cell_area is None:
                        cell_area = geo1.GetArea()

                    if geo2.Intersects(geo1):

                        intersection = geo2.Intersection(geo1)
                        intersection_area = intersection.GetArea()
                        frac_area = intersection_area / cell_area

                        if zones is None:
                            weight_dict.add_val(fid2, fid1, frac_area)
                        else:
                            weight_dict.add_val(zones[fid2], fid1, frac_area)

                        if ot_shp is not None:
                            # export to shape
                            out_feature = ogr.Feature(feature_dfn)
                            out_feature.SetGeometry(intersection)
                            out_layer.CreateFeature(out_feature)
                            out_feature = None

        # export to shape
        if ot_shp is not None:
            out_file = None
        return weight_dict

    @staticmethod
    def netcdf_to_polygons(out_path):
        """
        Converts NetCDF file to gdal/shapely polygon
        """

        # get a netcdf file template
        opendap_url = 'http://thredds.northwestknowledge.net:8080/thredds/dodsC'
        met_nc = '{}/{}.nc#fillmismatch'.format(
            opendap_url, 'agg_met_etr_1979_CurrentYear_CONUS')

        driver = ogr.GetDriverByName('ESRI Shapefile')
        out_file = driver.CreateDataSource(out_path)
        out_layer = out_file.CreateLayer('Grid',
                                         geom_type=ogr.wkbPolygon)
        feature_dfn = out_layer.GetLayerDefn()
        poly = []
        # open netcdf file with xarray
        ds = xr.open_dataset(met_nc)

        # read some data from netcdf file
        # print('\n The meta data is: \n', json.dumps(ds.attrs, indent=4))
        lat_handle = ds['lat']
        lon_handle = ds['lon']

        lon, lat = np.meshgrid(lon_handle, lat_handle)
        res = 0.04166666 / 2.0

        for i in range(np.shape(lon)[0]):
            for j in range(np.shape(lon)[1]):
                wkt_poly = "POLYGON (({0} {1}, {2} {3}, {4} {5}, {6} {7}, " \
                           "{0} {1}))".format(lon[i, j] + res, lat[i, j] - res,
                                              lon[i, j] + res, lat[i, j] + res,
                                              lon[i, j] - res, lat[i, j] + res,
                                              lon[i, j] - res, lat[i, j] - res)

                poly.append(ogr.CreateGeometryFromWkt(wkt_poly))

                out_feature = ogr.Feature(feature_dfn)
                out_feature.SetGeometry(ogr.CreateGeometryFromWkt(wkt_poly))
                out_layer.CreateFeature(out_feature)
                out_feature = None
        out_file = None
        return poly


def _read_shapefile(in_shp, zone_field=None, filter_field=None):

    print('my zone field is', zone_field)

    # start_time = time.time()
    """
    in_shp: shapefile
    returns: list of shapefile polygon ogr objects

    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(in_shp, 0)
    layer = data_source.GetLayer()
    geo_list = []
    attr_list = []

    # field limitter
    limit_field = None
    limit_vals = None
    if filter_field is not None:
        if not isinstance(filter_field, dict):
            raise Exception('Filter field must be dictionary {field: [values to process]')
        if len(filter_field.keys()) > 1:
            raise Exception("Filter field only supports one field")
        limit_field = list(filter_field.keys())[0]
        limit_vals = filter_field[limit_field]
        if not isinstance(limit_vals, list):
            limit_vals = [limit_vals]

    for feature in layer:
        if filter_field is not None:
            filter_val = feature.GetField(limit_field)
            if filter_val in limit_vals:
                process = True
            else:
                process = False
        else:
            process = True
        if process:
            if zone_field is not None:
                zone = feature.GetField(zone_field)
                attr_list.append(zone)
            geo = feature.GetGeometryRef().ExportToWkb()
            out_geo = ogr.CreateGeometryFromWkb(geo)
            geo_list.append(out_geo)
    layer.ResetReading()
    # print('read shapefile time', time.time() - start_time)
    if zone_field is None:
        attr_list = None
    return geo_list, attr_list


def _get_spatial_ref(in_shp):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(in_shp)
    # from Layer
    layer = dataset.GetLayer()
    spatial_ref = layer.GetSpatialRef()
    epsg = int(spatial_ref.GetAttrValue('AUTHORITY', 1))

    return epsg


if __name__ == '__main__':

    pass

    # start_time = time.perf_counter()

    # zoneShpPath = r".\data\test_data\ABCWUA_GridMETPrj.shp"
    # zoneField = 'FieldTest'
    # yearFilter = '1990-1991'
    # climateFilter = 'pr'
    # climateFilter = ['pet', 'etr']
    # climateFilter = 'break'
    # zoneShpPath = sys.argv[1]
    # zoneField = sys.argv[2]
    # yearFilter = sys.argv[3]
    # climateFilter = sys.argv[4].split(',')
    # multiprocessing = bool(sys.argv[5])
    # climateData = get_data(zoneShpPath, zoneField, year_filter=yearFilter,
    #             climate_filter=climateFilter, multiprocessing=multiprocessing)
    # end_time = time.perf_counter()
    # print('run time', end_time - start_time)

    # # print(climateData)
    # dc = DataCollector()
    # dc.
