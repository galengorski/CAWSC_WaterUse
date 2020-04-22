import sys
import os
import numpy as np
import xarray as xr
import json
import pandas as pd
import time
from osgeo import ogr
import rtree
import datetime as dt
import logging
import math
import multiprocessing as mp


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


class Grid:

    def __init__(self, verbose=True):

        grid_shp = os.path.join('.', 'data', 'GridMET.shp')
        index_file = os.path.join('.', 'data', 'MET_INDEX')
        index_path = index_file + '.idx'

        if not os.path.exists(grid_shp):
            if verbose:
                print('creating GridMet Polygongs ', grid_shp)
            self.grid_polys = netcdf_to_polygons(grid_shp)
        else:
            if verbose:
                print('GridMet polygons exist ', grid_shp)
            self.grid_polys = read_shapefile(grid_shp)[0]
        if not os.path.exists(index_path):
            if verbose:
                print('creating spatial index ', index_file)
            self.spatial_index = self._create_spatial_index(index_file)
        else:
            if verbose:
                print('spatial index exists', index_file)
            self.spatial_index = rtree.index.Index(index_file, interleaved=False)

    def _create_spatial_index(self, index_file):
        # create spatial index
        # startSpatialIndexTime = time.perf_counter()
        index = rtree.index.Index(index_file, interleaved=False)
        for fid in range(0, len(self.grid_polys)):
            poly = self.grid_polys[fid]
            xmin, xmax, ymin, ymax = poly.GetEnvelope()
            index.insert(fid, (xmin, xmax, ymin, ymax))
        # index.close()
        # self.spatial_index = index
        # endSpatialIndexTime = time.perf_counter()
        # totalSpatialIndexTime = endSpatialIndexTime - startSpatialIndexTime
        # print('Spatial index time ... ', totalSpatialIndexTime)

        # print(index)
        return index

    def intersect(self, in_geo, zones=None, ot_shp=None):
        # print(self.spatial_index.bounds)
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
            if self.spatial_index is not None:
                xmin, xmax, ymin, ymax = geo2.GetEnvelope()
                for fid1 in list(self.spatial_index.intersection((xmin, xmax, ymin,
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


def _parse_int_set(nputstr=""):
    selection = set()
    invalid = set()
    # tokens are comma separated values
    tokens = [x.strip() for x in nputstr.split(',')]
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
    return selection


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
    # out_feature = ogr.Feature(feature_dfn)

    start_time = time.perf_counter()
    poly = []
    # open netcdf file with xarray
    ds = xr.open_dataset(met_nc)

    # read some data from netcdf file
    # print('\n The meta data is: \n', json.dumps(ds.attrs, indent=4))
    lat_handle = ds['lat']
    lon_handle = ds['lon']

    lon, lat = np.meshgrid(lon_handle, lat_handle)
    res = 0.04166666 / 2.0

    # count = 0
    # print('Buidling nc polygons...')
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
    # end_time = time.perf_counter()
    # total_time = end_time - start_time
    # print('NC build time seconds... ', total_time)
    return poly


def read_shapefile(in_shp, zone_field=None):
    """
    in_shp: shapefile
    returns: list of shapefile polygon ogr objects

    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.Open(in_shp, 0)
    layer = data_source.GetLayer()
    geo_list = []
    attr_list = []
    for feature in layer:
        if zone_field is not None:
            field = feature.GetField(zone_field)
            attr_list.append(field)
        geo = feature.GetGeometryRef().ExportToWkb()
        out_geo = ogr.CreateGeometryFromWkb(geo)
        geo_list.append(out_geo)
    layer.ResetReading()
    return geo_list, attr_list


def get_rowcol(index, numofcols):
    row = math.floor(index/numofcols)
    col = index % numofcols
    return row, col


def downloader(weight_dict, year_filter=None, climate_filter=None,
               multiprocessing=False, verbose=True):
    # total_pull_time = 0
    # test pulling down girdmet data
    pd.options.display.float_format = '{:,.10f}'.format
    # set a year filter start year to end year
    # year_filter = ''
    # Specify column order for output .csv Variables:
    # set up url and paramters
    opendap_url = 'http://thredds.northwestknowledge.net:8080/thredds/dodsC'
    # elev_nc = '{}/{}'.format(
    #     opendap_url, '/MET/elev/metdata_elevationdata.nc#fillmismatch')
    params = {
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

    # sets up year filter
    if year_filter:
        year_list = sorted(list(_parse_int_set(year_filter)))
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
                           current_date.day - 1)
        date_list = pd.date_range(dt.datetime.strptime('1979-01-01',
                                                       '%Y-%m-%d'), end_date)

    start_date = min(date_list)
    end_date = max(date_list)

    met_dict = {}
    if climate_filter is not None:
        if isinstance(climate_filter, str):
            climate_filter = [climate_filter]
        elif isinstance(climate_filter, list):
            pass
        else:
             raise Exception('climate_filter is not str or list of str')

    for name in climate_filter:
        if name not in params:
            name = name.lower()
            raise Exception('Name {} is a valid GridMET climate '
                            'variable'.format(name))

    if verbose:
        print('Downloading climate data ....')
    for met_name in climate_filter:
        startPullTime = time.perf_counter()
        logging.debug('  Variable: {}'.format(met_name))
        # Pulling the full time series then filtering later seems faster than selecting here
        met_nc = '{}/{}.nc#fillmismatch'.format(
            opendap_url, params[met_name]['nc']
        )
        ds = xr.open_dataset(met_nc).sel(day=slice(start_date, end_date))

        # print('\n The meta data is: \n', json.dumps(ds.attrs, indent=4))
        # lathandle = ds['lat']
        # lonhandle = ds['lon']
        # timehandle = ds['day']
        # datahandle=ds['air_temperature'] # for non aggragated download
        datahandle = ds[params[met_name]['var']]  # for aggragated download
        # crshandle=ds['crs']
        # print('\n The crs meta data is \n', json.dumps(crshandle.attrs, indent=4))
        # crstransform = crshandle.attrs['GeoTransform']
        # print(crstransform)

        # collect data to describe geotransform
        # lonmin = float(ds.attrs['geospatial_lon_min'])
        # latmax = float(ds.attrs['geospatial_lat_max'])
        # lonres = float(ds.attrs['geospatial_lon_resolution'])
        # latres = float(ds.attrs['geospatial_lon_resolution'])

        # Print some information on the data

        # print('\n Data attributes, sizes, and coords \n')
        # print('\n Data attributes are: \n', json.dumps(datahandle.attrs, indent=4))
        # print('\n Data sizes are: \n', datahandle.sizes)
        # print('\n Data coords are: \n', datahandle.coords)

        ts = datahandle.sizes
        dayshape = ts['day']
        Lonshape = ts['lon']
        Latshape = ts['lat']
        # print(dayshape, Lonshape, Latshape)

        met_dict[met_name] = ds

        # endPullTime = time.perf_counter()
        # pullTime = endPullTime - startPullTime
        # print('Pull time ... ', pullTime)
        # total_pull_time += pullTime
    # print('total pull time', total_pull_time)

    # convert weight dict to df
    weight_df = pd.DataFrame(weight_dict)
    # weight_df.to_csv('test_weights.csv')

    fids = list(weight_df.index.values)
    rows = []
    cols = []
    fid_dict = {}
    for fid in fids:
        row, col = get_rowcol(fid, 1386)
        fid_dict[(row, col)] = fid
        rows.append(row)
        cols.append(col)

    if verbose:
        print('print processing climate data...')
    climate_dict = {}
    for met_name in met_dict:
        if verbose:
            print('Processing ', met_name)
        start_testing_to_dataframe = time.perf_counter()
        start_load_time = time.perf_counter()
        data_input = [[params[met_name]['var'], fid_dict[(row, col)], met_dict[
            met_name], row, col] for row, col in zip(rows,cols)]
        # data_input = [[met_dict['etr'], day] for day in range(dayshape)]
        # for row, col in zip(rows, cols):
        # print('pooling')
        if multiprocessing:
            pool = mp.Pool(processes=mp.cpu_count())
            data = pool.map(to_df, data_input)
            pool.close()
            pool.join()
        else:
            data = []
            for in_data in data_input:
                data.append(to_df(in_data))

        # end_load_time = time.perf_counter()
        # total_load_time = end_load_time - start_load_time
        # print('total tload time', total_load_time)
        # start_post_time = time.perf_counter()
        out_dict = {area_name: {} for area_name in weight_dict.keys()}
        area_dict = {area_name: {} for area_name in weight_dict.keys()}
        for fid, df in data:
            for area in weight_dict.keys():
                df[area] = df['VAL'] * weight_df.loc[fid, area]
                for day in df.index.to_list():
                    if day not in out_dict[area]:
                        out_dict[area][day] = np.nan
                        area_dict[area][day] = 0.0
                    if not pd.isna(df.loc[day, area]):
                        if np.isnan(out_dict[area][day]):
                            out_dict[area][day] = df.loc[day, area]
                            area_dict[area][day] += weight_dict[area][fid]
                        else:
                            out_dict[area][day] += df.loc[day, area]
                            area_dict[area][day] += weight_dict[area][fid]

        # end_post_time = time.perf_counter()
        for area, day_dict in out_dict.items():
            for day in day_dict:
                out_dict[area][day] /= float(area_dict[area][day])

        out_df = pd.DataFrame(out_dict)

        # print(out_df)
        # output_start_time = time.perf_counter()
        # # out_df.to_csv('testing_final_output.csv')
        # output_end_time = time.perf_counter()
        # total_output_time = output_end_time - output_start_time
        # print('output time', total_output_time)

        climate_dict[met_name] = out_df

        # end_testing_to_dataframe = time.perf_counter()
        # testing_to_dataframe = end_testing_to_dataframe - start_testing_to_dataframe
        # print(testing_to_dataframe)
    # print(out_dict)

    return climate_dict


def to_df(in_data):
    met_name, fid, ds, row, col = in_data
    return fid, ds.isel({'lat': row, 'lon': col}).drop(['crs', 'lat',
        'lon']).rename({met_name: 'VAL'}).to_dataframe()


def get_data(shp, zone_field, year_filter=None, climate_filter=None,
             multiprocessing=False, verbose=True):
    # print('getting grid')
    grid = Grid()
    # print('getting shp')
    shp_geo, shp_attr = read_shapefile(shp, zone_field)
    # print('making weights')
    weights = grid.intersect(shp_geo, zones=shp_attr,
        ot_shp=os.path.join('.', 'Data', 'test_intersect.shp'))
    data_dict = downloader(weights, year_filter=year_filter,
            climate_filter=climate_filter, multiprocessing=multiprocessing,
                           verbose=verbose)

    return data_dict


if __name__ == '__main__':

    # start_time = time.perf_counter()

    # zoneShpPath = r".\data\test_data\ABCWUA_GridMETPrj.shp"
    # zoneField = 'FieldTest'
    # yearFilter = '1990-1991'
    # climateFilter = 'pr'
    # climateFilter = ['pet', 'etr']
    # climateFilter = 'break'
    zoneShpPath = sys.argv[1]
    zoneField = sys.argv[2]
    yearFilter = sys.argv[3]
    climateFilter = sys.argv[4].split(',')
    multiprocessing = bool(sys.argv[5])
    climateData = get_data(zoneShpPath, zoneField, year_filter=yearFilter,
                climate_filter=climateFilter, multiprocessing=multiprocessing)
    # end_time = time.perf_counter()
    # print('run time', end_time - start_time)

    # print(climateData)

