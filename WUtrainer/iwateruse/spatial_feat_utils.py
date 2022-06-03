import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
import pyproj
from pyproj import Proj
from tqdm import tqdm
from scipy import stats
from joblib import Parallel, delayed


def latlon_to_xy_m(lat, lon):
    x, y = pyproj.transform(Proj(init='epsg:4326'), Proj(init='epsg:5070'), lon, lat)
    return x,y


def add_XY_meter(df, lat_col = 'LAT', lon_col = 'LONG'):
    """
    Add X and Y coordinate in meter to df

    """

    lat = df['LAT'].values
    long = df['LONG'].values
    xx, yy = latlon_to_xy_m(lat, long)
    df['X'] = xx
    df['Y'] = yy
    return df

def get_near_points(sys_id, df, sys_id_field, raduis):
    """

    :param sys_id: sys_id for which analysis is made
    :param df: dataframe with data
    :param sys_id_field: sys_id field in df
    :param raduis: analysis raduis in KM

    We assume that coordinate X and Y exist in the df. X and Y in meter (epsg : 5070)

    :return:
    """
    df = df.copy()
    mask = df[sys_id_field] == sys_id
    x0 = df.loc[mask, 'X'].values[0]
    y0 = df.loc[mask, 'Y'].values[0]
    dx = np.power(df['X'] - x0, 2.0)
    dy = np.power(df['Y'] - y0, 2.0)
    dist = np.power((dx+dy), 0.5)
    df.loc[:,'dist'] = dist.values
    mask = dist < (raduis * 1000)
    df_near = df[mask]
    df_near = df_near.sort_values('dist')
    df_near.drop_duplicates(subset=['sys_id'], keep='first', inplace = True)
    near_sys = df_near['sys_id'].values

    return near_sys

def get_near_pointsXY(x0, y0, df, sys_id_field, raduis):
    """

    :param sys_id: sys_id for which analysis is made
    :param df: dataframe with data
    :param sys_id_field: sys_id field in df
    :param raduis: analysis raduis in KM

    We assume that coordinate X and Y exist in the df. X and Y in meter (epsg : 5070)

    :return:
    """
    df = df.copy()
    df['dist'] = 0
    dx = np.power(df['X'] - x0, 2.0)
    dy = np.power(df['Y'] - y0, 2.0)
    dist = np.power((dx+dy), 0.5)
    df.loc[:,'dist'] = dist.values
    mask = dist < (raduis * 1000)
    df_near = df[mask]
    df_near = df_near.sort_values('dist')
    df_near.drop_duplicates(subset=['sys_id'], keep='first', inplace = True)
    near_sys = df_near['sys_id'].values

    return near_sys


def generate_local_statistics(df, sys_id_col = 'sys_id', max_points = 50, raduis=500):

    max_pc = 500
    min_pc = 20

    if (not ('X' in df.columns)) | (not ('Y' in df.columns)):
        df = add_XY_meter(df, lat_col='LAT', lon_col='LONG')

    sys_ids = df[sys_id_col].unique()
    pc_stat_df = []
    df_ = df.copy()
    east = 2e5

    def compute_states(sys_id, sys_id_col, df_ ):
        mask_x = df_[sys_id_col].isin([sys_id])
        x0 = df_[mask_x]['X'].values[0]
        y0 = df_[mask_x]['Y'].values[0]
        vv = [sys_id, x0, y0]
        if x0<east:
            max_pp = max_points * 2
        else:
            max_pp = max_points
        for feat in ['pc_pop', 'pc_swud', 'pc_tpopsrv']:
            mask = (df_[feat]>=min_pc) & (df_[feat]<=max_pc)
            neighbors = get_near_pointsXY(x0=x0, y0=y0, df=df_[mask], sys_id_field=sys_id_col, raduis=raduis)
            neighbors = neighbors[:max_pp]
            mask_near = df_[mask][sys_id_col].isin(neighbors)
            df_near = df_[mask][mask_near]
            var = df_near[feat].dropna()
            pc_trim_mean = var[(var >= var.quantile(0.2)) & (var <= var.quantile(0.8))]
            pc_median = var.quantile(0.5)
            vv = vv + [np.mean(pc_trim_mean), pc_median]
        return vv

    distributed = True
    if distributed:
        pc_stat_df = Parallel(n_jobs=10)(delayed(compute_states)(sys_id, sys_id_col, df_) for sys_id in tqdm(sys_ids))
    else:
        for sys_id in tqdm(sys_ids):
            res = compute_states(sys_id, sys_id_col, df_)
            pc_stat_df.append(res)

    pc_stat_df = pd.DataFrame(pc_stat_df, columns=['sys_id', 'x', 'y', 'pop_tmean', 'pop_median',
                                                   'swud_tmean', 'swud_median', 'tpop_tmean', 'tpop_median' ])
    return pc_stat_df


