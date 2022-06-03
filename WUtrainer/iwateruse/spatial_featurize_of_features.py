import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
import pyproj
from pyproj import Proj
from tqdm import tqdm


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

def generate_spatial_feature(df, dist, func, new_field):
    pass

def generate_wu_per_capita_spatial_stat(df, max_points = 50, raduis=500):

    if (not ('X' in df.columns)) | (not ('Y' in df.columns)):
        df = add_XY_meter(df, lat_col='LAT', lon_col='LONG')
    for pop_field in ['pop_swud15', 'plc_pop_interpolated' ]:
        df['dist'] = 0
        df_neutral = df[(df['Ecode_num'] == 0) & (df['wu_rate']>0) & (df[pop_field]>50)]
        df_neutral = df_neutral[~df_neutral['sys_id'].str.contains('TN')]

        sys_ids = df['sys_id'].unique()
        pc_stat_df = []
        for sys_id in tqdm(sys_ids):

            mask_x = df['sys_id'].isin([sys_id])
            x0 = df[mask_x]['X'].values[0]
            y0 = df[mask_x]['Y'].values[0]
            neighbors = get_near_pointsXY(x0 = x0, y0 = y0, df= df_neutral, sys_id_field = 'sys_id', raduis=raduis)

            neighbors = neighbors[:max_points]
            mask_near = df_neutral['sys_id'].isin(neighbors)
            df_near = df_neutral[mask_near]

            pc = df_near['wu_rate']/df_near[pop_field]

            ag_pc = df_near['wu_rate'].sum() / df_near[pop_field].sum()
            pc_mean = pc.mean()
            pc_std = pc.std()
            pc_median = pc.quantile(0.5)
            pc_trim_mean = pc[ (pc <= pc.quantile(0.75)) & (pc >= pc.quantile(0.25))].mean()

            pc_stat_df.append([sys_id, x0, y0, ag_pc, pc_mean, pc_std, pc_median, pc_trim_mean])
        pc_stat_df = pd.DataFrame(pc_stat_df, columns=['sys_id', 'x', 'y', 'ag_pc', 'pc_mean', 'pc_std', 'pc_median', 'pc_trim_mean'])
        pc_stat_df.to_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\spatial_features\pc_50_{}.csv".format(pop_field))
    pass



if __name__ == "__main__":

    train_db = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\wu_annual_training.csv")
    #pop_info = pd.read_csv(r"C:\work\water_use\ml_experiments\annual_v_0_0\pop_info.csv")

    # =======================================
    # generate wu per capita stats
    # =======================================
    generate_wu_per_capita_spatial_stat(train_db, max_points=50, raduis=500)


