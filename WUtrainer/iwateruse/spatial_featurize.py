import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
import pyproj
from pyproj import Proj
from tqdm import tqdm


def latlon_to_xy_m(lat, lon):
    x, y = pyproj.transform(
        Proj(init="epsg:4326"), Proj(init="epsg:5070"), lon, lat
    )
    return x, y


def add_XY_meter(df, lat_col="LAT", lon_col="LONG"):
    """
    Add X and Y coordinate in meter to df

    """

    lat = df["LAT"].values
    long = df["LONG"].values
    xx, yy = latlon_to_xy_m(lat, long)
    df["X"] = xx
    df["Y"] = yy
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
    x0 = df.loc[mask, "X"].values[0]
    y0 = df.loc[mask, "Y"].values[0]
    dx = np.power(df["X"] - x0, 2.0)
    dy = np.power(df["Y"] - y0, 2.0)
    dist = np.power((dx + dy), 0.5)
    df.loc[:, "dist"] = dist.values
    mask = dist < (raduis * 1000)
    df_near = df[mask]
    df_near = df_near.sort_values("dist")
    df_near.drop_duplicates(subset=["sys_id"], keep="first", inplace=True)
    near_sys = df_near["sys_id"].values

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

    dx = np.power(df["X"] - x0, 2.0)
    dy = np.power(df["Y"] - y0, 2.0)
    dist = np.power((dx + dy), 0.5)
    df.loc[:, "dist"] = dist.values
    mask = dist < (raduis * 1000)
    df_near = df[mask]
    df_near = df_near.sort_values("dist")
    df_near.drop_duplicates(subset=["sys_id"], keep="first", inplace=True)
    near_sys = df_near["sys_id"].values

    return near_sys


def generate_spatial_feature(df, dist, func, new_field):
    pass


def fix_negative_income(df):
    income_cols = [s for s in df.columns if "income_" in s]
    for feat in income_cols:
        df.loc[df[feat] < 0, feat] = 0

    sum_income = df[income_cols].sum(axis=1)

    for feat in income_cols:
        df["prc_" + feat] = df[feat] / sum_income

    df.loc[df["median_income"] < 20000, "median_income"] = 20000

    # compute mean income
    income_dic = {}
    for feat in income_cols:
        if "_lt_" in feat or "_gt_" in feat:
            v = feat.split("_")[2]
            v = v.upper()
            ave = 1000.0 * int(v.replace("K", ""))

        else:
            vals = feat.split("_")[1:]
            sumv = 0
            for v in vals:
                v = v.upper()
                v = 1000.0 * int(v.replace("K", ""))
                sumv = sumv + v
            ave = sumv / 2.0
        income_dic[feat] = ave
    df["average_income"] = 0
    for feat in income_cols:
        inc = income_dic[feat]
        df["average_income"] = df["average_income"] + inc * df["prc_" + feat]

    return df


def compute_average_house_age(df, ref_year=2022):
    year_dict = {
        "h_age_newer_2005": 2005,
        "h_age_2000_2004": 2002,
        "h_age_1990_1999": 1994.5,
        "h_age_1980_1989": 1984.5,
        "h_age_1970_1979": 1974.5,
        "h_age_1960_1969": 1964.5,
        "h_age_1950_1959": 1954.5,
        "h_age_1940_1949": 1944.5,
        "h_age_older_1939": 1939,
    }
    cols = year_dict.keys()
    df["av_house_age"] = 0
    df["hhsum"] = 0
    for col in cols:
        df["av_house_age"] = df["av_house_age"] + np.abs(df[col]) * (
            ref_year - year_dict[col]
        )
        df["hhsum"] = df["hhsum"] + np.abs(df[col])

    df["av_house_age"] = df["av_house_age"] / df["hhsum"]
    del df["hhsum"]
    return df


def fix_house_negative_age(df):
    "This is the raw train_Df"
    df.loc[df["median_h_year"] < 1930, "median_h_year"] = 1930
    df.loc[df["median_h_year"] > 2010, "median_h_year"] = 2010

    ag_cols = [s for s in df.columns if "h_age_" in s]
    for col in ag_cols:
        df.loc[df[col] < 0, col] = 0
    df = compute_average_house_age(df, ref_year=2022)

    # compute % of houses
    house_age = df[ag_cols]
    df["n_houses"] = house_age.sum(axis=1)
    for feat in ag_cols:
        df["prc_" + feat] = df[feat] / df["n_houses"]

    return df


def generate_wu_per_capita_spatial_stat(
    df, pop_field, max_points=50, raduis=500
):

    if (not ("X" in df.columns)) | (not ("Y" in df.columns)):
        df = add_XY_meter(df, lat_col="LAT", lon_col="LONG")

    df["dist"] = 0
    df_neutral = df[
        (df["Ecode_num"] == 0) & (df["wu_rate"] > 0) & (df[pop_field] > 50)
    ]
    df_neutral = df_neutral[~df_neutral["sys_id"].str.contains("TN")]

    sys_ids = df["sys_id"].unique()
    pc_stat_df = []
    for sys_id in tqdm(sys_ids):

        mask_x = df["sys_id"].isin([sys_id])
        x0 = df[mask_x]["X"].values[0]
        y0 = df[mask_x]["Y"].values[0]
        neighbors = get_near_pointsXY(
            x0=x0, y0=y0, df=df_neutral, sys_id_field="sys_id", raduis=raduis
        )

        neighbors = neighbors[:max_points]
        mask_near = df_neutral["sys_id"].isin(neighbors)
        df_near = df_neutral[mask_near]

        pc = df_near["wu_rate"] / df_near[pop_field]

        ag_pc = df_near["wu_rate"].sum() / df_near[pop_field].sum()
        pc_mean = pc.mean()
        pc_std = pc.std()
        pc_median = pc.quantile(0.5)
        pc_trim_mean = pc[
            (pc <= pc.quantile(0.75)) & (pc >= pc.quantile(0.25))
        ].mean()
        p95 = pc.quantile(0.95)
        p5 = pc.quantile(0.05)

        pc_stat_df.append(
            [
                sys_id,
                x0,
                y0,
                ag_pc,
                pc_mean,
                pc_std,
                pc_median,
                pc_trim_mean,
                p95,
                p5,
            ]
        )
    pc_stat_df = pd.DataFrame(
        pc_stat_df,
        columns=[
            "sys_id",
            "x",
            "y",
            "ag_pc",
            "pc_mean",
            "pc_std",
            "pc_median",
            "pc_trim_mean",
            "p95",
            "p5",
        ],
    )
    pc_stat_df.to_csv(
        r"C:\work\water_use\mldataset\ml\training\misc_features\spatial_features\pc_50_withTNin.csv"
    )

    pass


if __name__ == "__main__":

    train_db = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\wu_annual_training.csv"
    )
    raw_train_db = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\raw_wu_annual_training.csv"
    )

    if 0:
        raw_train_db = fix_house_negative_age(raw_train_db)
        raw_train_db = fix_negative_income(raw_train_db)
        raw_train_db.to_csv(
            r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\raw_wu_annual_training.csv"
        )
    # =======================================
    # generate wu per capita stats
    # =======================================
    train_db.loc[train_db["median_h_year"] < 1900, "median_h_year"] = 1900
    train_db.loc[train_db["median_h_year"] > 2010, "median_h_year"] = 2010

    generate_wu_per_capita_spatial_stat(
        train_db, pop_field="swud_pop", max_points=50, raduis=500
    )
