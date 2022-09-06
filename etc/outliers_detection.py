import os, sys
import numpy as np
import pandas as pd


def filter1(X, y):

    """

    :param X: original dataframe
    :param y: original tragtes
    :return: X_0, y_0: dropped data
             X_, y_: remaining data
    """

    return X_0, y_0, X_, y_


def drop_abnormal_percapita_wu(X, y, thresh_max, thresh_min):

    drop(X, y)

    return X, y


def get_monthly_frac(df, var_col="", date_col="Date", month_col=None):
    """

    :param df: pandas dataframe with data to calculate average monthly fractions
    :param var_col: the field name of the variable
    :param date_col: date column name
    :param month_col: optional
    :return:
    """

    if not (month_col is None):
        mon_mean = df.groupby(by=[month_col]).mean()
        mon_frac = mon_mean[var_col] / mon_mean[var_col].sum()
        return (mon_mean, mon_frac)
    temp_date = date_col + "_temp"
    temp_mon = date_col + "_mon"
    df[temp_date] = pd.to_datetime(df[date_col])
    df[temp_mon] = df[temp_date].dt.month

    mon_mean = df.groupby(by=[temp_mon]).mean()
    mon_frac = mon_mean[var_col] / mon_mean[var_col].sum()
    return (mon_mean, mon_frac)


def sign_correlation(x, y):
    """
    Find sign correlation
    :param x:
    :param y:
    :return:
    """
    xdf = np.diff(x)
    ydf = np.diff(y)
    xsign = np.zeros_like(xdf) + np.NAN
    ysign = np.zeros_like(ydf) + np.NAN

    xsign[xdf >= 0] = 1.0
    xsign[xdf < 0] = -1.0
    ysign[ydf >= 0] = 1.0
    ysign[ydf < 0] = -1.0

    corr = np.nansum(xsign * ysign) / len(xsign)

    return corr


def is_similair(x, y):
    """

    :param x:
    :param y:
    :return:
        return sign_similarity: if close to one means that x and y linearly correlated
        magnitude_similarity

    """

    xis_increasing = np.diff(x) > 0
    yis_increasing = np.diff(y) > 0

    sign_similarity = np.sum(xis_increasing == yis_increasing) / len(x)

    Xrel_change = np.diff(x) / np.sum(x)
    Yrel_change = np.diff(y) / np.sum(y)

    magnitude_similarity = np.mean(Xrel_change / Yrel_change)
    return sign_similarity


def flag_monthly_wu_abnormal_fac(df, sys_id, year, month, wu):

    temp_frac = """ 1     0.080058
                    2     0.080429
                    3     0.081725
                    4     0.083094
                    5     0.084614
                    6     0.085946
                    7     0.086592
                    8     0.086366
                    9     0.085263
                    10    0.083500
                    11    0.081833
                    12    0.080581"""

    mon_frac = temp_frac.strip().split("\n")
    for im, m in enumerate(mon_frac):
        mon, fr = m.strip().split()
        mon_frac[im] = [int(mon), float(fr)]

    df["frac"] = df[" monthly_wu_G"] / df["annual_wu_G"]
    mon_frac = pd.DataFrame(mon_frac, columns=["Month", "standard_fraction"])
    df = df.merge(mon_frac, on="Month", how="left")

    def seasonality_test(_df):
        # ddf = _df[[month, 'frac']]
        _df.sort_values(by=month, inplace=True)
        # cor = sign_correlation(_df['frac'].values, _df['standard_fraction'].values)
        cor = is_similair(_df["frac"].values, _df["standard_fraction"].values)
        return cor

    df2 = df.groupby(by=[sys_id, year]).apply(seasonality_test)
    x = 1


if 0:
    x = np.array([1, 5, 7, 9, 2])
    y = np.array([2, 6, 9, 15, 3])
    sign_correlation(x, y)

    swud = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\SWUDS_v17.csv",
        encoding="cp1252",
    )

    monthly_swud = pd.read_csv(
        r"C:\work\water_use\recycle_bin\monthly_swud.csv"
    )
    flag_monthly_wu_abnormal_fac(
        monthly_swud,
        sys_id="WSA_AGIDF",
        year="YEAR",
        month="Month",
        wu="monthly_wu_G",
    )

    df = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\features\monthly_climate.csv"
    )
    x = 1
