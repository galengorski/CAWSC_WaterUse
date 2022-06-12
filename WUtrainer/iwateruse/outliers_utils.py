import os, sys
import pandas as pd
import numpy as np

import os, sys
import numpy as np
import pandas as pd
import operator


def outliers_func(model, args=None):
    df = model.df_train
    df = df.dropna(axis=0)
    df = df[df['pop'] > 100]
    mask = (df[model.target] >= 20) & (df[model.target] <= 1000)
    df = df[mask]

    # drop na
    df.dropna(axis=0, inplace=True)

    model.df_train = df


def drop_values(model, **kwargs):
    """

    :param model:
    :param kwargs:
    :return:
    """
    ops_dict = {'>': operator.gt,
                '<': operator.lt,
                '>=': operator.ge,
                '<=': operator.le,
                '==': operator.eq}

    def parse(op):
        op_order = ['>=', '<=', '>', '<', '==']
        for o in op_order:
            if o in ['=>', '=<']:
                raise ValueError("Unknown operator")

            if o in op:
                col, cut = op.split(o)
                return col, o, cut

    opts = kwargs['opts']

    df = model.df_train.copy()
    for opt in opts:
        op = opt.strip()
        col, rel, cut = parse(op)
        mask = ops_dict[rel](df[col], float(cut))
        df = df[~mask]

    model.df_train = df
    model.log.info("Outliers are removed based on this role : {}".format(opts))
    model.log.info("Data after data removed is")
    model.log.to_table(model.df_train)

def drop_na_target(model):

    df = model.df_train.copy()
    target = df[model.target]
    df = df[~target.isna()]
    model.df_train = df

    model.log.info("Drop any NaN target.. ")
    model.log.info("Data after data removed is")
    model.log.to_table(model.df_train)



def filter1(X, y):
    """

    :param X: original dataframe
    :param y: original tragtes
    :return: X_0, y_0: dropped data
             X_, y_: remaining data
    """

    return 1


def drop_abnormal_percapita_wu(X, y, thresh_max, thresh_min):
    drop(X, y)

    return X, y


def get_monthly_frac(df, var_col='', date_col='Date', month_col=None):
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


def is_similair(x_, y_):
    """

    :param x:
    :param y:
    :return:
        return sign_similarity: if close to one means that x and y linearly correlated
        magnitude_similarity

    """
    x = x_.copy()
    y = y_.copy()
    mask_na = np.logical_not(np.isnan(x))
    x = x[mask_na]
    y = y[mask_na]

    if len(x) <= 1:
        return 0

    xis_increasing = (np.diff(x) > 0)
    yis_increasing = np.diff(y) > 0

    sign_similarity = np.sum(xis_increasing == yis_increasing) / len(x)

    Xrel_change = np.diff(x) / np.sum(x)
    Yrel_change = np.diff(y) / np.sum(y)

    magnitude_similarity = np.mean(Xrel_change / Yrel_change)
    return sign_similarity


def flag_monthly_wu_abnormal_fac(df, sys_id, year, month, mon_wu, ann_wu):
    """
    Return a value between (0,1) that measures the similarity of
    monthly fractions to temperature averages.
    :param df:
    :param sys_id:
    :param year:
    :param month:
    :param wu:
    :return: return df with a new column "monfrac_flg"
    """
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

    df['frac'] = df[mon_wu] / df[ann_wu]
    mon_frac = pd.DataFrame(mon_frac, columns=[month, 'standard_fraction'])
    df = df.merge(mon_frac, on=month, how='left')

    def seasonality_test(_df):
        _df.sort_values(by=month, inplace=True)
        cor = is_similair(_df['frac'].values, _df['standard_fraction'].values)
        return cor

    df2 = df.groupby(by=[sys_id, year]).apply(seasonality_test)
    df2 = pd.DataFrame(df2).reset_index()
    df = df.merge(df2, how='left', on=[sys_id, year])
    df.drop(columns=['frac', 'standard_fraction'], inplace=True)
    df.rename(columns={0: 'seasonality_simil'}, inplace=True)

    return df
