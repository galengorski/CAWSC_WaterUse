import numpy as np
import pandas as pd


def spatial_detect(df, radius, funcs):
    """
    Driver method for spatial outlier detection. Method
    uses distance formula to determine if records are within a
    certain distance. The user passes outlier detection function(s).

    Parameters
    ----------
    df : pd.DataFrame
    radius : float
        radius in meters to consider data from for each WSA
    funcs : list [object,]
        python function(s) for outlier detection

    """
    if not isinstance(funcs, list):
        funcs = [funcs]

    xc = df.x_centroid.values
    yc = df.y_centroid.values
    agidfs = df.wsa_agidf.values

    for ix, agidf in enumerate(agidfs):
        if (ix % 250) == 0:
            print("Percent done: {:.3f}".format(ix/len(df)))
        a2 = np.power(xc - xc[ix], 2)
        b2 = np.power(yc - yc[ix], 2)
        dist = np.sqrt(a2 + b2)
        idx = np.where(dist <= radius)[0]
        neighbors = agidfs[idx]
        for fun in funcs:
            df = fun(df, agidf, neighbors, 1)

    return df


def mean_stdev(df, agidf, neighbors, iteration=1):
    """
    Method to take the a mean and standard deviation of a group of
    water service areas and flag the water service area of interest
    if it's data is outside of the first moment of the distribution.

    Parameters
    ----------
    df : pd.DataFrame
    agidf : str
        water service area agidf of interest
    neighbors : list [str,]
        "neighborhood" water service area agidfs
    iter : int
        iteration number

    Returns:
    -------
        df
    """
    mbase = "mean_{}".format(iteration)
    olbase = "std_flg_{}".format(iteration)
    if iteration == 1:
        key = 'wu_pp_gd'
    else:
        key = "mean_{}".format(iteration - 1)

    if mbase not in list(df):
        df[mbase] = np.zeros((len(df),)) * np.nan
        df[olbase] = np.zeros((len(df),))

    tdf = df[df.wsa_agidf.isin(neighbors)]
    tdf = tdf[tdf.wsa_agidf != agidf]
    val = df.loc[df["wsa_agidf"] == agidf, key].values[0]

    if len(tdf) >= 1:
        mean = tdf[key].mean(skipna=True)
        if np.isinf(mean):
            print('break')
        std = tdf[key].std()
        std2 = 2 * std
        std3 = 3 * std
        std_half = 0.5 * std
        std_qrt = 0.25 * std
        std_ei = 0.125 * std
    else:
        mean = val
        std = np.nan
        std2 = np.nan
        std3 = np.nan
        std_half = np.nan
        std_qrt = np.nan
        std_ei = np.nan

    if (mean + std3) < val or val < (mean - std3):
        flg = 3
    elif (mean + std2) < val or val < (mean - std2):
        flg = 2
    elif (mean + std) < val or val < (mean - std):
        flg = 1
    elif (mean + std_half) < val or val < (mean - std_half):
        flg = 0.50
    elif (mean + std_qrt) < val or val < (mean - std_half):
        flg = 0.25
    elif (mean + std_qrt) < val or val < (mean - std_half):
        flg = 0.125
    else:
        flg = 0

    if val == 0:
        flg = -1

    df.loc[df.wsa_agidf == agidf, mbase] = mean
    df.loc[df.wsa_agidf == agidf, olbase] = flg
    return df
