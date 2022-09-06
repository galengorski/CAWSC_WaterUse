import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def baseflow_low_pass_filter(arr, beta=0.9, T=1, enforce=False):
    """
    User method to apply digital baseflow separation filter
    (Lyne & Hollick, 1979; Nathan & Mcmahon, 1990;
    Boughton, 1993; Chapman & Maxwell, 1996).

    Method removes "spikes" which would be consistent with storm flow
    events from data.

    Parameters
    ----------
    arr : np.array
        streamflow or municipal pumping time series

    beta : float
        baseflow filtering parameter that ranges from 0 - 1
        values in 0.8 - 0.95 range used in literature for
        streamflow baseflow separation

    T : int
        number of recursive filtering passes to apply to the data

    enforce : bool
        enforce physical constraint of baseflow less than measured flow

    Returns
    -------
        np.array of baseflow
    """
    for _ in range(T):
        arr = _baseflow_low_pass_filter(arr, beta, enforce)

    return arr


def _baseflow_low_pass_filter(arr, beta, enforce):
    """
    Private method to apply digital baseflow separation filter
    (Lyne & Hollick, 1979; Nathan & Mcmahon, 1990;
    Boughton, 1993; Chapman & Maxwell, 1996).

    This method should not be called by the user!

    Method removes "spikes" which would be consistent with storm flow
    events from data.

    Parameters
    ----------
    arr : np.array
        streamflow or municipal pumping time series

    beta : float
        baseflow filtering parameter that ranges from 0 - 1
        values in 0.8 - 0.95 range used in literature for
        streamflow baseflow separation

    enforce : bool
        enforce physical constraint of baseflow less than measured flow

    Returns
    -------
        np.array of filtered data
    """
    # prepend 10 records to data for initial spin up
    # these records will be dropped before returning data to user
    qt = np.zeros((arr.size + 10,), dtype=float)
    qt[0:10] = arr[0:10]
    qt[10:] = arr[:]

    qdt = np.zeros((arr.size + 10,), dtype=float)
    qbf = np.zeros((arr.size + 10,), dtype=float)

    y = (1.0 + beta) / 2.0

    for ix in range(qdt.size):
        if ix == 0:
            qbf[ix] = qt[ix]
            continue

        x = beta * qdt[ix - 1]
        z = qt[ix] - qt[ix - 1]
        qdt[ix] = x + (y * z)

        qb = qt[ix] - qdt[ix]
        if enforce:
            if qb > qt[ix]:
                qbf[ix] = qt[ix]
            else:
                qbf[ix] = qb

        else:
            qbf[ix] = qb

    return qbf[10:]


def sliding_density(df, nn=3, fp_distance=0.0):
    """
    User method to apply a density fiter to time series
    data. Uses local distances to flag outliers

    Parameters
    ----------
    df : pd.dataframe
        streamflow or municipal pumping time series

    nn : int
        number of nearest neighbors to consider

    distance : float
        distance check factor for false positives

    """
    arr = df.mgal.values

    arr_size = int(arr.size + (2 * nn))
    darr = np.zeros((arr_size,))
    flags = np.zeros((arr_size,))

    # prepend and append nn number of records to darr to precondition data
    inn = -1 * nn
    darr[0:nn] = arr[0:nn]
    darr[nn:inn] = arr[:]
    darr[inn:] = arr[inn:]

    # set our window index stops
    if nn % 2 == 0:
        high = int(nn / 2)
        low = int(nn / 2)
    else:
        high = int((nn + 1) / 2)
        low = int(((nn - 1) / 2))

    lrange = low
    hrange = darr.size - high
    for ix in range(lrange, hrange):
        xo = darr[ix]

        for ixx in range(ix - low, ix + high + 1):
            if ixx == ix:
                continue
            else:
                xp = darr[ixx]
                xarr = list(darr[ix - low : ix + high + 1])
                xarr.remove(xo)
                xarr.remove(xp)
                xarr = np.array(xarr)
                d0 = np.abs(xp - xo)
                d1 = np.median(np.abs(xarr - xo))

                if d0 > d1:
                    # check xp vs. nearest neighbor in xarr
                    dn = np.min(np.abs(xarr - xp))
                    if dn < d1 and d0 / d1 < fp_distance:
                        flags[ixx] -= 1
                    else:
                        flags[ixx] += 1
                else:
                    flags[ixx] -= 1

    flags[flags > 1] = 1
    flags[flags <= 0] = 0

    df["outlier"] = flags[nn:inn]
    df0 = df[df["outlier"] == 1]
    df1 = df[df["outlier"] == 0]

    return df1, df0


def moving_window(df, window=0.0192, order=2, plot=False):
    """
    A moving window method to detect outliers and clean data
    for machine learning

    Parameters
    ----------
    df : pd.dataframe
        pandas dataframe of input data
    window : float
        value between 0 and 1 that represent the length of the moving
        window in years
    order : float
        number of standard deviations to clean data to.
    plot : bool
        plot the outlier removal data

    Returns
    -------
    tuple of pandas dataframes:
        (data, outliers, stats)
    """
    dyear = np.unique(df.dyear.values)

    outliers = {i: [] for i in list(df)}
    data = {i: [] for i in list(df)}
    means = []
    std_ups = []
    std_dns = []
    win_start = 0
    win_end = window
    win_dyears = []
    while win_start < 1.0:
        td = df[(df.dyear >= win_start) & (df.dyear < win_end)]
        win_dyear = (win_start + win_end) / 2
        mean = np.mean(td.mgal.values)
        std = np.std(td.mgal.values) * order
        std_up = mean + std
        std_dn = mean - std
        if std_dn <= 0:
            std_dn = 1e-8

        dfoutlier = td[(td.mgal > std_up) | (td.mgal < std_dn)]
        dfdata = td[(td.mgal <= std_up) & (td.mgal >= std_dn)]

        for key in list(df):
            outliers[key] += dfoutlier[key].tolist()
            data[key] += dfdata[key].tolist()

        means.append(mean)
        std_ups.append(std_up)
        std_dns.append(std_dn)
        win_dyears.append(win_dyear)
        win_start += window
        win_end += window

    outliers = pd.DataFrame.from_dict(outliers)
    data = pd.DataFrame.from_dict(data)

    stats = pd.DataFrame.from_dict(
        {
            "dyear": win_dyears,
            "mean": means,
            "h_bound": std_ups,
            "l_bound": std_dns,
        }
    )

    if plot:
        plt.plot(data.dyear.values, data.mgal.values, "bo")
        plt.plot(outliers.dyear.values, outliers.mgal.values, "ro")
        plt.plot(win_dyears, means, "k-")
        plt.plot(win_dyears, std_ups, "k--")
        plt.plot(win_dyears, std_dns, "k--")
        plt.xlabel("mgal pumped")
        plt.ylabel("scaled year")
        plt.show()

    return data, outliers, stats
