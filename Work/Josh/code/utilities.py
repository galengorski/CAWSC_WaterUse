import pandas as pd
import numpy as np


FIELDS = (
    "wsa_agidf",
    "sum_gu_pop",
    "x_centroid",
    "y_centroid",
    "tot_wd_mgd",
    "year"
)


def get_input_data(f, dataframe=None):
    """
    Method to read and clean input data for processing in outlier detection
    work

    Parameters
    ----------
    f : str
        csv file name to be imported by pandas
    df : None or pd.Dataframe
        if not none we can join by wsa

    Returns
    -------
        pd.DataFrame
    """
    df = pd.read_csv(f)

    drop = []
    for col in list(df):
        if col.lower() not in FIELDS:
            drop.append(col)

    lowered = {i: i.lower() for i in list(df)}
    df = df.drop(columns=drop)
    df = df.rename(columns=lowered)
    print(len(df))
    if dataframe is not None:
        df = pd.merge(
            left=dataframe,
            right=df,
            left_on='wsa_agidf',
            right_on='wsa_agidf'
        )

        if "sum_gu_pop" in list(df) and "tot_wd_mgd" in list(df):
            df["wu_pp_gd"] = (df.tot_wd_mgd / df.sum_gu_pop) * 10e+6
            df = df[df.wu_pp_gd != 0]
            df = df.replace([np.inf, -np.inf], 0)
        if "year" in list(df):
            df = df.loc[df.year == 2010]


    print(len(df))
    return df


def point_in_polygon(xc, yc, polygon):
    """
    Use the ray casting algorithm to determine if a point
    is within a polygon. Enables very fast
    intersection calculations!

    Parameters
    ----------
    xc : np.ndarray
        array of xpoints
    yc : np.ndarray
        array of ypoints
    polygon : iterable (list)
        polygon vertices [(x0, y0),....(xn, yn)]
        note: polygon can be open or closed

    Returns
    -------
    mask: np.array
        True value means point is in polygon!

    """
    x0, y0 = polygon[0]
    xt, yt = polygon[-1]

    # close polygon if it isn't already
    if (x0, y0) != (xt, yt):
        polygon.append((x0, y0))

    ray_count = np.zeros(xc.shape, dtype=int)
    num = len(polygon)
    j = num - 1
    for i in range(num):

        tmp = polygon[i][0] + (polygon[j][0] - polygon[i][0]) * (
            yc - polygon[i][1]
        ) / (polygon[j][1] - polygon[i][1])

        comp = np.where(
            ((polygon[i][1] > yc) ^ (polygon[j][1] > yc)) & (xc < tmp)
        )

        j = i
        if len(comp[0]) > 0:
            ray_count[comp[0], comp[1]] += 1

    mask = np.ones(xc.shape, dtype=bool)
    mask[ray_count % 2 == 0] = False

    return mask


def load_national_polygons(f):
    """
    Method to load the national boundaries from US Census into
    polygons

    Parameters
    ----------
    f : str
        shapefile name

    Returns
    -------
        list of vertices
    """
    import shapefile
    t = []
    with shapefile.Reader(f) as r:
        for shape in r.shapes():
            feat = shape.__geo_interface__
            for poly in feat['coordinates']:
                t.append(poly[0])
    return t