import pandas as pd
import numpy as np
import threading
import queue
import os

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


def threaded_point_in_polygon(q, xc, yc, polygon, container):
    """
    Theaded method for doing point in polygon array operations

    q : Queue.Queue object
    xc : np.ndarray
        array of xpoints
    yc : np.ndarray
        array of ypoints
    polygon : iterable (list)
        polygon vertices [(x0, y0),....(xn, yn)]
        note: polygon can be open or closed
    container : threading.BoundedSemaphore

    Returns
    -------
    mask: np.array
        True value means point is in polygon!
    """
    container.acquire()
    mask = point_in_polygon(xc, yc, polygon)
    q.put(mask)
    container.release()


def get_interp_mask(xc, yc, polygons,
                    multithread=False,
                    num_threads=8):
    """
    Method to build, save, and load a mask for interpolation

    Parameters
    ----------
    xc : np.ndarray
        xcenters array
    yc : np.ndarray
        ycenters array
    polygons : list
        list of polygons vertices
    multithread : bool
        boolean flag to enable multithreading
    num_threads : int
        number of threads to use for calculation
    Returns
    -------
        mask: boolean np.ndarray
    """
    yshape, xshape = xc.shape

    fbase = "interp_mask_{}_{}.dat".format(yshape, xshape)
    fname = os.path.join("..", "data", fbase)
    if os.path.exists(fname):
        mask = np.genfromtxt(fname, dtype=int)
        mask = np.asarray(mask, dtype=bool)

    elif multithread:
        q = queue.Queue()
        container = threading.BoundedSemaphore(num_threads)
        threads = []
        mask = np.zeros((yshape, xshape), dtype=int)
        for ix, poly in enumerate(polygons):
            t = threading.Thread(target=threaded_point_in_polygon,
                                 args=(q, xc, yc, poly, container))
            threads.append(t)

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        result = [q.get() for _ in range(len(threads))]
        for m in result:
            mask[m] = 1

        np.savetxt(fname, mask, fmt="%d", delimiter="  ")
        mask = np.asarray(mask, dtype=bool)

    else:
        mask = np.zeros((yshape, xshape), dtype=int)
        for ix, poly in enumerate(polygons):
            print("Creating mask: Percent done: "
                  "{:.3f}".format(ix/len(polygons)))
            m = point_in_polygon(xc, yc, list(poly))
            mask[m] = 1

        np.savetxt(fname, mask, fmt="%d", delimiter="  ")
        mask = np.asarray(mask, dtype=bool)

    return mask
