import pandas as pd
import numpy as np
import threading
import queue
import os
import matplotlib.pyplot as plt
import shapefile


COMMON = (
    "wsa_agidf",
    "sum_gu_pop",
    "x_centroid",
    "y_centroid",
    "year",
)


YEARLY = (
    "tot_wd_mgd",
)

MONTHLY = (
    "jan_mgd",
    "feb_mgd",
    "mar_mgd",
    "apr_mgd",
    "may_mgd",
    "jun_mgd",
    "jul_mgd",
    "aug_mgd",
    "sep_mgd",
    "oct_mgd",
    "nov_mgd",
    "dec_mgd"
)


def get_input_data(f, dataframe=None, monthly=False, normalized=False):
    """
    Method to read and clean input data for processing in outlier detection
    work

    Parameters
    ----------
    f : str
        csv file name to be imported by pandas
    dataframe : None or pd.Dataframe
        if not none we can join by wsa
    monthly : bool
        flag to read in montly water use data
    normalized : bool
        flag to normalize monthly water use data

    Returns
    -------
        pd.DataFrame
    """
    if not monthly:
        FIELDS = COMMON + YEARLY
    else:
        FIELDS = COMMON + MONTHLY

    df = pd.read_csv(f)

    drop = []
    for col in list(df):
        if col.lower() not in FIELDS:
            drop.append(col)

    lowered = {i: i.lower() for i in list(df)}
    df = df.drop(columns=drop)
    df = df.rename(columns=lowered)
    if dataframe is not None:
        df = pd.merge(
            left=dataframe,
            right=df,
            left_on='wsa_agidf',
            right_on='wsa_agidf'
        )

        if "sum_gu_pop" in list(df):
            if "tot_wd_mgd" in list(df):
                df["wu_pp_gd"] = (df.tot_wd_mgd / df.sum_gu_pop) * 10e+6
                df = df.replace([np.inf, -np.inf], 0)
                df = df[df.wu_pp_gd != 0]
            elif "jan_mgd" in list(df) and not normalized:
                for field in MONTHLY:
                    new = field.split("_")[0] + "_pp_gd"
                    df[new] = (df[field] / df.sum_gu_pop) * 10e+6

                df = df.replace([np.inf, -np.inf], 0)
                df = df.replace([np.nan,], 0)

            elif "jan_mgd" in list(df) and normalized:
                df["tot_wu_mgd"] = np.zeros((len(df),))
                for field in MONTHLY:
                    df['tot_wu_mgd'] += df[field]

                for field in MONTHLY:
                    new = field.split("_")[0] + "_norm"
                    df[new] = df[field] / df['tot_wu_mgd']

                df = df.drop(columns=list(MONTHLY))
                df = df[df['tot_wu_mgd'].notna()]

        if "year" in list(df):
            df = df.loc[df.year == 2010]

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


def array_to_shapefile(shp_name, array, xc, yc):
    """
    Method to write a dictionary of arrays to a shapefile

    Parameters
    ----------
    shp_name : str
    array : dict
        {attribute name : array}
    xc : np.ndarray
        array of cell centers
    yc : np.ndarray
        array of cell centers

    Returns
    -------
        None
    """
    minx, maxx = np.min(xc), np.max(xc)
    miny, maxy = np.min(yc), np.max(yc)

    dx = abs(xc[0, 0] - xc[0, 1])
    dy = abs(yc[0, 0] - yc[1, 0])
    xxv = np.arange(minx - dx / 2, maxx + 1 + dx / 2, dx)
    yyv = np.arange(miny - dy / 2, maxy + 1 + dy / 2, dy)

    xxv, yyv = np.meshgrid(xxv, yyv)

    with shapefile.Writer(shp_name, shapeType=shapefile.POLYGON) as w:
        for k in array.keys():
            w.field(k, 'N')

        for i in range(xc.shape[0]):
            for j in range(xc.shape[1]):
                verts = [(xxv[i, j], yyv[i, j]),
                         (xxv[i, j+1], yyv[i, j+1]),
                         (xxv[i+1, j+1], yyv[i+1, j+1]),
                         (xxv[i+1, j], yyv[i+1, j])]

                vals = [v[i, j] for k, v in array.items()]
                if np.isnan(vals).all():
                    continue

                else:
                    w.poly([verts])
                    w.record(*vals)

    make_alb83_proj(shp_name)


def points_to_shapefile(shp_name, df):
    """
    Method to export a pandas dataframe to shapefile

    Parameters
    ----------
    shp_name : str
    df : pd.DataFrame
        dataframe must contain x_centroid and y_centriod fields!

    Returns
    -------
        None
    """
    cols = list(df)
    if "x_centroid" not in cols or "y_centroid" not in cols:
        raise AssertionError("spatial information not in df")

    with shapefile.Writer(shp_name, shapeType=shapefile.POINT) as w:
        for col in cols:
            if col == "wsa_agidf":
                w.field(col, "C")
            else:
                w.field(col, 'N', decimal=1)

        for iloc, rec in df.iterrows():
            w.point(rec.x_centroid, rec.y_centroid)
            vals = [rec[i] for i in cols]
            w.record(*vals)

    make_alb83_proj(shp_name)


def make_alb83_proj(shp_name):
    """
    Method to add an albers83 projection file to shapefile

    Parameters
    ----------
    shp_name : str

    Returns
    -------
        None
    """
    proj = 'PROJCS["NAD_1983_Contiguous_USA_Albers",' \
           'GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",' \
           'SPHEROID["GRS_1980",6378137.0,298.257222101]],' \
           'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],' \
           'PROJECTION["Albers"],PARAMETER["False_Easting",0.0],' \
           'PARAMETER["False_Northing",0.0],' \
           'PARAMETER["Central_Meridian",-96.0],' \
           'PARAMETER["Standard_Parallel_1",29.5],' \
           'PARAMETER["Standard_Parallel_2",45.5],' \
           'PARAMETER["Latitude_Of_Origin",23.0],UNIT["Meter",1.0]]'
    prj_name = shp_name[:-4] + ".prj"
    with open(prj_name, "w") as foo:
        foo.write(proj)


def plot_map(array, xc, yc, ax=None, **kwargs):
    """
    Method to plot up mapped arrays of water use

    Parameters
    ----------
    array : np.ndarray
        array of interpolated data
    xc : np.ndarray
        array of xvertices (cell centers)
    yc : np.ndarray
        array of yvertices (cell centers)
    ax : Axes object
        optional matplotlib axes object
    kwargs : matplotlib keyword arguments

    Returns
    -------
        ax : matplotlib axes object
    """
    minx, maxx = np.min(xc), np.max(xc)
    miny, maxy = np.min(yc), np.max(yc)

    dx = abs(xc[0, 0] - xc[0, 1])
    dy = abs(yc[0, 0] - yc[1, 0])
    xxv = np.arange(minx - dx / 2, maxx + 1 + dx / 2, dx)
    yyv = np.arange(miny - dy / 2, maxy + 1 + dy / 2, dy)

    xxv, yyv = np.meshgrid(xxv, yyv)

    if ax is None:
        ax = plt.gca()

    quadmesh = ax.pcolormesh(xxv, yyv, array, **kwargs)
    return quadmesh


def plot_1_to_1(df, fields, ax=None, **kwargs):
    """
    Method to create one to one plots colored by outlier level

    Parameters
    ----------
    df : pd.dataframe
    fields : list of str
        fields (observed, processed data) to plot
    ax : matplotlib ax object (optional
    kwargs : matplotlib keyword arguments

    """
    if ax is None:
        ax = plt.gca()

    obs = df[fields[0]].values
    sim = df[fields[1]].values

    ax.scatter(obs, sim, **kwargs)
    return ax
