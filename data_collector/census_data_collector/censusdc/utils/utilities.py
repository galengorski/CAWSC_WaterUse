import os
import requests
import shapefile
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import threading
import datetime


def isnumeric(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def isdatetime(s):
    if isinstance(s, (datetime.date, datetime.datetime)):
        return True
    else:
        return False


def isbytes(s):
    if isinstance(s, bytes):
        return True
    else:
        return False


def get_wkt_wkid_table(refresh=False):
    """
    Method to build or load the ArcGIS Well Known Text and Well Known ID
    table

    Returns
    -------

    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    table_file = os.path.join(utils_dir, "..", "cache", "wkid_wkt_table.dat")

    if not os.path.isfile(table_file) or refresh:
        data = []
        for url in (
            "https://developers.arcgis.com/rest/services-reference/"
            "projected-coordinate-systems.htm",
            "https://developers.arcgis.com/rest/services-reference/"
            "geographic-coordinate-systems.htm",
        ):
            r = requests.get(url, verify=False)
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.findAll("table")
            table_body = table[0].find("tbody")
            rows = table_body.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                cols = [ele.text.strip() for ele in cols]
                data.append([ele for ele in cols if ele])

        data2 = ["\t".join(d) + "\n" for d in data]
        data2.insert(0, "wkid\tname\twkt\n")
        with open(table_file, "w") as foo:
            foo.writelines(data2)

    df = pd.read_csv(
        table_file,
        delimiter="\t",
        index_col=False,
    )
    return df


def thread_count():
    """
    Method to estimate the thread count on a user's machine

    Returns
    -------
        int : number of threads
    """
    import os

    nthreads = os.cpu_count()
    return nthreads


class RestartableThread(threading.Thread):
    """
    Restartable instance of a thread
    """

    def __init__(self, *args, **kwargs):
        self._args, self._kwargs = args, kwargs
        super().__init__(*args, **kwargs)

    def clone(self):
        return RestartableThread(*self._args, **self._kwargs)


def create_filter(shp, criteria, return_field):
    """
    Filter creation method for large and complex shapefiles

    Parameters
    ----------
    shp : shapefile path
    criteria : dict
        filter criteria {field_name_0 : [tag], ...,
                        field_name_n : [tag_0, ..., tag_m]}
    return_field : str
        filtered return field name, can be passed to CensusTimeSeries
        and TigerWeb

    Returns
    -------
        tuple
    """
    with shapefile.Reader(shp) as foo:
        header = [i[0].lower() for i in foo.fields[1:]]
        data = {i: [] for i in header}

        for record in foo.records():
            for ix, v in enumerate(record):
                if isnumeric(v):
                    data[header[ix]].append(v)
                elif isdatetime(v):
                    data[header[ix]].append(v)
                else:
                    if isbytes(v):
                        v = v.decode()
                    data[header[ix]].append(v.lower())

    df = pd.DataFrame.from_dict(data)

    # prep criteria dictionary
    c2 = {k.lower(): [] for k in criteria.keys()}
    for k, v in criteria.items():
        if not isinstance(v, (tuple, list, np.ndarray)):
            v = [v]

        for i in v:
            if isnumeric(i):
                c2[k.lower()].append(i)
            elif isdatetime(i):
                c2[k.lower()].append(i)
            else:
                c2[k.lower()].append(i.lower())

    for key in c2.keys():
        if key not in header:
            err = "{}: is not a valid shapefile attribute name".format(key)
            raise KeyError(err)

    # intersect criteria with shapefile attribute df
    for k, v in c2.items():
        df = df.loc[df[k].isin(v)]

    x = tuple(df[return_field].values)
    return x
