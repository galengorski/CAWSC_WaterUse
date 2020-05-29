import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import threading


def get_wkt_wkid_table(refresh=False):
    """
    Method to build or load the ArcGIS Well Known Text and Well Known ID
    table

    Returns
    -------

    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    table_file = os.path.join(utils_dir, '..', 'cache', 'wkid_wkt_table.dat')

    if not os.path.isfile(table_file) or refresh:
        data = []
        for url in ('https://developers.arcgis.com/rest/services-reference/'
                    'projected-coordinate-systems.htm',
                    'https://developers.arcgis.com/rest/services-reference/'
                    'geographic-coordinate-systems.htm'):
            r = requests.get(url, verify=False)
            soup = BeautifulSoup(r.text, 'html.parser')
            table = soup.findAll('table')
            table_body = table[0].find('tbody')
            rows = table_body.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                data.append([ele for ele in cols if ele])

        data2 = ['\t'.join(d) + '\n' for d in data]
        data2.insert(0, 'wkid\tname\twkt\n')
        with open(table_file, 'w') as foo:
            foo.writelines(data2)

    df = pd.read_csv(table_file, delimiter='\t', index_col=False,)
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
        return RestartableThread(self._args, self._kwargs)
