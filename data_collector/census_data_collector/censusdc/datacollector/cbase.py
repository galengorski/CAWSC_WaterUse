import requests
from .tigerweb import TigerWebVariables
from ..utils import Acs5Server, Acs1Server, Sf3Server, RestartableThread, \
    thread_count
import threading
import platform
import copy
import warnings
warnings.simplefilter('always', UserWarning)
try:
    from simplejson.errors import JSONDecodeError
except ImportError:
    from json import JSONDecodeError

if platform.system().lower() != "windows":
    import ray
else:
    # fake ray wrapper function for windows
    from ..utils import ray


class CensusBase(object):
    """
    Base class for Sf3, Acs1 and Acs5 data

    Parameters
    ----------
    features: dict
        features from TigerWeb data collections
        {polygon_name: [geojson, geojson,...]}
    year : int
        census data year of the TigerWeb data
    apikey : str
        users specific census apikey (obtained from
        https://api.census.gov/data/key_signup.html)

    """
    def __init__(self, features, year, apikey, server):
        if not isinstance(features, dict):
            raise TypeError('features must be in a dictionary format')
        self._features = features
        self._year = year
        self.__apikey = apikey
        self._text = server

        if server == 'acs1':
            self._server = Acs1Server
            self.__level_dict = {1: 'state', 2: 'county',
                                 3: 'county_subdivision'}
            self.__ilevel_dict = {'state': 1, 'county': 2,
                                  'county_subdivision': 3}
        elif server == 'acs5':
            self._server = Acs5Server
            self.__level_dict = {1: 'state', 2: 'county',
                                 3: 'tract', 4: 'block_group'}
            self.__ilevel_dict = {"state": 1, "county": 2,
                                  "tract": 3, "block_group": 4}
        elif server == "sf3":
            self._server = Sf3Server
            self.__level_dict = {1: "state", 2: "county",
                                 3: 'tract', 4: 'block_group'}
            self.__ilevel_dict = {"state": 1, "county": 2,
                                  "tract": 3, "block_group": 4}
        else:
            raise AssertionError("unknown server type")

        self._features_level = 'undefined'
        self.__thread_fail = {}
        self.__set_features_level()

    @property
    def year(self):
        """
        Method to return the ACS query year
        """
        return self._year

    @property
    def features(self):
        """
        Method to return the feature dictionary
        """
        return copy.deepcopy(self._features)

    @property
    def feature_names(self):
        """
        Method to return the feature names
        """
        return list(self._features.keys())

    @property
    def features_level(self):
        """
        Minimum common level (maximum resolution) of all the
        features supplied to Acs

        """
        return self._features_level

    def get_feature(self, name):
        """
        Method to get a set of features

        Parameters
        ----------
        name : str
            feature set name

        Returns
        -------
            list of geoJSON features
        """
        if name not in self._features:
            name = str(name)

        if name not in self._features:
            raise KeyError("Name: {} not present in feature dict".format(name))
        else:
            return copy.deepcopy(self._features[name])

    def __set_features_level(self):
        """
        Internal method to determine a common 'finest' discretization
        of all supplied featues

        """
        level = []
        for name in self.feature_names:
            for feature in self.get_feature(name):
                tmp = 0
                for key in (TigerWebVariables.state,):
                    if key in feature.properties:
                        tmp = 1
                    else:
                        raise AssertionError("State number must be in "
                                             "properties dict")
                if tmp == 1:
                    for key in (TigerWebVariables.county,
                                TigerWebVariables.state):
                        if key in feature.properties:
                            tmp = 2
                        else:
                            tmp = 1
                            break
                if tmp == 2:
                    if self._text == 'acs1':
                        for key in (TigerWebVariables.cousub,
                                    TigerWebVariables.state,
                                    TigerWebVariables.county):
                            if key in feature.properties:
                                tmp = 3
                            else:
                                tmp = 2
                                break
                    else:
                        for key in (TigerWebVariables.tract,
                                    TigerWebVariables.state,
                                    TigerWebVariables.county):
                            if key in feature.properties:
                                tmp = 3
                            else:
                                tmp = 2
                                break
                if tmp == 3:
                    for key in (TigerWebVariables.blkgrp,
                                TigerWebVariables.state,
                                TigerWebVariables.county,
                                TigerWebVariables.tract):
                        if key in feature.properties:
                            tmp = 4
                        else:
                            tmp = 3
                            break

                level.append(tmp)

        if not level:
            # hack around canadian huc12's for now. print a warning in future
            msg = "Cannot determine census data level: setting to 'tract'"
            warnings.warn(msg, UserWarning)
            self._features_level = self.__level_dict[3]
        else:
            self._features_level = self.__level_dict[min(level)]

    def get_data(self, level='finest', variables=(), retry=100, verbose=True,
                 multiproc=False, multithread=False, thread_pool=4):
        """
        Method to get data from the Acs5 servers and set it to feature
        properties!

        Parameters
        ----------
        level : str
            determines the geographic level of data queried
            default is 'finest' available based on census dataset and
            the geoJSON feature information
        variables : list, tuple
            user specified Acs5 variables, default pulls variables from
            the Acs5Variables class
        retry : int
            number of retries for HTTP connection issues before failure
        verbose : bool
            verbose operation mode
        multiproc : bool
            multiprocessing support using ray, linux only!
        multithread : bool
            flag to use a multithreaded method of collecting census data
        thread_pool : int
            number of threads to perform data collection while multithreading
        """
        url = self._server.base.format(self.year)

        lut = None
        if level == 'finest':
            for level in self._server.levels:
                lut = self._server.__dict__[level]
                if self.year in lut:
                    break
                else:
                    lut = None

            if level != self.features_level:
                level = self.features_level
                lut = self._server.__dict__[level]
                if self.year not in lut:
                    lut = None

        else:
            if level in self._server.__dict__:
                lut = self._server.__dict__[level]
                if self.year in lut:
                    pass
                else:
                    lut = None

            if self.__ilevel_dict[level] > \
                    self.__ilevel_dict[self.features_level]:
                raise AssertionError("Cannot grab level data finer than {}"
                                     .format(self.features_level))

        if lut is None:
            raise KeyError("No {} server could be found for {} and {}"
                           .format(self._text, self.year, level))

        if variables:
            if isinstance(variables, str):
                variables = (variables,)
            variables = ",".join(variables)
        else:
            variables = lut[self.year]["variables"]

        fmt = lut[self.year]['fmt']

        if multiproc and platform.system().lower() == "windows":
            multiproc = False
            multithread = True
            thread_pool = thread_count() - 1

        if multiproc:
            year = self.year
            apikey = self.__apikey
            actors = []
            twv = {k: v for k, v in TigerWebVariables.__dict__.items()
                   if not k.startswith("__")}
            for name in self.feature_names:
                for featix, feature in enumerate(self.get_feature(name)):
                    actor = multiproc_data_request.remote(
                        year, apikey, feature, featix,
                        name, level, fmt, variables, url,
                        retry, verbose, twv)

                    actors.append(actor)

            output = ray.get(actors)

            for out in output:
                if out is None:
                    continue
                else:
                    name, featix, data = out
                    for dix, header in enumerate(data[0]):
                        if header == "NAME":
                            continue
                        elif header not in variables:
                            continue
                        else:
                            try:
                                self._features[name][featix].properties[header] = \
                                    float(data[1][dix])
                            except (TypeError, ValueError):
                                self._features[name][featix].properties[header] = \
                                    float('nan')

        elif multithread:
            container = threading.BoundedSemaphore(thread_pool)
            thread_list = []
            thread_id = 0
            for name in self.feature_names:
                for featix, feature in enumerate(self.get_feature(name)):
                    x = RestartableThread(target=self.threaded_request_data,
                                          args=(feature, featix, name,
                                                level, fmt, variables,
                                                url, retry, verbose, thread_id,
                                                container))
                    thread_list.append(x)
                    thread_id += 1

            for thread in thread_list:
                thread.start()
            for thread in thread_list:
                thread.join()

            restart = []
            for thread_id, fail in self.__thread_fail.items():
                if fail:
                    restart.append(thread_list[thread_id].clone())
            for thread in restart:
                thread.start()
            for thread in restart:
                thread.join()

        else:
            for name in self.feature_names:
                for featix, feature in enumerate(self.get_feature(name)):
                    self.__request_data(feature, featix, name, level, fmt,
                                        variables, url, retry, verbose)

    def __request_data(self, feature, featix, name, level, fmt, variables,
                       url, retry, verbose):
        """
        Request data method for serial and multithreaded applications

        Parameters
        ----------
        feature : geoJSON
            geoJSON feature
        featix : int
            feature index (in an enumeration)
        name : str
            feature name
        level : str
            census data level
        fmt : str
            format of level request
        variables : str
            string of census variables
        url : str
            string of census url
        retry : int
            number of retries based on connection error
        verbose : bool
            verbose operation flag

        Returns
        -------

        """
        loc = ""
        if level == "block_group":
            loc = fmt.format(
                feature.properties[TigerWebVariables.blkgrp],
                feature.properties[TigerWebVariables.state],
                feature.properties[TigerWebVariables.county],
                feature.properties[TigerWebVariables.tract])
        elif level == "tract":
            loc = fmt.format(
                feature.properties[TigerWebVariables.tract],
                feature.properties[TigerWebVariables.state],
                feature.properties[TigerWebVariables.county])
        elif level == "county_subdivision":
            loc = fmt.format(
                feature.properties[TigerWebVariables.cousub],
                feature.properties[TigerWebVariables.state],
                feature.properties[TigerWebVariables.county])
        elif level == "county":
            loc = fmt.format(
                feature.properties[TigerWebVariables.county],
                feature.properties[TigerWebVariables.state])
        elif level == "state":
            loc = fmt.format(
                feature.properties[TigerWebVariables.state])
        else:
            raise AssertionError("level is undefined")

        s = requests.session()

        if self.year == 1990:
            payload = {"get": variables,
                       "for": loc,
                       "key": self.__apikey}
        else:
            payload = {'get': "NAME," + variables,
                       'for': loc,
                       'key': self.__apikey}

        payload = "&".join(
            '{}={}'.format(k, v) for k, v in payload.items())

        n = 0
        e = "Unknown connection error"
        while n < retry:
            try:
                r = s.get(url, params=payload)
                r.raise_for_status()
                break
            except (requests.exceptions.HTTPError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ReadTimeout) as e:
                n += 1
                print("Connection Error: Retry number "
                      "{}".format(n))

        if n == retry:
            raise requests.exceptions.HTTPError(e)

        if verbose:
            print('Getting {} data for {} '
                  'feature # {}'.format(level, name, featix))
        try:
            data = r.json()
        except JSONDecodeError:
            data = []
            if verbose:
                print('Error getting {} data for {} '
                      'feature # {}'.format(level, name, featix))

        if len(data) == 2:
            for dix, header in enumerate(data[0]):
                if header == "NAME":
                    continue
                elif header not in variables:
                    continue
                else:
                    try:
                        self._features[name][featix].properties[header] = \
                            float(data[1][dix])
                    except (TypeError, ValueError):
                        self._features[name][featix].properties[header] = \
                            float('nan')

    def threaded_request_data(self, feature, featix, name, level, fmt, variables,
                               url, retry, verbose, thread_id, container):
        """
        Multithread method for requesting census data

        Parameters
        ----------
        feature : geoJSON
            geoJSON feature
        featix : int
            feature index (in an enumeration)
        name : str
            feature name
        level : str
            census data level
        fmt : str
            format of level request
        variables : str
            string of census variables
        url : str
            string of census url
        retry : int
            number of retries based on connection error
        verbose : bool
            verbose operation flag
        thread_id : int
            identifier for the thread
        container : BoundedSemaphore
            bound semaphore instance for thread pool management

        """
        container.acquire()
        self.__thread_fail[thread_id] = True
        self.__request_data(feature, featix, name, level, fmt, variables,
                            url, retry, verbose)
        self.__thread_fail[thread_id] = False
        container.release()

@ray.remote
def multiproc_data_request(year, apikey, feature, featix, name,
                           level, fmt, variables, url, retry, verbose,
                           TigerwebVariables):
    """
    Multithread method for requesting census data

    Parameters
    ----------
    year : int
        data year
    apikey : str
        census apikey string
    feature : geoJSON
        geoJSON feature
    featix : int
        feature index (in an enumeration)
    name : str
        feature name
    level : str
        census data level
    fmt : str
        format of level request
    variables : str
        string of census variables
    url : str
        string of census url
    retry : int
        number of retries based on connection error
    verbose : str
        verbose operation flag
    """
    loc = ""
    if level == "block_group":
        loc = fmt.format(
            feature.properties[TigerwebVariables['blkgrp']],
            feature.properties[TigerwebVariables['state']],
            feature.properties[TigerwebVariables['county']],
            feature.properties[TigerwebVariables['tract']])
    elif level == "tract":
        loc = fmt.format(
            feature.properties[TigerwebVariables['tract']],
            feature.properties[TigerwebVariables['state']],
            feature.properties[TigerwebVariables['county']])
    elif level == "county_subdivision":
        loc = fmt.format(
            feature.properties[TigerwebVariables['cousub']],
            feature.properties[TigerwebVariables['state']],
            feature.properties[TigerwebVariables['county']])
    elif level == "county":
        loc = fmt.format(
            feature.properties[TigerwebVariables['county']],
            feature.properties[TigerwebVariables['state']])
    elif level == "state":
        loc = fmt.format(
            feature.properties[TigerwebVariables['state']])
    else:
        raise AssertionError("level is undefined")

    s = requests.session()

    if year == 1990:
        payload = {"get": variables,
                   "for": loc,
                   "key": apikey}
    else:
        payload = {'get': "NAME," + variables,
                   'for': loc,
                   'key': apikey}

    payload = "&".join(
        '{}={}'.format(k, v) for k, v in payload.items())

    n = 0
    e = "Unknown connection error"
    while n < retry:
        try:
            r = s.get(url, params=payload)
            r.raise_for_status()
            break
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ReadTimeout) as e:
            n += 1
            print("Connection Error: Retry number {}".format(n))

    if n == retry:
        raise requests.exceptions.HTTPError(e)

    if verbose:
        print('Getting {} data for {} feature # {}'.format(level,
                                                           name,
                                                           featix))
    try:
        data = r.json()
    except JSONDecodeError:
        data = []
        if verbose:
            print('Error getting {} data for {} feature # {}'.format(
                  level, name, featix))
    if len(data) == 2:
        return name, featix, data
    else:
        return None
