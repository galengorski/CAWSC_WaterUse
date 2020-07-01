"""
Development code for TigerWeb REST data collection.
"""
import geojson
import requests
import os
import shapefile
import pycrs
import copy
from ..utils import get_wkt_wkid_table, TigerWebMapServer, thread_count
from ..utils.geometry import calculate_circle
import threading
import platform

if platform.system().lower() != "windows":
    import ray
else:
    # fake ray wrapper function for windows
    from ..utils import ray


class TigerWebVariables(object):
    """
    Tigerweb variable names for querying data
    """
    mtfcc = 'MTFCC'
    oid = 'OID'
    geoid = 'GEOID'
    state = 'STATE'
    county = 'COUNTY'
    cousub = 'COUSUB'
    tract = 'TRACT'
    blkgrp = 'BLKGRP'
    block = 'BLOCK'
    basename = 'BASENAME'
    name = 'NAME'
    lsadc = 'LSADC'
    funcstat = 'FUNCSTAT'
    arealand = 'AREALAND'
    areawater = 'AREAWATER'
    stgeometry = 'STGEOMETRY'
    centlat = 'CENTLAT'
    centlon = 'CENTLON'
    intptlat = 'INTPTLAT'
    intptlon = 'INTPTLON'
    objectid = 'OBJECTID'
    population = 'POP100'


class TigerWebBase(object):
    """
    Base class object for all Tigerweb spatial queries

    Parameters
    ----------
    shp : str
        shapefile path
    field : str
        shapefile tag reference field
    geotype : str
        shapefile geometry type
    filter : tuple
        tuple of names or polygon numbers to pull from
        default is () which grabs all polygons

    """
    def __init__(self, shp, field, geotype, filter):
        if not os.path.isfile(shp):
            raise FileNotFoundError("{} not a valid file path".format(shp))
        prj = shp[:-4] + ".prj"
        if not os.path.isfile(prj):
            raise FileNotFoundError("{}: projection file not found"
                                    .format(prj))

        self._geotype = geotype
        self._shpname = shp
        self._prjname = prj

        if field is not None:
            self._field = field.lower()
        else:
            self._field = field

        self.sf = shapefile.Reader(self._shpname)
        self.prj = pycrs.load.from_file(self._prjname)

        self._wkid = None
        self._shapes = {}
        self._points = {}
        self._esri_json = {}
        self._features = {}

        if filter:
            if isinstance(filter, (int, str, float)):
                filter = (filter,)
            filter = tuple([i.lower() if isinstance(i, str) else
                            i for i in filter])

        self._filter = filter

    @property
    def esri_wkid(self):
        """
        Method to grab the esri WKID from projection information

        Returns
        -------
            wkid : int
        """
        if self._wkid is None:
            df = get_wkt_wkid_table()
            iloc = df.index[
                df['name'].str.lower() == self.prj.name.lower()].values
            if len(iloc) == 1:
                wkid = df.loc[iloc, 'wkid'].values[0]
                self._wkid = int(wkid)

            else:
                raise Exception("Can't find a matching esri wkid "
                                "for current projection")

        return self._wkid

    @property
    def points(self):
        """
        Method to get the shapefile points for each shape

        Returns
        -------
            dict : {name: point}
        """
        return self._points

    @property
    def shapes(self):
        """
        Method to get the shape vertices for each shape

        Returns
        -------
            dict : {name: [vertices]}
        """
        return copy.deepcopy(self._shapes)

    @property
    def features(self):
        """
        Method to get all features in the feature dictionary

        Returns
        -------
            dict : {name: geoJSON object}
        """
        if not self._features:
            print("Warning, no features in feature dict,")
            print('Please run get_data() to get TigerWeb features')

        return copy.deepcopy(self._features)

    @property
    def feature_names(self):
        """
        Gets the feature names from the feature dict

        Returns
        -------
            list: feature names
        """
        return list(self._features.keys())

    def get_feature(self, name):
        """
        Method to get a single GeoJSON feature from the feature dict

        Parameters
        ----------
        name : str or int
            feature dictionary key

        Returns
        -------
            geoJSON object
        """
        if name not in self._features:
            name = str(name)

        if name not in self._features:
            raise KeyError("Name: {} not present in feature dict".format(name))
        else:
            return copy.deepcopy(self._features[name])

    def get_shape(self, name):
        """
        Method to get the shapefile shapes from the shapes dict

        Parameters
        ----------
        name : str or int
            feature dictionary key

        Returns
        -------
            list
        """
        if name not in self._shapes:
            raise KeyError("Name: {} not present in shapes dict".format(name))
        else:
            return self._shapes[name]

    def get_point(self, name):
        """
        Method to get the shapefile point from the points dict

        Parameters
        ----------
        name : str or int
            feature dictionary key

        Returns
        -------
            list
        """
        if name not in self._points:
            raise KeyError("Name: {} not present in points dict".format(name))
        else:
            return self._points[name]

    def point_to_esri_json(self, point):
        """
        Method to create an esri json string for defining point geometry

        Parameters
        ----------
        point : list
            list of x, y coordinates

        Returns
        -------
        str : json geometry string
        """
        d = {"geometryType": "esriGeometryPoint",
             'x': point[0],
             'y': point[1],
             "spatialReference": {}}

        if isinstance(self.esri_wkid, int):
            d['spatialReference'] = {"wkid": self.esri_wkid}
        else:
            raise TypeError("wkid must be a well known id number string")

        s = d.__str__()
        s = s.replace("'", '"')
        s = s.replace(" ", "")
        return s

    def polygon_to_esri_json(self, polygon):
        """
        Method to create an esri json string for defining geometry to
        query

        Parameters
        ----------
        polygon : list
            list of x,y coordinates

        Returns
        -------
        str : json geometry string
        """
        d = {"geometryType": "esriGeometryPolygon",
             "rings": [],
             "spatialReference": {}}
        ring = []
        for pts in polygon:
            if isinstance(pts, tuple):
                pts = list(pts)
            elif isinstance(pts, list):
                pass
            else:
                try:
                    pts = list(pts)
                except ValueError:
                    raise Exception("Points must be supplied as a list")

            if len(pts) == 2:
                ring.append(pts)

            else:
                raise AssertionError("Each point must consist of only "
                                     "two coordinates")

        d["rings"].append(ring)

        if isinstance(self.esri_wkid, int):
            d['spatialReference'] = {"wkid": self.esri_wkid}
        else:
            raise TypeError("wkid must be a well known id number string")

        s = d.__str__()
        s = s.replace("'", '"')
        s = s.replace(" ", "")
        return s

    def get_data(self, year, level='finest', outfields=(), verbose=True,
                 multiproc=False, multithread=False, thread_pool=4, retry=100):
        """
        Method to pull data feature data from tigerweb

        Parameters
        ----------
        year : int
            data year to grab features from
        level : str
            block, block group, tract, or "finest"...
        outfields : tuple
            tuple of output variables to grab from tigerweb
            default is () which grabs GEOID,BLKGRP,STATE,COUNTY,TRACT
            from TigerWeb
        verbose : bool
            verbose operation mode
        multiproc : bool
            flag to enable multiprocessing on linux using ray
        multithread : bool
            flag to enable/disable multithreaded data collection
        thread_pool : int
            number of threads requested for multithreaded operations
        retry : int
            number of retries if connection is interupted

        Returns
        -------

        """
        level = level.lower()

        lut = None
        if level == 'finest':
            for level in TigerWebMapServer.levels:
                lut = TigerWebMapServer.__dict__[level]
                if year in lut:
                    break
                else:
                    lut = None

        else:
            if level in TigerWebMapServer.__dict__:
                lut = TigerWebMapServer.__dict__[level]
                if year in lut:
                    pass
                else:
                    lut = None

        if lut is None:
            raise KeyError("No TigerWeb server could be found for {} and {}"
                           .format(year, level))

        if level == "county_subdivision":
            base = TigerWebMapServer.lcdbase[year]
        elif level == "county":
            base = TigerWebMapServer.cobase[year]
        else:
            base = TigerWebMapServer.base[year]

        mapserver = lut[year]['mapserver']

        if self._geotype == 'point':
            geotype = 'esriGeometryPoint'
        else:
            geotype = 'esriGeometryPolygon'

        if outfields:
            if isinstance(outfields, str):
                outfields = (outfields,)
            outfields = ','.join(outfields).upper()
        else:
            outfields = lut[year]['outFields']

        if "GEOID" not in outfields:
            outfields += ",GEOID"

        if multiproc and platform.system().lower() == "windows":
            multiproc = False
            multithread = True
            thread_pool = thread_count() - 1

        if multiproc:
            actors = []
            for key, esri_json in self._esri_json.items():

                actor = multiproc_request_data.remote(key, base, mapserver,
                                                      esri_json, geotype,
                                                      outfields, verbose,
                                                      retry)
                actors.append(actor)

            output = ray.get(actors)

            for out in output:
                if out is None:
                    continue
                else:
                    key, features = out
                    self._features[key] = features

        elif multithread:
            thread_list = []
            container = threading.BoundedSemaphore(thread_pool)
            for key, esri_json in self._esri_json.items():

                x = threading.Thread(target=self.threaded_request_data,
                                     args=(key, base, mapserver, esri_json,
                                           geotype, outfields, verbose,
                                           retry, container))
                thread_list.append(x)

            for thread in thread_list:
                thread.start()
            for thread in thread_list:
                thread.join()

        else:
            for key, esri_json in self._esri_json.items():

                self.__request_data(key, base, mapserver, esri_json, geotype,
                                    outfields, verbose, retry)

        # cleanup duplicate features after query!
        for key, features in self._features.items():
            geocodes = []
            poplist = []
            for ix, feat in enumerate(features):
                # properties = feat.properties
                if feat.properties["GEOID"] in geocodes:
                    poplist.insert(0, ix)
                else:
                    geocodes.append(feat.properties["GEOID"])

            for p in poplist:
                features.pop(p)

            self._features[key] = features

    def __request_data(self, key, base, mapserver, esri_json, geotype,
                       outfields, verbose, retry):
        """
        Request data method for serial and multithread applications

        Parameters
        ----------
        key : str or int
            feature identifier
        base : str
            base url
        mapserver : int
            map server number
        esri_json : str
            json geography string
        geotype : str
            geometery type
        outfields : str
            string of requested variables
        verbose : bool
            verbose operation mode
        retry : int
            number of retries

        Returns
        -------

        """
        s = requests.session()
        url = '/'.join([base, str(mapserver), "query?"])

        s.params = {'where': '',
                    'text': '',
                    'objectIds': '',
                    'geometry': esri_json,
                    'geometryType': geotype,
                    'inSR': '',
                    'spatialRel': 'esriSpatialRelIntersects',
                    'relationParam': '',
                    'outFields': outfields,
                    'returnGeometry': True,
                    'returnTrueCurves': False,
                    'maxAllowableOffset': '',
                    'geometryPrecision': '',
                    'outSR': '',
                    'returnIdsOnly': False,
                    'returnCountOnly': False,
                    'orderByFields': '',
                    'groupByFieldsForStatistics': '',
                    'outStatistics': '',
                    'returnZ': False,
                    'returnM': False,
                    'gdbVersion': '',
                    'returnDistinctValues': False,
                    'f': 'geojson',
                    }
        start = 0
        done = False
        features = []
        n = 0

        while not done:
            try:
                r = s.get(url, params={'resultOffset': start,
                                       'resultRecordCount': 32})
                r.raise_for_status()
            except (requests.exceptions.HTTPError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:
                n += 1
                if verbose:
                    print("ConnectionError: retry number {}".format(n))

                if n == retry:
                    raise Exception(e)
                else:
                    continue

            # print(r.text)
            counties = geojson.loads(r.text)
            newfeats = counties.__geo_interface__['features']
            if newfeats:
                features.extend(newfeats)
                # crs = counties.__geo_interface__['crs']
                start += len(newfeats)
                if verbose:
                    print("Received", len(newfeats), "entries,",
                          start, "total")
            else:
                done = True

        self._features[key] = features

    def threaded_request_data(self, key, base, mapserver, esri_json, geotype,
                              outfields, verbose, retry, container):
        """
        Multithread handler method to request data from the TigerWeb server

        Parameters
        ----------
        key : str or int
            feature identifier
        base : str
            base url
        mapserver : int
            map server number
        esri_json : str
            json geography string
        geotype : str
            geometery type
        outfields : str
            string of requested variables
        verbose : bool
            verbose operation mode
        retry : int
            number of retries
        container : threading.BoundSemaphore

        """
        container.acquire()
        self.__request_data(key, base, mapserver, esri_json, geotype,
                            outfields, verbose, retry)
        container.release()


@ray.remote
def multiproc_request_data(key, base, mapserver, esri_json, geotype,
                           outfields, verbose, retry):
    """
    Ray multiprocessing handler method to request data from the TigerWeb server

    Parameters
    ----------
    key : str or int
        feature identifier
    base : str
        base url
    mapserver : int
        map server number
    esri_json : str
        json geography string
    geotype : str
        geometery type
    outfields : str
        string of requested variables
    verbose : bool
        verbose operation mode
    retry : int
        number of retries

    """
    s = requests.session()
    url = '/'.join([base, str(mapserver), "query?"])

    s.params = {'where': '',
                'text': '',
                'objectIds': '',
                'geometry': esri_json,
                'geometryType': geotype,
                'inSR': '',
                'spatialRel': 'esriSpatialRelIntersects',
                'relationParam': '',
                'outFields': outfields,
                'returnGeometry': True,
                'returnTrueCurves': False,
                'maxAllowableOffset': '',
                'geometryPrecision': '',
                'outSR': '',
                'returnIdsOnly': False,
                'returnCountOnly': False,
                'orderByFields': '',
                'groupByFieldsForStatistics': '',
                'outStatistics': '',
                'returnZ': False,
                'returnM': False,
                'gdbVersion': '',
                'returnDistinctValues': False,
                'f': 'geojson',
                }
    start = 0
    done = False
    features = []

    n = 0
    while not done:
        try:
            r = s.get(url, params={'resultOffset': start,
                                   'resultRecordCount': 32})
            r.raise_for_status()
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            n += 1
            if verbose:
                print("ConnectionError: retry number {}".format(n))

            if n == retry:
                raise Exception(e)
            else:
                continue

        counties = geojson.loads(r.text)
        newfeats = counties.__geo_interface__['features']
        if newfeats:
            features.extend(newfeats)
            # crs = counties.__geo_interface__['crs']
            start += len(newfeats)
            if verbose:
                print("Received", len(newfeats), "entries,",
                      start, "total")
        else:
            done = True

    return key, features


class TigerWebPoint(TigerWebBase):
    """
    Class to query data from TigerWeb by using shapefile point(s)

    Parameters
    ----------
    shp : str
        shapefile path
    field : str
        shapefile field to id multiple points
    radius : str or value
        shapefile radius field (in projection units) or a floating point value
        to define the polygon to query
    filter : tuple
        tuple of names or polygon numbers to pull from
        default is () which grabs all polygons
    """
    def __init__(self, shp, field=None, radius=0, filter=()):
        if isinstance(radius, str) or radius > 0:
            super(TigerWebPoint, self).__init__(shp, field, "polygon", filter)
        else:
            super(TigerWebPoint, self).__init__(shp, field, "point", filter)

        self.radius = radius

        if self._geotype == "polygon":
            self._get_polygons()

        else:
            self._get_points()

        self.sf.close()

    def _get_polygons(self):
        """
        Method to build and store polygons from a shapefile for
        latter processing

        Returns
        -------
            None
        """
        if self.sf.shapeType not in (1, 11, 21):
            raise TypeError('Shapetype: {}, is not a valid point'
                            .format(self.sf.shapeTypeName))

        named = False
        fidx = 0
        if self._field is None:
            pass
        else:
            for ix, field in enumerate(self.sf.fields):
                if field[0].lower() == self._field:
                    named = True
                    fidx = ix - 1
                    break

        ridx = 0
        if not isinstance(self.radius, str):
            pass
        else:
            for ix, field in enumerate(self.sf.fields):
                if field[0].lower() == self.radius.lower():
                    ridx = ix - 1
                    break

        name = -1
        for ix, shape in enumerate(self.sf.shapes()):
            points = shape.points[0]
            rec = self.sf.record(ix)
            if isinstance(self.radius, str):
                radius = rec[ridx]
            else:
                radius = self.radius

            polygon = calculate_circle(points[0], points[1], radius)
            polygon = polygon.T

            esri_json = self.polygon_to_esri_json(polygon)

            if named:
                rec = self.sf.record(ix)
                name = rec[fidx]
                if isinstance(name, str):
                    name = name.lower()
            else:
                name += 1

            if self._filter:
                if name not in self._filter:
                    continue

            self._esri_json[name] = esri_json
            self._points[name] = points
            polygon = geojson.Polygon(points)
            self._shapes[name] = geojson.Feature(geometry=polygon)

    def _get_points(self):
        """
        Method to build and store point features from a shapefile
        for later processing

        Returns
        -------

        """
        if self.sf.shapeType not in (1, 11, 21):
            raise TypeError('Shapetype: {}, is not a valid point'
                            .format(self.sf.shapeTypeName))

        named = False
        fidx = 0
        if self._field is None:
            pass
        else:
            for ix, field in enumerate(self.sf.fields):
                if field[0].lower() == self._field:
                    named = True
                    fidx = ix - 1
                    break

        name = -1
        for ix, shape in enumerate(self.sf.shapes()):
            points = shape.points[0]
            esri_json = self.point_to_esri_json(points)

            if named:
                rec = self.sf.record(ix)
                name = rec[fidx]
                if isinstance(name, str):
                    name = name.lower()
            else:
                name += 1

            if self._filter:
                if name not in self._filter:
                    continue

            self._esri_json[name] = esri_json
            self._points[name] = points
            self._shapes[name] = points


class TigerWebPolygon(TigerWebBase):
    """
    Class to query data from TigerWeb by using shapefile polygon(s)

    Parameters
    ----------
    shp : str
        shapefile path
    field : str
        shapefile field to id multiple polygons
    filter : tuple
        tuple of names or polygon numbers to pull from
        default is () which grabs all polygons

    """
    def __init__(self, shp, field=None, filter=()):
        super(TigerWebPolygon, self).__init__(shp, field, 'polygon', filter)

        self._get_polygons()
        self.sf.close()

    def _get_polygons(self):
        """
        Method to read and store polygons from a shapefile for later
        processing.

        Returns
        -------
            None
        """
        if self.sf.shapeType not in (5, 15, 25):
            raise TypeError('Shapetype: {}, is not a valid polygon'
                            .format(self.sf.shapeTypeName))

        named = False
        fidx = 0
        if self._field is None:
            pass
        else:
            for ix, field in enumerate(self.sf.fields):
                if field[0].lower() == self._field:
                    named = True
                    fidx = ix - 1
                    break

        name = -1
        for ix, shape in enumerate(self.sf.shapes()):
            shape = self.sf.shape(ix)
            if len(shape.points) > 20:
                # get the bbox to do the tigerweb data pull
                bbox = shape.bbox
                points = [(bbox[0], bbox[1]), (bbox[2], bbox[1]),
                          (bbox[2], bbox[3]), (bbox[0], bbox[3]),
                          (bbox[0], bbox[1])]
                esri_json = self.polygon_to_esri_json(points)
            else:
                esri_json = self.polygon_to_esri_json(shape.points)
            if named:
                rec = self.sf.record(ix)
                name = rec[fidx]
                if isinstance(name, str):
                    name = name.lower()
            else:
                name += 1

            if self._filter:
                if name not in self._filter:
                    continue

            self._esri_json[name] = esri_json

            geofeat = shape.__geo_interface__
            if geofeat['type'].lower() == "polygon":
                poly = geojson.Polygon(geofeat['coordinates'])
            else:
                poly = geojson.MultiPolygon(geofeat['coordinates'])

            geofeat = geojson.Feature(geometry=poly)

            self._shapes[name] = geofeat


class TigerWeb(object):
    """
    Method to query data from TigerWeb

    Parameters
    ----------
    shp : str
        shapefile path
    field : str
        shapefile field to id multiple polygons
    radius : float or str
        radius around points to build a query, or shapefile field with
        radius information
    filter : tuple
        tuple of names or polygon numbers to pull from
        default is () which grabs all polygons

    """
    def __new__(cls, shp, field=None, radius=0, filter=()):
        with shapefile.Reader(shp) as sf:
            shapetype = sf.shapeType
            shapename = sf.shapeTypeName

        if shapetype in (1, 11, 21):
            return TigerWebPoint(shp, field, radius, filter)
        elif shapetype in (5, 15, 25):
            return TigerWebPolygon(shp, field, filter)
        else:
            raise TypeError('Shapetype: {}, is not a valid point or polygon'
                            .format(shapename))
