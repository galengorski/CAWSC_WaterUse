import numpy as np
import pandas as pd
import threading
from shapely.geometry import Polygon, MultiPolygon
from shapely.strtree import STRtree
from shapely.validation import explain_validity
from . import thread_count
import platform
import geojson
import shapefile
import copy

if platform.system().lower() != "windows":
    import ray
else:
    # fake ray wrapper function for windows
    from ..utils import ray


def _IGNORE():
    from ..datacollector.tigerweb import TigerWebVariables
    from ..datacollector.acs import AcsVariables
    from ..datacollector.dec import Sf3Variables, Sf3Variables1990

    return (
        TigerWebVariables.mtfcc,
        TigerWebVariables.oid,
        TigerWebVariables.geoid,
        TigerWebVariables.state,
        TigerWebVariables.county,
        TigerWebVariables.cousub,
        TigerWebVariables.tract,
        TigerWebVariables.blkgrp,
        TigerWebVariables.basename,
        TigerWebVariables.name,
        TigerWebVariables.lsadc,
        TigerWebVariables.funcstat,
        TigerWebVariables.arealand,
        TigerWebVariables.areawater,
        TigerWebVariables.stgeometry,
        TigerWebVariables.centlat,
        TigerWebVariables.centlon,
        TigerWebVariables.intptlat,
        TigerWebVariables.intptlon,
        TigerWebVariables.objectid,
        TigerWebVariables.population,
        AcsVariables.median_income,
        Sf3Variables1990.median_income,
        Sf3Variables.median_income,
        "pop_density",
    )


def _AVERAGE():
    from ..datacollector.acs import AcsVariables
    from ..datacollector.dec import Sf3Variables, Sf3Variables1990

    return (
        AcsVariables.median_income,
        Sf3Variables1990.median_income,
        Sf3Variables.median_income,
        "pop_density",
    )


def _POPULATION():
    from ..datacollector.acs import AcsVariables
    from ..datacollector.dec import Sf3Variables, Sf3Variables1990

    return (
        Sf3Variables.population,
        AcsVariables.population,
        Sf3Variables1990.population,
    )


class GeoFeatures(object):
    """
    Class that facilitates the intersection and adjustment
    of geoJSON geometries with arbitrary polygons

    Parameters
    ----------
    features : list
        list of geoJSON features

    """

    def __init__(self, features, name=None):

        self.__name = name
        self._features = features
        self._shapely_features = []
        self._ifeatures = None

        self._create_shapely_geoms()
        self.IGNORE = _IGNORE()
        self.POPULATION = _POPULATION()

        self._strtree = None
        if len(self._shapely_features) > 100:
            self._strtree = STRtree(self._shapely_features)

    def _create_shapely_geoms(self):
        """
        Method to set geoJSON features to shapely geometry objects
        """
        for ix, feature in enumerate(self._features):
            # coords, utm_zone = geoJSON_lat_lon_to_utm(feature)
            if feature.geometry.type == "MultiPolygon":
                polys = []
                for coordinates in feature.geometry.coordinates:
                    if len(coordinates) > 1:
                        coords = coordinates[0]
                        holes = coordinates[1:]
                        polys.append(Polygon(coords, holes=holes))
                    else:
                        coords = coordinates[0]
                        polys.append(Polygon(coords))

                poly = MultiPolygon(polys)

            else:
                coordinates = feature.geometry.coordinates
                if len(coordinates) > 1:
                    coords = coordinates[0]
                    holes = coordinates[1:]
                    poly = Polygon(coords, holes=holes)
                else:
                    poly = Polygon(coordinates[0])

            if not poly.is_valid:
                poly = poly.buffer(0)

            poly.ix = ix

            self._shapely_features.append(poly)

    @property
    def name(self):
        return self.__name

    @property
    def features(self):
        return copy.deepcopy(self._features)

    @property
    def intersected_features(self):
        return copy.deepcopy(self._ifeatures)

    def intersect(
        self,
        polygons,
        verbose=False,
        multiproc=False,
        multithread=False,
        thread_pool=4,
    ):
        """
        Intersection method that creates a new dictionary of geoJSON
        features. input polygons must be provided in WGS84 (decimal lat lon.)

        Parameters
        ----------
        polygons : list
            list of shapely polygons, list of shapefile.Shape objects,
            shapefile.Reader object, shapefile.Shape object, or
            list of xy points [[(lon, lat)...(lon_n, lat_n)],[...]]
        verbose : bool
            boolean flag for verbosity
        multiproc : bool
            flag for multiprocessing with ray on linux machines
        multithread : bool
            flag to enable multithreaded operations
        thread_pool : int
            number of threads to use for multi-threaded operations

        """
        self._ifeatures = None

        flag = ""

        if isinstance(polygons, shapefile.Reader):
            polygons = [shape for shape in polygons.shapes()]
            flag = "shapefile"

        elif isinstance(polygons, list):
            if isinstance(polygons[0], shapefile.Shape):
                flag = "shapefile"

            elif isinstance(polygons[0], Polygon):
                flag = "shapely"

            elif isinstance(polygons[0], (list, tuple)):
                if isinstance(polygons[0][0], (float, np.float_, int)):
                    polygons = [polygons]
                elif isinstance(polygons[0][0], (list, tuple)):
                    if isinstance(polygons[0][0][0], (float, np.float_, int)):
                        pass
                    elif isinstance(polygons[0][0][0], (list, tuple)):
                        raise IndexError(
                            "Polygons must be in a 2 or 3 " "dimensional list"
                        )
                    else:
                        raise ValueError(
                            "Polygon indicies must be float or "
                            "int type values"
                        )
                else:
                    raise IndexError(
                        "Polygons must be in a 2 or 3 dimensional list"
                    )

                flag = "list"

            elif isinstance(polygons[0], geojson.Feature):
                flag = "geojson"

        elif isinstance(polygons, shapefile.Shape):
            polygons = [
                polygons,
            ]
            flag = "shapefile"

        elif isinstance(polygons, (Polygon, MultiPolygon)):
            polygons = [
                polygons,
            ]
            flag = "shapely"

        elif isinstance(polygons, (geojson.Feature)):
            polygons = [
                polygons,
            ]
            flag = "geojson"

        else:
            raise TypeError("{}: not yet supported".format(type(polygons)))

        if flag == "shapefile":
            t = []
            for shape in polygons:

                shape_type = shape.__geo_interface__["type"]
                coords = shape.points
                if shape_type.lower() == "polygon":
                    t.append(
                        Polygon(coords),
                    )

                elif shape_type.lower() == "multipolygon":
                    parts = list(shape.parts)
                    for ix in range(1, len(parts)):
                        i0 = parts[ix - 1]
                        i1 = parts[ix]
                        t.append(Polygon(coords[i0:i1]))

                        if len(parts) == ix + 1:
                            t.append(Polygon(coords[i1:]))

                        else:
                            pass

                else:
                    raise NotImplementedError(
                        "{} intersection is not supported".format(shape_type)
                    )

            polygons = t

        elif flag == "shapely":
            t = []
            for shape in polygons:
                if isinstance(shape, Polygon):
                    t.append(Polygon)

                elif isinstance(shape, MultiPolygon):
                    for geom in shape.geoms:
                        if isinstance(geom, Polygon):
                            t.append(geom)
                        else:
                            continue

                else:
                    raise NotImplementedError("{}".format(shape.geom_type))

            polygons = t

        elif flag == "list":
            t = []
            for shape in polygons:
                t.append(Polygon(shape))

            polygons = t

        elif flag == "geojson":
            # preferred data type!
            t = []
            for shape in polygons:
                if shape.geometry.type.lower() == "polygon":
                    if len(shape.geometry.coordinates) > 1:
                        # handle holes
                        p = Polygon(
                            shape.geometry.coordinates[0],
                            shape.geometry.coordinates[1:],
                        )
                    else:
                        p = Polygon(shape.geometry.coordinates[0])

                    t.append(p)

                elif shape.geometry.type.lower() == "multipolygon":
                    for coords in shape.geometry.coordinates:
                        if len(coords) > 1:
                            # handle holes
                            p = Polygon(coords[0], coords[1:])
                        else:
                            p = Polygon(coords[0])

                        t.append(p)

                else:
                    raise TypeError(
                        "Only multipolygon and polygon "
                        "features are supported"
                    )

            polygons = t

        else:
            raise Exception("Code shouldn't have made it here!")

        # repair self-intersections
        t = []
        for p in polygons:
            if not p.is_valid:
                if verbose:
                    print(
                        "Repairing Geometry {}".format(self.__name),
                        ":",
                        explain_validity(p),
                    )
                p = p.buffer(0.0)
                if isinstance(p, MultiPolygon):
                    p = list(p)
                else:
                    p = [
                        p,
                    ]
            else:
                p = [
                    p,
                ]

            for poly in p:
                t.append(poly)

        polygons = t

        if multiproc and platform.system().lower() == "windows":
            multiproc = False
            multithread = True
            thread_pool = thread_count() - 1

        if multiproc:
            fid = ray.put(self.features)
            actors = []
            n = 0
            shapely_features = self._shapely_features
            for polygon in polygons:
                if self._strtree is not None:
                    t = self._strtree.query(polygon)
                    ixs = sorted([i.ix for i in t])
                    shapely_features = [self._shapely_features[i] for i in ixs]

                for feature in shapely_features:
                    actor = multiproc_intersection.remote(
                        self.IGNORE,
                        self.POPULATION,
                        fid,
                        polygon,
                        feature.ix,
                        feature,
                        n,
                    )
                    actors.append(actor)
                    n += 1

            output = ray.get(actors)

            for out in output:
                if out and out is not None:
                    for record in out:
                        n, m, geofeature = record
                        if self._ifeatures is None:
                            self._ifeatures = {
                                "{}_{}".format(n, m): geofeature
                            }
                        else:
                            self._ifeatures["{}_{}".format(n, m)] = geofeature

        elif multithread:

            thread_list = []
            container = threading.BoundedSemaphore(thread_pool)
            n = 0
            shapely_features = self._shapely_features
            for polygon in polygons:
                if self._strtree is not None:
                    t = self._strtree.query(polygon)
                    ixs = sorted([i.ix for i in t])
                    shapely_features = [self._shapely_features[i] for i in ixs]

                for feature in shapely_features:
                    x = threading.Thread(
                        target=self.__threaded_intersection,
                        args=(polygon, feature.ix, feature, n, container),
                    )
                    thread_list.append(x)
                    n += 1

            for thread in thread_list:
                thread.start()
            for thread in thread_list:
                thread.join()

        else:
            n = 0
            shapely_features = self._shapely_features
            for polygon in polygons:
                if self._strtree is not None:
                    t = self._strtree.query(polygon)
                    ixs = sorted([i.ix for i in t])
                    shapely_features = [self._shapely_features[i] for i in ixs]

                for feature in shapely_features:
                    self.__intersection(polygon, feature.ix, feature, n)
                    n += 1

        if self._ifeatures is not None:
            self._ifeatures = [v for k, v in self._ifeatures.items()]
        else:
            self._ifeatures = []

    def __intersection(self, polygon, ix, feature, n):
        """
        Intersection method for threaded and serial function calls

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
        ix : int
            enumeration number
        feature : shapely.geometry.Polygon
            feature
        n : int
            counter

        Returns
        -------

        """
        properties = self.features[ix].properties

        a = polygon.intersection(feature)

        if a.geom_type == "MultiPolygon":
            p = list(a)

        elif a.geom_type == "GeometryCollection":
            t = list(a)
            p = []
            for shape in t:
                if shape.geom_type == "Polygon":
                    p.append(shape)
                elif shape.geom_type == "MultiPolygon":
                    for sh in list(shape):
                        p.append(sh)

                else:
                    pass

        else:
            p = [
                a,
            ]

        m = 0
        for a in p:
            area = a.area
            if area == 0:
                continue
            geoarea = feature.area
            ratio = area / geoarea

            adj_properties = {}
            pop = 0
            for k, v in properties.items():
                if k in self.IGNORE:
                    adj_properties[k] = v
                else:
                    if k in self.POPULATION:
                        pop = v * ratio
                    try:
                        adj_properties[k] = v * ratio
                    except TypeError:
                        adj_properties[k] = v
                        print("DEBUG NOTE: ", k, v)

            if pop > 0:
                adj_properties["pop_density"] = pop / area

            xy = np.array(a.exterior.xy, dtype=float).T
            xy = [(i[0], i[1]) for i in xy]

            geopolygon = geojson.Polygon([xy])
            geofeature = geojson.Feature(
                geometry=geopolygon, properties=adj_properties
            )

            if self._ifeatures is None:
                self._ifeatures = {"{}_{}".format(n, m): geofeature}
            else:
                self._ifeatures["{}_{}".format(n, m)] = geofeature
            m += 1

    def __threaded_intersection(self, polygon, ix, feature, n, container):
        """
        Multithreaded intersection operation handler

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
        ix : int
            enumeration number
        feature : shapely.geometry.Polygon
            feature
        n : int
            counter
        container : threading.BoundedSemaphore

        """
        container.acquire()
        self.__intersection(polygon, ix, feature, n)
        container.release()

    @staticmethod
    def features_to_dataframe(year, features, hr_dict=None):
        """
        Method to take a group of features and accumulate values from
        that dataframe into a single record.

        Parameters
        ----------
        year : int
        features : dict
            geoJSON features
        hr_dict : dict
            human readable column labels for census fields

        Returns
        -------
            pd.DataFrame

        """
        IGNORE = _IGNORE()
        AVERAGE = _AVERAGE()
        d = {}
        for feature in features:
            props = feature.properties
            for prop, value in props.items():
                if prop in IGNORE:
                    if prop in AVERAGE:
                        if prop in d:
                            d[prop].append(value)
                        else:
                            d[prop] = [
                                value,
                            ]
                else:
                    if prop in d:
                        d[prop] += value
                    else:
                        d[prop] = value

        for prop in AVERAGE:
            if prop in d:
                d[prop] = np.nanmean(d[prop])

        outdic = dict()
        outdic["year"] = [
            year,
        ]
        if hr_dict is not None:
            keys = list(d.keys())
            for key in keys:
                try:
                    new_key = hr_dict[key]
                    outdic[new_key] = [
                        d[key],
                    ]
                except KeyError:
                    outdic[key] = [
                        d[key],
                    ]
        else:
            for key, value in d.items():
                outdic[key] = [
                    value,
                ]

        df = pd.DataFrame.from_dict(outdic)
        return df

    @staticmethod
    def compiled_feature(
        year, polygon, feature_name, df=None, features=None, hr_dict=None
    ):
        """
        Method to compile intersected features attributes with a supplied
        polygon and create a new geoJSON feature

        User must supply either a compiled dataframe from
        GeoFeatures.features_to_dataframe or a dictionary of geoJSON feature
        objects

        Parameters
        ----------
        year : int
            census year
        polygon : list, shapefile.Shape, shapely.geometry.Polygon,
                  shapely.Geometry.MultiPolygon, geojson.Feature,
                  geojson.Polygon, geojson.MultiPolygon
            feature to compile and set attributes to
        feature_name : str, int
            polygon identifier, recommend using feature_name from tw
        df : pd.DataFrame
            optional: output from feature_to_dataframe method
        features : dict
            geoJSON features
        hr_dict : dict
            human readable column labels for census fields

        Returns
        -------
            geoJSON feature object

        """
        if df is None and features is None:
            raise AssertionError(
                "User must supply a dataframe or " "geoJSON features"
            )
        elif df is None:
            df = GeoFeatures.features_to_dataframe(year, features, hr_dict)
        else:
            pass

        if isinstance(polygon, list):
            if isinstance(polygon[0], (tuple, list)):
                if isinstance(polygon[0][0], (float, np.float_, int)):
                    polygon = [polygon]
                elif isinstance(polygon[0][0], (list, tuple)):
                    if isinstance(polygon[0][0][0], (float, np.float_, int)):
                        pass
                    elif isinstance(polygon[0][0][0], (list, tuple)):
                        raise IndexError(
                            "Polygons must be in a 2 or 3 " "dimensional list"
                        )
                    else:
                        raise ValueError(
                            "Polygon indicies must be float or "
                            "int type values"
                        )
                else:
                    raise IndexError(
                        "Polygons must be in a 2 or 3 dimensional list"
                    )

                if len(polygon) > 1:
                    t = []
                    for p in polygon:
                        t.append((p,))
                    polygon = geojson.MultiPolygon(t)
                else:
                    polygon = geojson.Polygon(polygon)

            else:
                err = "Only single polygons/multipolygons are supported"
                raise NotImplementedError(err)

        elif isinstance(polygon, shapefile.Shape):
            shape_type = polygon.__geo_interface__["type"]
            coords = polygon.points
            if shape_type.lower() == "polygon":
                polygon = geojson.Polygon([coords])

            elif shape_type.lower() == "multipolygon":
                t = []
                parts = list(polygon.parts)
                for ix in range(1, len(parts)):
                    i0 = parts[ix - 1]
                    i1 = parts[ix]
                    t.append((coords[i0:i1],))

                    if len(parts) == ix + 1:
                        t.append((coords[i1:],))

                    else:
                        pass

                polygon = geojson.MultiPolygon(t)

        elif isinstance(polygon, (Polygon, MultiPolygon)):

            if isinstance(polygon, Polygon):
                x, y = polygon.exterior.xy
                coords = list(zip(x, y))
                polygon = geojson.Polygon([coords])
            else:
                t = []
                for geom in polygon.geoms:
                    if isinstance(geom, Polygon):
                        x, y = geom.exterior.xy
                        coords = list(zip(x, y))
                        t.append((coords,))

                polygon = geojson.MultiPolygon(polygon)

        elif isinstance(polygon, geojson.Feature):
            if isinstance(polygon, geojson.Feature):
                polygon = polygon.geometry

        else:
            err = (
                "Input type not yet implemented: Method currently "
                "supports (Shapely, Shapefile.shape, and lists of verticies"
            )
            raise NotImplementedError(err)

        cols = list(df)

        properties = {"feat_name": feature_name}
        if "year" in cols:
            df = df[df.year == year]

        for c in cols:
            properties[c] = df[c].values[0]

        new_feat = geojson.Feature(geometry=polygon, properties=properties)
        return new_feat


@ray.remote
def multiproc_intersection(
    IGNORE, POPULATION, features, polygon, ix, feature, n
):
    """
    Multithreaded intersection operation handler

    Parameters
    ----------
    IGNORE : list
        list of keys to ignore
    POPULATION : list
        list of potential population keys to calc. pop density
    features : ray.put() object
        ray put object of features
    polygon : shapely.geometry.Polygon
    ix : int
        enumeration number
    feature : shapely.geometry.Polygon
        feature
    n : int
        counter

    """
    out = []

    properties = features[ix].properties

    a = polygon.intersection(feature)

    if a.geom_type == "MultiPolygon":
        p = list(a)

    elif a.geom_type == "GeometryCollection":
        t = list(a)
        p = []
        for shape in t:
            if shape.geom_type == "Polygon":
                p.append(shape)
            elif shape.geom_type == "MultiPolygon":
                for sh in list(shape):
                    p.append(sh)

            else:
                pass

    else:
        p = [
            a,
        ]

    m = 0
    for a in p:
        area = a.area
        if area == 0:
            continue
        geoarea = feature.area
        ratio = area / geoarea

        adj_properties = {}
        pop = 0
        for k, v in properties.items():
            if k in IGNORE:
                adj_properties[k] = v
            else:
                if k in POPULATION:
                    pop = v * ratio
                try:
                    adj_properties[k] = v * ratio
                except TypeError:
                    adj_properties[k] = v
                    print("DEBUG NOTE: ", k, v)

        if pop > 0:
            adj_properties["pop_density"] = pop / area

        xy = np.array(a.exterior.xy, dtype=float).T
        xy = [(i[0], i[1]) for i in xy]

        geopolygon = geojson.Polygon([xy])
        geofeature = geojson.Feature(
            geometry=geopolygon, properties=adj_properties
        )

        out.append([n, m, geofeature])
        m += 1

    return out
