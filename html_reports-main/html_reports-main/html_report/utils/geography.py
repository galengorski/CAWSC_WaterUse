import os
import geojson
import shapefile
from shapely.geometry import MultiPolygon, Polygon
from shapely.strtree import STRtree
from .albers import lat_lon_geojson_to_albers_geojson


class Features(object):
    """
    Class that reads and stores features from a shapefile

    Parameters
    ----------
    shp : str
        shapefile path name
    field : str, None
        optional attribute table field to label features
    strtree : bool
        flag to create a ST-Rtree for fast intersection
    to_albers : bool
        flag for conversion from WGS84 to Albers equal area

    """
    def __init__(self, shp, field=None, strtree=False, to_albers=False):
        if not os.path.isfile(shp):
            raise FileNotFoundError("{} not a valid file path".format(shp))
        self._shpname = shp

        if field is not None:
            self._field = field.lower()
        else:
            self._field = field

        self.sf = shapefile.Reader(self._shpname)
        self._shapes = {}
        self._to_albers = to_albers

        self._import_shapefile()

        if strtree:
            shapely_geoms = [self.get_shapely_geometry(name) for
                             name in self.feature_names]
            self.strtree = STRtree(shapely_geoms)
        else:
            self.strtree = None

    @property
    def feature_names(self):
        """
        Method to get all feature names

        Returns
        -------
            list : feature names
        """
        return list(sorted(self._shapes.keys()))

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
        name = name.lower()
        if name not in self._shapes:
            raise KeyError("{}: invalid feature name".format(name))
        else:
            return self._shapes[name]

    def get_shapely_geometry(self, name):
        """
        Method to get a single shapely geometry object for a feature

        Parameters
        ----------
        name : str or int
            feature dictionary key

        Returns
        -------
            shapely.geometry object
        """
        feature = self.get_feature(name)
        geom = self._create_shapely_geoms(feature, name)
        return geom

    def _create_shapely_geoms(self, feature, name):
        """
        Method to set geoJSON features to shapely geometry objects

        Parameter
        ---------
        feature : geojson.Feature object

        Returns
        -------
            shapley.geometry.Polygon or shapely.geometry.MultiPolygon
        """
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

        poly.name = name
        return poly

    def _import_shapefile(self):
        """
        Method to read and store polygons from a shapefile

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

        fields = [field[0].lower() for field in self.sf.fields[1:]]

        name = -1
        for ix, shape in enumerate(self.sf.shapes()):
            shape = self.sf.shape(ix)
            rec = self.sf.record(ix)
            properties = {f: rec[ix] for ix, f in enumerate(fields)}

            if named:
                name = rec[fidx]
                if isinstance(name, str):
                    name = name.lower()
            else:
                name += 1

            geofeat = shape.__geo_interface__
            if geofeat['type'].lower() == "polygon":
                poly = geojson.Polygon(geofeat['coordinates'])
            else:
                poly = geojson.MultiPolygon(geofeat['coordinates'])

            geofeat = geojson.Feature(geometry=poly, properties=properties)

            if self._to_albers:
                geofeat = lat_lon_geojson_to_albers_geojson(geofeat,
                                                            precision=100.)

            self._shapes[name] = geofeat
