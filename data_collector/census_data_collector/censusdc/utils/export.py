import shapefile
import geojson
import numpy as np


def geojson_to_shapefile(f, features):
    """
    Method to write a list of geoJSON features to shapefile

    Parameters
    ----------
    f : str
        shapefile name
    features : list
        list of geoJSON feature classes

    """
    if not f.endswith(".shp"):
        f += ".shp"

    if isinstance(features, geojson.Feature):
        features = [
            features,
        ]

    elif isinstance(features, (list, tuple)):
        pass

    else:
        raise TypeError("{} not supported".format(type(features)))

    fields = {}
    for feature in features:
        for property in feature.properties.keys():
            if property not in fields:
                ftype = type(feature.properties[property])
                if ftype == str:
                    atype = "C"
                else:
                    atype = "N"

                fields[property] = atype

    with shapefile.Writer(f, shapeType=5) as w:
        for k, v in sorted(fields.items()):
            if v == "N":
                w.field(k, v, decimal=2)
            else:
                w.field(k, v)

        for feature in features:
            if feature.geometry.type.lower() == "polygon":
                w.poly(feature.geometry.coordinates)

            elif feature.geometry.type.lower() == "multipolygon":
                poly = []
                for coord in feature.geometry.coordinates:
                    poly.append(coord[0])
                w.poly(poly)

            else:
                raise TypeError(
                    "Unsupported shape type {}".format(feature.geometry.type)
                )

            record = []
            for k in sorted(fields):
                if k in feature.properties:
                    record.append(feature.properties[k])
                else:
                    record.append(0)

            w.record(*record)
