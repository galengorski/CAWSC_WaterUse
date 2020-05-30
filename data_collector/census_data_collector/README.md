# Census Data Collector  
The census data collector is a geographic based tool to query census data from
the TigerWeb REST services and the US Census API. Queryable census products
 include:  
* TigerWeb
* Decenial Census data from Sf3
* ACS 1-Year
* ACS 5-Year

## Installation
_Method 1_: Download the census data collector from 
https://github.com/jlarsen-usgs/census-data-collector navigate to the package 
in either an anaconda prompt, terminal, or command prompt window and 
 run:
```shell script
pip install .
```

_Method 2_: Install directly from Github using"
```shell script
pip install https://github.com/jlarsen-usgs/census-data-collector/zipball/master
```

## Basic usage
The census data collector uses geographic information from shapefiles to query 
data.

__Importing census data collector__
```python
import censusdc
```

__Grabbing features from TigerWeb__  
TigerWeb hosts feature layers that are tagged with geographic id information
that can be used for creating census api data pulls. The `TigerWeb` class 
accepts polygon and point shapefiles and can query a host of 
geographic information for the census.  

*__Example 1__*:  
_Sacramento neighborhood polygons:_  
Two polygons have been drawn; one for the Tahoe Park and one for the La Riviera
 neighborhood  
<p align="center">
  <img src="https://raw.githubusercontent.com/jlarsen-usgs/census-data-collector/master/data/Sacramento_neighborhoods.png" alt="Sacto"/>
</p>

We will use this shapefile and these polygon features to query block
information for the 2010 census
```python
from censusdc import TigerWeb
from censusdc import TigerWebVariables as TWV
import os

shp_file = os.path.join('data','Sacramento_neighborhoods.shp')

# if the shapefile has a label field for polygons we can tag data 
# using the field parameter
tigweb = TigerWeb(shp_file, field="name")

# if you have only specific fields that you would like
tigweb.get_data(2010, outfields=(TWV.geoid, TWV.state, TWV.county,
                                 TWV.tract, TWV.blkgrp, TWV.block))

# default method gets all relevant tigerweb attributes
tigweb.get_data(2010)

features = tigweb.features
```
The features parameter returns a dictionary of geoJSON objects that can be
accessed and exported.  

To get the features from a single polygon we can use
```python
# get all polygon names
names = tigweb.feature_names

# get all GeoJSON features associated with a single polygon
feature = tigweb.get_feature("la_riviera")
```
 and we can visualize the geoJSON features using Descartes and matplotlib
 ```python
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import numpy as np
import utm

fig = plt.figure()
ax = fig.gca()
for name in tigweb.feature_names:
    for feature in tigweb.get_feature(name):
        ax.add_patch(PolygonPatch(feature.geometry, alpha=0.5))
    
    # get the input polygon shapes and convert from UTM to WGS84
    x, y = np.array(tigweb.get_shape(name)).T
    y, x = utm.to_latlon(x, y, 11, zone_letter='N')
    ax.plot(x, y, 'r-')

ax.axis('scaled')
plt.show()
```
<p align="center">
  <img src="https://raw.githubusercontent.com/jlarsen-usgs/census-data-collector/master/data/Tigerweb_example.png" alt="TigerWeb"/>
</p>

*__Example 2__*:  
_Sacramento neighborhood points:_  
Point shapefiles can also be supplied to `TigerWeb`. Here we have on in the
Tahoe Park neighborhood and one in the La Riviera neighborhood
<p align="center">
  <img src="https://raw.githubusercontent.com/jlarsen-usgs/census-data-collector/master/data/Sacramento_points.png" alt="Sacto_pts"/>
</p>

We can either use the points by themselves or define a radius around the 
points to query data from. The `radius=` parameter accepts either a string
which references an attribute column in the point shapefile or a float that
applies a constant radius to all points.   
```python
from censusdc import TigerWeb
from censusdc import TigerWebVariables as TWV
import os

shp_file = os.path.join('data','Sacramento_points.shp')

# radius infromation must be in the same units as the shapefile projection!
tigweb = TigerWeb(shp_file, field="name", radius="radius")
tigweb.get_data(2013, level='tract')
```
and here we can visualize the census block group features within the 
defined radius from our points
```python
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import numpy as np
import utm

fig = plt.figure()
    ax = fig.gca()
    for name in tigweb.feature_names:
        for feature in tigweb.get_feature(name):
            ax.add_patch(PolygonPatch(feature.geometry, alpha=0.5))

        x, y = np.array(tigweb.get_shape(name)).T
        y, x = utm.to_latlon(x, y, 11, zone_letter='N')
        px, py = np.array(tigweb.get_point(name))
        py, px = utm.to_latlon(px, py, 11, zone_letter='N')
        ax.plot(x, y, 'r-')
        ax.plot(px, py, 'ro')

    ax.axis('scaled')
    plt.show()
```
<p align="center">
  <img src="https://raw.githubusercontent.com/jlarsen-usgs/census-data-collector/master/data/Tigerweb_points_example.png" alt="TigerWeb_pts"/>
</p>

__*Using tigerweb features to grab census data*__

After we get a feature set from tigerweb, we can grab data. In this example we 
are getting data from the 2013 American Community Survey 5 year data set and 
plot population of census block groups. Data from the Census data pull is 
added to each GeoJSON feature to keep spatial correlation. Here is an example:

```python
from censusdc import TigerWeb, Acs5
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import os
import utm

# the user must have a census api key to pull data from the Acs5
with open("api_key.dat") as api:
    apikey = api.readline().strip()

data = 'data'
point_name = "Sacramento_points.shp"

tigweb = TigerWeb(os.path.join(data, point_name), field='name',
                  radius='radius')
tigweb.get_data(2013, level='tract')

# get ACS5 data
acs = Acs5(tigweb.features, 2013, apikey)
acs.get_data(retry=20)  # number of retries for connection issues

fig = plt.figure()
ax = fig.gca()
population = []
patches = []
for name in acs.feature_names:
    for feature in acs.get_feature(name):
        patches.append(PolygonPatch(feature.geometry))
        population.append(feature.properties["B01003_001E"])
    x, y = np.array(tigweb.get_shape(name)).T
    y, x = utm.to_latlon(x, y, 11, zone_letter='N')
    px, py = np.array(tigweb.get_point(name))
    py, px = utm.to_latlon(px, py, 11, zone_letter='N')
    ax.plot(x, y, 'r-')
    ax.plot(px, py, 'ro')

p = PatchCollection(patches, cmap="viridis", alpha=0.75)
p.set_array(np.array(population))
ax.add_collection(p)
ax.axis('scaled')
plt.colorbar(p, shrink=0.7)
plt.show()
```
<p align="center">
  <img src="https://raw.githubusercontent.com/jlarsen-usgs/census-data-collector/master/data/Tigerweb_points_population.png" alt="Acs5_pop_2013"/>
</p>

__*Intersecting and area weight adjustment of geoJSON features*__

The `GeoFeature` class allows the user to intersect census designated place 
data with arbitrary polygons to create new area weighted features that 
represent the user's area of interest. Input polygons can be supplied to 
the `GeoFeature` class as a pyshp `shapefile.Reader` instance, a list
`shapefile.Shape` instances, a list of `shapely.geometry.Polygon` or
 `shapely.geometry.MultiPolygon` instances, or a list of x,y points that define
 a closed polygon.
 
 _Note: shapes must be supplied in the WGS84 projection to intersect with the 
 TigerLine data, as this is the projection returned from TigerWeb_ 
 
 Example:
 ```python
# building off of the previous example...

from censusdc.utils import GeoFeatures
import shapefile


ishp_name = "multipolygon_test.shp"
ishp = shapefile.Reader(os.path.join(data, ishp_name))
gf = GeoFeatures(acs.get_feature(acs.feature_names[0]))
gf.intersect(ishp)

    fig = plt.figure()
    ax = fig.gca()
    population = []
    patches = []
    census_patches = []
    for feature in acs.get_feature(acs.feature_names[0]):
        census_patches.append(PolygonPatch(feature.geometry))

    for feature in gf.intersected_features:
        patches.append(PolygonPatch(feature.geometry))
        population.append(feature.properties["B01003_001E"])

    p = PatchCollection(patches, cmap="viridis", alpha=0.75)
    cp = PatchCollection(census_patches, cmap='spring')
    p.set_array(np.array(population))
    ax.add_collection(cp)
    ax.add_collection(p)
    ax.axis('scaled')
    plt.colorbar(p, shrink=0.7)
    plt.show()
```
<p align="center">
  <img src="https://raw.githubusercontent.com/jlarsen-usgs/census-data-collector/master/data/Tigerweb_points_intersection.png" alt="Acs5_intersect_2013"/>
</p>

__*Using the `CensusTimeSeries` class to get a pandas dataframe of timeseries data*__  

Instead of requesting data year by year, the `CensusTimeSeries` class allows 
the user to pull multiple timeseries of data from the US Census in a few simple
calls. The default parameters the `CensusTimeSeries` class pulls are defined
in the `AcsVariables`, `Sf3Variables`, and `Sf3Variables1990` classes. The 
`CensusTimeSeries` object uses data caching methods to pull all census data
within a shapefile defined region and allow the user to do multiple time
series with that data.

_Basic example_
```python
import os
import shapefile
from censusdc.utils import CensusTimeSeries

ws = os.path.abspath(os.path.dirname(__file__))
shp = os.path.join(ws, "data", "Sacramento_neighborhoods_WGS.shp")
apikey = os.path.join(ws, "api_key.dat")

with open(apikey) as api:
    apikey = api.readline().strip()

ts = CensusTimeSeries(shp, apikey, field="name")

shp = shapefile.Reader(shp)
polygon = shp.shape(0)
df = ts.get_timeseries("la_riviera", polygons=polygon)

polygon = shp.shape(1)
df1 = ts.get_timeseries("tahoe_park", polygons=polygon)
```
## Example Jupyter Notebooks
Fully functioning example notebooks can be found [here](https://github.com/jlarsen-usgs/census-data-collector/tree/master/examples)
## Development
This project is in active development and is in the pre-alpha stages. There 
will be many updates and changes to the source code in the near future.

## Authors
Joshua Larsen