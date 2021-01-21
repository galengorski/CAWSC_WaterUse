import os
import numpy as np
from scipy.interpolate import griddata
import utilities as utl
import spatial_outlier_detection as sod
import matplotlib.pyplot as plt


miles_to_m = 1.60934 * 1000
radius = 100 * miles_to_m
ws = os.path.abspath(os.path.dirname(__file__))
us_shapefile = os.path.join(ws, "..", "data", "US_nation_2010_alb83.shp")
wsa_locations = os.path.join(ws, "..", "data", "WSA_v2_1_alb83_attrib.txt")
wsa_data = os.path.join(ws, "..", "data", "Join_swud_nswud.csv")

boundary = utl.load_national_polygons(us_shapefile)

df = utl.get_input_data(wsa_locations)
df.drop(columns=["tot_wd_mgd"], inplace=True)
df = utl.get_input_data(wsa_data, df)
df1 = sod.spatial_detect(df, radius, sod.mean_stdev)
print(df1.std_flg_1.unique())
df1.to_csv(os.path.join(ws, "..", "data", "wu_mean_spatial_2010.csv"),
           index=False)

xshape, yshape = 1500, 3000
xv = df1.x_centroid.values
yv = df1.y_centroid.values
points = list(zip(xv, yv))

minx = np.min(xv)
maxx = np.max(xv)
miny = np.min(yv)
maxy = np.max(yv)

xpts = np.linspace(minx, maxx + 1, num=xshape)
ypts = np.linspace(miny, maxy + 1, num=yshape)
xx, yy = np.meshgrid(xpts, ypts)
xpts = xx.ravel()
ypts = yy.ravel()

values = df1.mean_1.values

t = griddata((xv, yv), values, (xpts, ypts), method="linear")
t.shape = (yshape, xshape)

print("Starting raster intersections")
for poly in boundary:
    mask = utl.point_in_polygon(xx, yy, list(poly))
    t[~mask] = np.nan
print("Finished raster intersections")

dx = abs(xx[0, 0] - xx[0, 1])
dy = abs(yy[0, 0] - yy[1, 0])
xxv = np.arange(minx - dx/2, maxx + 1 + dx/2, dx)
yyv = np.arange(miny - dy/2, maxy + 1 + dy/2, dy)

xxv, yyv = np.meshgrid(xxv, yyv)

print('break')
plt.pcolormesh(xxv, yyv, t)
plt.scatter(xv, yv,)  #  'bo')
plt.colorbar()
plt.show()

