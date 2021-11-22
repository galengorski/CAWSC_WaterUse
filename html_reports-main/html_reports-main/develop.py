# from report import Report
from html_report import ReportSection
import pandas as pd
import numpy as np
import datetime as dt
import os
from html_report.utils import Features
import geojson


tmax = np.random.random(365) * 100.
tmin = tmax - 25.
mgal = np.random.random(365) * 100
sdate = dt.datetime(2020, 1, 1)
date = [sdate + dt.timedelta(days=i) for i in range(365)]

d = {'tmax': tmax,
     'tmin': tmin,
     'mgal': mgal,
     "date": date}

df = pd.DataFrame.from_dict(d)

section = ReportSection('Machine learning model report development',
                        "Input data",
                        "Here is a table and figure of input data",
                        include_plotlyjs=True)
section.pandas_to_scatter(df, "date", mode="lines+markers")
section.pandas_to_table(df)
html = section.compile_html((-1, 1))

### map example section
huc_shp_file = os.path.join(".", "data", "huc13_censusdc.shp")

huc12s = Features(huc_shp_file, field="feat_name")

feature_names = []
with open(os.path.join(".", "data", 'abq.txt')) as foo:
    for ix, line in enumerate(foo):
        if ix == 0:
            continue
        else:
            t = line.strip().split(",")
            feature_names.append(t[1])

dd = {'huc12': [], "population": []}
hucs = []
for name in feature_names:
    huc = huc12s.get_feature(name)
    hucs.append(huc)
    dd['huc12'].append(huc.properties["feat_name"])
    dd['population'].append(huc.properties['population'])

hucs = geojson.FeatureCollection(hucs)

df2 = pd.DataFrame.from_dict(dd)

section2 = ReportSection(section="Map development",
                         description="We can add maps of data like these "
                                     "census population data aggregated for "
                                     "HUC12s in 2015",
                         include_plotlyjs=False)
section2.geojson_to_heatmap(hucs, "properties.feat_name", df2.huc12,
                            df2.population, colorscale="Viridis", zmin=0,
                            zmax=100000, marker_opacity=0.5)

html2 = section2.compile_html(layout={'mapbox_style': 'carto-positron',
                                      'mapbox_zoom': 6,
                                      'mapbox_center': {"lat": 35.0844,
                                                        "lon": -106.6504}})
htmls = [html, html2]

html = "".join(htmls)
with open(
        os.path.join("examples",
                     'development_example.html'),
        'w'
         ) as foo:
    foo.write(html)
