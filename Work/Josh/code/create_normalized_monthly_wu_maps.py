import os
import numpy as np
from scipy.interpolate import griddata
from CAWSC_WaterUse.etc import utilities as utl
import spatial_outlier_detection as sod
import matplotlib.pyplot as plt


months = (
    "jan_norm",
    "feb_norm",
    "mar_norm",
    "apr_norm",
    "may_norm",
    "jun_norm",
    "jul_norm",
    "aug_norm",
    "sep_norm",
    "oct_norm",
    "nov_norm",
    "dec_norm",
)

miles_to_m = 1.60934 * 1000
miles_list = [
    25,
    50,
    100,
    310.686,
]
for ix, month in enumerate(months):
    for miles in miles_list:
        # miles = 100
        radius = miles * miles_to_m
        ws = os.path.abspath(os.path.dirname(__file__))
        us_shapefile = os.path.join(
            ws, "..", "data", "US_nation_2010_alb83.shp"
        )
        wsa_locations = os.path.join(
            ws, "..", "data", "WSA_v2_1_alb83_attrib.txt"
        )
        wsa_data = os.path.join(ws, "..", "data", "Join_swud_nswud.csv")
        buy_sell_data = os.path.join(
            ws, "..", "data", "water_exchange_info.csv"
        )
        interp_shp = os.path.join(
            ws,
            "..",
            "output",
            "monthly",
            "2010_interp_normalized_{}_{}m.shp".format(
                month.split("_")[0], miles
            ),
        )
        point_shp = os.path.join(
            ws,
            "..",
            "output",
            "monthly",
            "2010_point_{}_normalized_{}m.shp".format(
                month.split("_")[0], miles
            ),
        )
        interp_fig = os.path.join(
            ws,
            "..",
            "output",
            "monthly",
            "2010_{}_normalized_{}m.png".format(month.split("_")[0], miles),
        )

        boundary = utl.load_national_polygons(us_shapefile)

        df = utl.get_input_data(wsa_locations, monthly=True)
        df = utl.get_input_data(wsa_data, df, monthly=True, normalized=True)
        df = utl.get_input_data(buy_sell_data, df)

        df = df[df["ecode"] == "N"]

        df = sod.spatial_detect(
            df, radius, [sod.mean_stdev, sod.mean_stdev], field=month
        )
        df.to_csv(
            os.path.join(
                ws,
                "..",
                "output",
                "monthly",
                "wu_mean_spatial_2010_normalized.csv",
            ),
            index=False,
        )

        drop_list = []
        for ixx, m in enumerate(months):
            if ixx == ix:
                continue
            else:
                drop_list.append(m)

        df = df.drop(columns=drop_list)

        xshape, yshape = 1000, 1000  # 5000, 8000
        xv = df.x_centroid.values
        yv = df.y_centroid.values
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

        values = df.mean_1.values

        t = griddata((xv, yv), values, (xpts, ypts), method="linear")
        t.shape = (yshape, xshape)

        print("Starting raster intersections")
        mask = utl.get_interp_mask(
            xx, yy, boundary, multithread=True, num_threads=8
        )
        t[~mask] = np.nan
        print("Finished raster intersections")

        utl.array_to_shapefile(interp_shp, {month: t}, xx, yy)
        utl.points_to_shapefile(point_shp, df)

        _, ax = plt.subplots(figsize=(16, 9))
        quadmesh = utl.plot_map(t, xx, yy, ax=ax, vmin=0.0, vmax=1.0)
        plt.colorbar(quadmesh)
        plt.title(
            "2010 {} Normalized water use, {} mile "
            "radius".format(month.split("_")[0], miles)
        )
        # plt.show()
        plt.savefig(interp_fig)
        plt.close()
