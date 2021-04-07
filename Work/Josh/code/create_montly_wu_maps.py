import os
import numpy as np
from scipy.interpolate import griddata
from CAWSC_WaterUse.etc import utilities as utl
import spatial_outlier_detection as sod
import matplotlib.pyplot as plt

months = (
    "jan_pp_gd",
    "feb_pp_gd",
    "mar_pp_gd",
    "apr_pp_gd",
    "may_pp_gd",
    "jun_pp_gd",
    "jul_pp_gd",
    "aug_pp_gd",
    "sep_pp_gd",
    "oct_pp_gd",
    "nov_pp_gd",
    "dec_pp_gd"
)
miles_to_m = 1.60934 * 1000
miles_list = [25, 50, 100]
# todo: figure out nh issue!!!!
for ix, month in enumerate(months):
    for miles in miles_list:
        # miles = 100
        radius = miles * miles_to_m
        ws = os.path.abspath(os.path.dirname(__file__))
        us_shapefile = os.path.join(ws, "..", "data", "US_nation_2010_alb83.shp")
        wsa_locations = os.path.join(ws, "..", "data", "WSA_v2_1_alb83_attrib.txt")
        wsa_data = os.path.join(ws, "..", "data", "Join_swud_nswud.csv")
        interp_shp = os.path.join(ws, "..", "output", "monthly",
                                  "2010_interp_{}_{}_mi.shp".format(
                                      month.split("_")[0], miles))
        point_shp = os.path.join(ws, "..", "output", "monthly",
                                 "2010_point_{}_{}_mi.shp".format(
                                      month.split("_")[0], miles))
        interp_fig = os.path.join(ws, "..", "output", "monthly",
                                  "2010_{}_{}m_interp.png".format(
                                      month.split("_")[0], miles))
        point_fig = os.path.join(ws, "..", "output", "monthly",
                                 "2010_{}_{}m_1to1.png".format(
                                      month.split("_")[0], miles))

        boundary = utl.load_national_polygons(us_shapefile)

        df = utl.get_input_data(wsa_locations, monthly=True)
        df = utl.get_input_data(wsa_data, df, monthly=True)

        drop_list = []
        for iix, mo in enumerate(months):
            if iix == ix:
                continue
            drop_list.append(mo)
            drop_list.append(utl.MONTHLY[iix])

        df = df.drop(columns=drop_list)
        df = df[df[month] != 0]
        df = df[df[month] <= 1e6]

        df1 = sod.spatial_detect(df, radius, sod.mean_stdev, field=month)
        # todo: ask about NH units, they are probably gallons/day not mgd
        df1 = df1[df1.mean_1 < 1e+6]
        print(df1.std_flg_1.unique())
        print(df1.mean_1.min(), df1.mean_1.max())
        print(df1[month].min(), df1[month].max())
        df1.to_csv(os.path.join(ws, "..", "output", "monthly",
                                "wu_mean_spatial_2010_{}_{}m.csv".format(
                                    month.split("_")[0], miles)),
                   index=False)

        xshape, yshape = 1000, 1000  # 5000, 8000
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
        mask = utl.get_interp_mask(xx, yy, boundary,
                                   multithread=True,
                                   num_threads=8)
        t[~mask] = np.nan
        print("Finished raster intersections")

        utl.array_to_shapefile(interp_shp, {month: t}, xx, yy)
        utl.points_to_shapefile(point_shp, df1)

        _, ax = plt.subplots(figsize=(16, 9))
        quadmesh = utl.plot_map(t, xx, yy, ax=ax)
        plt.colorbar(quadmesh)
        plt.title("2010 {} Mean yearly water use, "
                  "in gallons per person: {} mile radius".format(
                       month.split("_")[0], miles))
        # plt.show()
        plt.savefig(interp_fig)
        plt.close()

        cdict = {0.: 'k', 0.125: 'darkblue',
                 0.25: 'b', 0.5: 'c',
                 1.: 'yellow', 2.: "r",
                 3.: "darkred"}

        _, ax = plt.subplots(figsize=(8, 7))
        for val in sorted(df1.std_flg_1.unique()):
            tdf = df1[df1.std_flg_1 == val]
            label = "stdev {}".format(val)
            color = cdict[val]
            ax = utl.plot_1_to_1(tdf, [month, "mean_1"],
                                 ax=ax, color=color, label=label)

        plt.xlim([0, df1.mean_1.max()])
        plt.ylim([0, df1.mean_1.max()])
        plt.xlabel("SWUD water use, in gallons per person")
        plt.ylabel("{} mean 'neighborhood' water use, {} mile radius".format(
                   month.split("_")[0], miles))
        plt.legend()
        # plt.show()
        plt.savefig(point_fig)
        plt.close()
