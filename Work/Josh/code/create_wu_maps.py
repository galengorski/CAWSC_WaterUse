import os
import numpy as np
from scipy.interpolate import griddata
import utilities as utl
import spatial_outlier_detection as sod
import matplotlib.pyplot as plt


miles_to_m = 1.60934 * 1000
miles_list = [25, 50, 100, 310.686, ]  # [25, 50, 100]
for miles in miles_list:
    # miles = 100
    radius = miles * miles_to_m
    ws = os.path.abspath(os.path.dirname(__file__))
    us_shapefile = os.path.join(ws, "..", "data", "US_nation_2010_alb83.shp")
    wsa_locations = os.path.join(ws, "..", "data", "WSA_v2_1_alb83_attrib.txt")
    wsa_data = os.path.join(ws, "..", "data", "Join_swud_nswud.csv")
    buy_sell_data = os.path.join(ws, "..", "data", "water_exchange_info.csv")
    csv_out = os.path.join(ws, "..", "output",
                           "2010_95p_neutrals_{}m.csv".format(miles))
    interp_shp = os.path.join(ws, "..", "output",
                              "2010_95p_neutrals_interp_{}_mi.shp".format(miles))
    point_shp = os.path.join(ws, "..", "output",
                             "2010_95p_neutrals_point_{}_mi.shp".format(miles))
    interp_fig = os.path.join(ws, "..", "output",
                              "2010_95p_neutrals_interp_{}_mi.png".format(miles))
    point_fig = os.path.join(ws, "..", "output",
                             "2010_95p_neutrals_1to1_{}mi.png".format(miles))

    boundary = utl.load_national_polygons(us_shapefile)

    df = utl.get_input_data(wsa_locations)
    df.drop(columns=["tot_wd_mgd"], inplace=True)
    df = utl.get_input_data(wsa_data, df)
    df = utl.get_input_data(buy_sell_data, df)
    print(len(df))

    df = df[df['ecode'] == "N"]
    print(len(df))

    df1 = sod.spatial_detect(df, radius, [sod.mean_stdev, sod.mean_stdev])
    print(df1.std_flg_1.unique())
    print(df1.mean_1.min(), df1.mean_1.max())
    df1.to_csv(csv_out, index=False)

    # filter the few sites with large water use numbers (most likely unit issue)
    # df1 = df1[df1["wu_pp_gd"] < 1e3]

    for field in ('tot_wd_mgd', "wu_pp_gd", "mean_1", "pop_srv"):
        ax_lst = utl.scatter_plot_with_histograms(df1, (field, "std_flg_1"),
                                                  xbins=100, ybins=7,
                                                  c=df1['std_flg_1'].values,
                                                  cmap='viridis'
                                                  )
        ax_lst[0].set_xlabel(field)
        ax_lst[0].set_ylabel("std_flg_1")
        plt.show()

    for field in ('tot_wd_mgd', 'wu_pp_gd', "mean_1"):
        ax_list = utl.scatter_plot_with_histograms(df1,
                                                   ("pop_srv", field),
                                                   xbins=100,
                                                   ybins=100,
                                                   c=df1['std_flg_1'].values,
                                                   cmap='viridis')
        ax_list[0].set_xlabel("pop_srv")
        ax_list[0].set_ylabel(field)
        plt.show()

    ax_list = utl.scatter_plot_with_histograms(df1,
                                               ("wu_pp_gd", "mean_1"),
                                               xbins=100,
                                               ybins=100,
                                               c=df1['std_flg_1'].values,
                                               cmap='viridis')
    ax_list[0].set_xlabel("wu_pp_gd")
    ax_list[0].set_ylabel("mean_1")
    plt.show()

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

    utl.array_to_shapefile(interp_shp, {"m_wu_gpd": t}, xx, yy)
    utl.points_to_shapefile(point_shp, df1)

    _, ax = plt.subplots(figsize=(16, 9))
    quadmesh = utl.plot_map(t, xx, yy, ax=ax)
    plt.colorbar(quadmesh)
    plt.title("2010 Mean yearly water use, "
              "in gallons per person: {} mile radius".format(miles))
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
        ax = utl.plot_1_to_1(tdf, ["wu_pp_gd", "mean_1"],
                             ax=ax, color=color, label=label)

    plt.xlim([0, df1.mean_1.max()])
    plt.ylim([0, df1.mean_1.max()])
    plt.xlabel("SWUD water use, in gallons per person")
    plt.ylabel("Mean 'neighborhood' water use, {} mile radius".format(miles))
    plt.legend()
    # plt.show()
    plt.savefig(point_fig)
    plt.close()
