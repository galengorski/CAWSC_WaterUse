import os
import numpy as np
import matplotlib.pyplot as plt
from flopy.plot import styles
from sklearn.metrics import r2_score, mean_squared_error
import joblib

import geopandas
import contextily as cx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import joblib

def show_figure(fig):
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

def save_figure(obj, fn):
    joblib.dump(obj, fn)

def load_figure(fn):
    load_fig = joblib.load(fn)
    show_figure(load_fig[0].figure)

def get_fig_data(fig):
    pass


def one_to_one(y_actual, y_hat, heading, xlabel, ylabel, figfile):
    with styles.USGSPlot():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(y_actual, y_hat, marker="o", s=20, alpha=0.5)
        plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r')
        accuracy = r2_score(y_actual, y_hat)
        rmse = np.power(mean_squared_error(y_actual, y_hat), 0.5)
        styles.heading(ax=ax,
                       heading=heading,
                       idx=0, fontsize=16)
        msg = r"$r^{2}$" + " = {:.2f}".format(np.round(accuracy, 2))
        msg = msg + "\nrmse = {} GPCD".format(np.round(rmse, 2))
        styles.add_text(ax=ax, text=msg,
                        x=0.02, y=0.92, backgroundcolor='white')

        styles.xlabel(ax=ax, fontsize=12, label=xlabel)
        styles.ylabel(ax=ax, fontsize=12, label=ylabel)
        plt.savefig(figfile)
        joblib.dump(fig, figfile+".fig")
        plt.close(fig)



def feature_importance(estimator, max_num_feature, type, figfile):
    from xgboost import plot_importance
    with styles.USGSPlot():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        f = plot_importance(ax=ax, booster=estimator, max_num_features=max_num_feature, importance_type=type)
        styles.heading(ax=ax,
                       heading='Feature Importance',
                       idx=0, fontsize=16)
        styles.xlabel(ax=ax, fontsize=16, label=type)
        styles.ylabel(ax=ax, fontsize=16, label='Feature')
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.savefig(figfile)
        plt.close(fig)
        joblib.dump(fig, figfile + ".fig")


def plot_huc2_map(shp_file, usa_map, info_df, legend_column, log_scale=False, epsg=5070, cmap='cool',
                  title='', figfile=''):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    usa_bkg = geopandas.read_file(usa_map)

    shp_df = geopandas.read_file(shp_file)
    shp_df.to_crs(epsg=epsg, inplace=True)
    usa_bkg.to_crs(epsg=epsg, inplace=True)
    info_df['HUC2'] = info_df['HUC2'].astype(int).astype(str).str.zfill(2)
    shp_df = shp_df[['HUC2', 'geometry']]
    shp_df = shp_df.merge(info_df, how='left', on='HUC2')

    if log_scale:
        norm = matplotlib.colors.LogNorm(vmin=shp_df[legend_column].min(), vmax=shp_df[legend_column].max())
    else:
        norm = None

    with styles.USGSMap():
        ax = shp_df.plot(ax=ax, column=legend_column, alpha=1, cmap=cmap, markersize=5, legend=True,
                         legend_kwds={'shrink': 0.6}, cax=cax, norm=norm)
        usa_bkg.plot(ax=ax, facecolor="none", edgecolor='k', linewidth=0.5)

        styles.heading(ax=ax,
                       heading=title,
                       idx=0, fontsize=16)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()

        plt.savefig(figfile)
        joblib.dump(fig, figfile + ".fig")
        plt.close(fig)
        basename = os.path.splitext(figfile)[0]
        shp_df.to_file(basename + ".shp")


def plot_scatter_map(lon, lat, df, legend_column, cmap, title, figfile, log_scale=False, epsg=5070):
    df_shp = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(lon, lat, crs="EPSG:4326"))  # 2163

    df_shp.to_crs(epsg=epsg, inplace=True)
    legend_kwds = {'label': "Population by Country",
                   'orientation': "horizontal"}
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    legend_kwds = {'label': "Population by Country",
                   'orientation': "horizontal"}
    if log_scale:
        norm = matplotlib.colors.LogNorm(vmin=df_shp[legend_column].min(), vmax=df_shp[legend_column].max())
    else:
        norm = None

    with styles.USGSMap():
        ax = df_shp.plot(ax=ax, column=legend_column, alpha=1, cmap=cmap, markersize=5, legend=True,
                         legend_kwds={'shrink': 0.6}, cax=cax, norm=norm)
        cx.add_basemap(ax, crs=df_shp.crs.to_string(), source=cx.providers.Stamen.TonerLines, alpha=1,
                       attribution=False)
        cx.add_basemap(ax, crs=df_shp.crs.to_string(), source=cx.providers.Stamen.TonerBackground, alpha=0.1,
                       attribution=False)

        styles.heading(ax=ax,
                       heading=title,
                       idx=0, fontsize=16)
        styles.xlabel(ax=ax, fontsize=16, label='X (meter)')
        styles.ylabel(ax=ax, fontsize=16, label='Y (meter)')
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.tight_layout()

        plt.savefig(figfile)
        plt.close(fig)

        basename = os.path.splitext(figfile)[0]
        df_shp.to_file(basename + ".shp")
