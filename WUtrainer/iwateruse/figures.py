import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flopy.plot import styles
from sklearn.metrics import r2_score, mean_squared_error
import joblib

import geopandas
import contextily as cx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import joblib

import seaborn as sns
import statsmodels.api as sm

from statsmodels.graphics.gofplots import ProbPlot

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
        plt.tight_layout()
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
        plt.tight_layout()
        plt.savefig(figfile)
        plt.close(fig)
        joblib.dump(fig, figfile + ".fig")


def plot_timeseries_per_huc2(df, x, y, figfile):
    nrows = 3
    ncols = 6
    fig, axes = plt.subplots(ncols, nrows, figsize=(13.33, 7.5), sharex=True)
    huc2s = df['HUC2'].unique()
    n = 0

    for h in huc2s:
        curr_ = df[df['HUC2'] == h]
        i = int(n / nrows)
        j = np.mod(n, nrows)
        n = n + 1
        ax = axes[i][j]
        with styles.USGSMap():
            ax.plot(curr_['Year'], curr_['est_per_capita'], marker='o', markerfacecolor='none')
            title = "$HUC2 = {}$".format(h)
            ax.text(0.97, 0.75, title,
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='m', fontsize=12)
            # ax.set_ylabel('GPCD')
            # ax.set_xlabel('Year')
            ax.xaxis.set_ticks(np.arange(2000, 2021, 5))
            ax.set_xticklabels(ax.get_xticks(), rotation=45)

    fig.text(0.5, 0.01, 'Year', ha='center', fontsize=12)
    fig.text(0.08, 0.5, 'Per Capita Per Day (G) ', va='center', rotation='vertical', fontsize=12)

    plt.savefig(figfile)
    joblib.dump(fig, figfile + ".fig")
    plt.close(fig)

def plot_monthly_fraction_per_huc2(df, x, y, figfile):
    nrows = 3
    ncols = 6
    fig, axes = plt.subplots(ncols, nrows, figsize=(13.33, 7.5), sharex=True)
    huc2s = df['HUC2'].unique()
    n = 0

    for h in huc2s:
        curr_ = df[df['HUC2'] == h]
        i = int(n / nrows)
        j = np.mod(n, nrows)
        n = n + 1
        ax = axes[i][j]
        with styles.USGSMap():
            ax.plot(curr_['Year'], curr_['est_per_capita'], marker='o', markerfacecolor='none')
            title = "$HUC2 = {}$".format(h)
            ax.text(0.97, 0.75, title,
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='m', fontsize=12)
            # ax.set_ylabel('GPCD')
            # ax.set_xlabel('Year')
            ax.xaxis.set_ticks(np.arange(2000, 2021, 5))
            ax.set_xticklabels(ax.get_xticks(), rotation=45)

    fig.text(0.5, 0.01, 'Year', ha='center', fontsize=12)
    fig.text(0.08, 0.5, 'Per Capita Per Day (G) ', va='center', rotation='vertical', fontsize=12)

    plt.savefig(figfile)
    joblib.dump(fig, figfile + ".fig")
    plt.close(fig)

def plot_pc_hist_per_huc2(df, figfile):

    nrows = 3
    ncols = 6
    fig, axes = plt.subplots(ncols, nrows, figsize=(13.33, 7.5), sharex=True)
    huc2s = df['HUC2'].unique()
    huc2s = np.sort(huc2s)
    n = 0

    for h in huc2s:
        curr_ = df[df['HUC2'] == h]
        curr_ = curr_.groupby(by = ['sys_id']).mean()
        curr_.reset_index(inplace=True)
        i = int(n / nrows)
        j = np.mod(n, nrows)
        n = n + 1
        ax = axes[i][j]
        with styles.USGSMap():
            #ax.hist(curr_['est_per_capita'], bins = 15)
            sns.histplot(ax = ax, data=curr_, x="est_per_capita", stat = 'density', bins = 30, kde=True,
                  )
            ax.lines[0].set_color('red')
            #sns.kdeplot(ax = ax, data=curr_, x="est_per_capita", color='orange')
            title = "$HUC2 = {}$".format(h)
            ax.text(0.97, 0.75, title,
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='g', fontsize=12)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.xaxis.set_ticks(np.arange(25, 500, 50))
            ax.set_xticklabels(ax.get_xticks(), rotation=45)

    fig.text(0.5, 0.01, 'Gallon Per Capita Per Day', ha='center', fontsize=12)
    fig.text(0.08, 0.5, 'density', va='center', rotation='vertical', fontsize=12)

    plt.savefig(figfile)
    joblib.dump(fig, figfile + ".fig")
    plt.close(fig)


def plot_national_pc_change(df_, x, y, figfile=''):
    df = df_.copy()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    with styles.USGSMap():
        bars = ax.bar(
            x=df[x].values,
            height=df[y].values,
            color='tomato', edgecolor='blue', alpha = 0.2
        )

        ax.plot(df[x], df[y],'--', marker = 'o', color = 'm', markeredgecolor = 'g', markerfacecolor = 'g')

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.spines['bottom'].set_color('#DDDDDD')
        # ax.tick_params(bottom=False, left=False)
        # ax.set_axisbelow(True)
        # ax.yaxis.grid(True, color='#EEEEEE')
        # ax.xaxis.grid(False)

        # Add text annotations to the top of the bars.
        bar_color = bars[0].get_facecolor()
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                round(bar.get_height(), 1),
                horizontalalignment='center',
                color='k',
                weight='bold',
                rotation=-45,
                fontsize = 9
            )

        # extra space between the text and the tick labels.
        ax.set_xlabel('Year',  color='#333333', fontsize=14)#labelpad=15,
        ax.set_ylabel('Water Use - Gallons Per Capita Per Day (gpcd)',  color='#333333', fontsize=14)#labelpad=15,
        title = 'Annual Average Per Capita Water Use [2000-2020]'
        styles.heading(ax=ax,
                       heading=title,
                       idx=0, fontsize=16)
        ax.set_ylim([100, 140])

        plt.savefig(figfile)
        joblib.dump(fig, figfile + ".fig")
        plt.close(fig)



def time_bars(df_, x, y, figfile=''):
    df = df_.copy()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    with styles.USGSMap():
        df[y] = df[y] / 1e6
        bars = ax.bar(
            x=df[x].values,
            height=df[y].values
        )

        ax.plot(df[x], df[y], color = 'r', marker = 'o')
        # Axis formatting.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(False)
        #
        # Add text annotations to the top of the bars.
        bar_color = bars[0].get_facecolor()
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 300,
                int(bar.get_height()),
                horizontalalignment='center',
                color=bar_color,
                weight='bold',
                rotation=-45
            )

        # extra space between the text and the tick labels.
        ax.set_xlabel('Year', labelpad=15, color='#333333')
        ax.set_ylabel('Total Water Use - Million of Gallons Per Day (mgpd)', labelpad=15, color='#333333')

        title = 'Annual Public Water Use [2000-2020]'
        styles.heading(ax=ax,
                       heading=title,
                       idx=0, fontsize=16)

        ax.set_ylim([2.5e4, 3.9e4])
        plt.savefig(figfile)
        joblib.dump(fig, figfile + ".fig")
        plt.close(fig)


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
    return shp_df

def plot_county_map(shp_file, usa_map, info_df, legend_column, log_scale=False, epsg=5070, cmap='cool',
                  title='', figfile='', polygon_col_id = 'HUC2'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    usa_bkg = geopandas.read_file(usa_map)

    shp_df = geopandas.read_file(shp_file)
    shp_df.to_crs(epsg=epsg, inplace=True)
    usa_bkg.to_crs(epsg=epsg, inplace=True)
    info_df['county_id'] = info_df['county_id'].astype(int).astype(str).str.zfill(5)
    shp_df = shp_df[[polygon_col_id, 'geometry']]
    shp_df = shp_df.merge(info_df, how='left', left_on=polygon_col_id, right_on = 'county_id')

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

    return shp_df

def plot_state_map(shp_file, usa_map, info_df, legend_column, log_scale=False, epsg=5070, cmap='cool',
                  title='', figfile='', polygon_col_id = 'HUC2'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    usa_bkg = geopandas.read_file(usa_map)

    shp_df = geopandas.read_file(shp_file)
    shp_df.to_crs(epsg=epsg, inplace=True)
    usa_bkg.to_crs(epsg=epsg, inplace=True)
    info_df['state_id'] = info_df['state_id'].astype(int).astype(str).str.zfill(2)
    shp_df = shp_df[[polygon_col_id, 'geometry']]
    shp_df = shp_df.merge(info_df, how='left', left_on=polygon_col_id, right_on = 'state_id')

    if log_scale:
        norm = matplotlib.colors.LogNorm(vmin=shp_df[legend_column].min()-1, vmax=shp_df[legend_column].max()+1)
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

    return shp_df


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

        return df_shp

def plot_multiple_scatter_map(df, xcol, ycol, legend_column,
                                            fig_info, figfile):


    df_ = df.groupby(by=['sys_id', 'Month']).mean()
    df_.reset_index(inplace=True)

    lon = df_[xcol]
    lat = df_[ycol]
    df_shp = geopandas.GeoDataFrame(
        df_, geometry=geopandas.points_from_xy(lon, lat, crs="EPSG:4326"))  # 2163

    # fig = fig_info['fig']
    # axes = fig_info['axs']
    nrows = fig_info['nrows']
    ncols = fig_info['ncols']
    cmap = fig_info['cmap']
    log_scale = fig_info['log_scale']
    epsg = fig_info['epsg']
    title = fig_info['title']
    super_title = fig_info['super_title']

    df_shp.to_crs(epsg=epsg, inplace=True)

    fig, axes = plt.subplots(nrows=ncols, ncols=nrows, figsize=(10, 8))
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December']
    with styles.USGSMap():
        for month in range(1, 13):
            df_ = df_shp[df_shp['Month'] == month]
            max_v = df_[legend_column].quantile(0.9999)
            df_.loc[df_[legend_column]>max_v, legend_column] = max_v
            n = month - 1
            i = int(n / nrows)
            j = np.mod(n, nrows)

            ax = axes[i][j]

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            if log_scale:
                norm = matplotlib.colors.LogNorm(vmin=df_[legend_column].min(), vmax=df_[legend_column].max())
                ax = df_.plot(ax=ax, column=legend_column, alpha=1, cmap=cmap, markersize=2, legend=True,
                              legend_kwds={'shrink': 0.6}, cax=cax, norm=norm)
            else:
                norm = None
                ax = df_.plot(ax=ax, column=legend_column, alpha=1, cmap=cmap, markersize=1, legend=True,
                              legend_kwds={'shrink': 0.6}, cax=cax,vmin = 0.03, vmax = 0.18, norm=norm) #, vmin = 0.03, vmax = 0.18




            cx.add_basemap(ax, crs=df_.crs.to_string(), source=cx.providers.Stamen.TonerLines, alpha=1,
                           attribution=False)
            cx.add_basemap(ax, crs=df_.crs.to_string(), source=cx.providers.Stamen.TonerBackground, alpha=0.1,
                           attribution=False)

            styles.heading(ax=ax, letter= months[n],
                           heading=title,
                           idx=0, fontsize=10)
            styles.xlabel(ax=ax, fontsize=10, label='X (meter)')
            styles.ylabel(ax=ax, fontsize=10, label='Y (meter)')
            plt.tick_params(labelsize=12)
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)
            ax.get_yaxis().get_offset_text().set_position((-0.25, 0.0))

        plt.suptitle(super_title,fontsize=20)
        plt.tight_layout()
        plt.savefig(figfile)
        plt.close(fig)
        return df_shp

        nrows = 3
        ncols = 6
        fig, axes = plt.subplots(ncols, nrows, figsize=(13.33, 7.5), sharex=True)
        huc2s = df['HUC2'].unique()
        huc2s = np.sort(huc2s)
        n = 0
        for h in huc2s:
            curr_ = df[df['HUC2'] == h]
            curr_ = curr_.groupby(by=['HUC2', 'Year', 'Month']).mean()
            curr_.reset_index(inplace=True)
            curr_['date'] = curr_['Year'] + (curr_['Month'] - 0.5) / 12
            curr_.reset_index(inplace=True)
            i = int(n / nrows)
            j = np.mod(n, nrows)
            n = n + 1
            ax = axes[i][j]
            with styles.USGSMap():
                ax.plot(curr_['date'], curr_['est_month_frac'])
                title = "$HUC2 = {}$".format(h)
                ax.text(0.97, 0.75, title,
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=ax.transAxes,
                        color='m', fontsize=8)
                # ax.set_ylabel('GPCD')
                # ax.set_xlabel('Year')
                ax.set_ylim([0.03, 0.18])
                ax.xaxis.set_ticks(np.arange(2000, 2021, 5))
                ax.set_xticklabels(ax.get_xticks(), rotation=45)

        fig.text(0.5, 0.01, 'Year', ha='center', fontsize=12)
        fig.text(0.08, 0.5, 'Per Capita Per Day (G) ', va='center', rotation='vertical', fontsize=12)

        plot = sns.lineplot(data=df, x='Month', y='est_month_frac', hue="HUC2", style="HUC2")
        handles, labels = plot.axes[0].get_legend_handles_labels()
        plot._legend.remove()
        plot.fig.legend(handles, labels, ncol=2, loc='upper center',
                        bbox_to_anchor=(0.5, 1.15), frameon=False)




def residual_vs_fitted(model, estimator, figfile):
    ws = model.model_ws
    fig_folder = os.path.join(ws, "figs")
    files_info = model.files_info

    features = model.features
    target = model.target
    X_train = model.splits["X_train"]
    X_test = model.splits["X_test"]
    y_train = model.splits["y_train"]
    y_test = model.splits["y_test"]

    y_hat = estimator.predict(X_test[features])
    err = pd.DataFrame(y_test - y_hat)

    err = err.rename(columns={model.target: "err"})
    err['y_actual'] = y_test
    err['y_pred'] = y_hat

    fig, axes = plt.subplots(3,1, figsize=(8, 8))

    ax=sns.residplot(ax = axes[0], data=err, y='err', x='y_actual',
                          scatter_kws={'alpha': 0.5, 's':4},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    ax.set_title('Residuals vs Fitted')
    ax.set_xlabel('Actual values')
    ax.set_ylabel('Residuals');

    QQ = ProbPlot((err['err']-err['err'].mean())/err['err'].std())
    plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1, ax = axes[1])
    plot_lm_2.axes[0].set_title('Normal Q-Q')
    plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_2.axes[0].set_ylabel('Standardized Residuals');

    sns.histplot(ax= axes[2], data=err, x="err", kde=True, stat = 'probability')
    plt.tight_layout()
    plt.savefig(figfile)
    joblib.dump(fig, figfile + ".fig")
    plt.close(fig)

