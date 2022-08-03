import os

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, max_error, mean_squared_error, r2_score

try:
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
except:
    from sklearn.metrics import mean_absolute_error, median_absolute_error
from iwateruse import figures
from sklearn.pipeline import Pipeline


def generate_metrics(y_true, y_pred):
    scores = []
    names = []

    scores.append(r2_score(y_true=y_true, y_pred=y_pred))
    names.append("r2")

    scores.append(mean_squared_error(y_true=y_true, y_pred=y_pred))
    names.append("mse")

    scores.append(np.power(mean_squared_error(y_true=y_true, y_pred=y_pred), 0.5))
    names.append("rmse")

    scores.append(mean_absolute_error(y_true=y_true, y_pred=y_pred))
    names.append("mae")

    scores.append(median_absolute_error(y_true=y_true, y_pred=y_pred))
    names.append("mdae")

    try:
        scores.append(mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred))
        names.append("mape")
    except:
        print("Mape is not available in this sklearn")

    scores.append(max_error(y_true=y_true, y_pred=y_pred))
    names.append("maxe")

    scores.append(explained_variance_score(y_true=y_true, y_pred=y_pred))
    names.append("evar")

    df = pd.DataFrame([scores], columns=names)
    return df


def generat_metric_by_category(estimator, X, y_true, features, category='HUC2'):
    ids = X[category].unique()
    all_metrics = []
    for id in ids:
        mask = X[category] == id
        X_ = X[mask].copy()
        y_hat = estimator.predict(X_[features])
        df_ = generate_metrics(y_true[mask], y_hat)
        df_[category] = id
        df_['n_samples'] = len(X_)
        all_metrics.append(df_)
    all_metrics = pd.concat(all_metrics)
    all_metrics.reset_index(inplace=True)
    all_metrics.drop(columns=['index'], inplace=True)

    return all_metrics


def complete_model_diagnose(model, estimator=None, basename="initial"):
    """

    :return:
    """
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
    df_metric = generate_metrics(y_true=y_test, y_pred=y_hat)

    # -----------------------
    # one2one plot
    # -----------------------

    heading = "Scatter Plot for Annual Water Use"
    xlabel = "Actual Per Capita Water Use - Gallons"
    ylabel = "Estimated Per Capita Water Use - Gallons"
    figfile = os.path.join(fig_folder, "1_annual_1to1_{}.pdf".format(basename))

    # -----------------------
    # Model performance
    # -----------------------
    figures.one_to_one(y_test, y_hat, heading=heading, xlabel=xlabel, ylabel=ylabel, figfile=figfile)
    model.log.info("\n\n\n ======= {} model performance ==========".format(basename))
    model.log.info("See initial model perfromance at {}".format(figfile))
    model.log.to_table(df_metric)
    df_metric.to_csv(os.path.join(ws, "2_performance_metric_{}.csv".format(basename)), index=False)

    # -----------------------
    # regional performance
    # -----------------------
    perf_df_huc2 = generat_metric_by_category(estimator, X_test, y_test, features=features, category='HUC2')
    perf_df_huc2.to_csv(os.path.join(ws, "3_performance_by_huc2_{}.csv".format(basename)), index=False)
    perf_df_state = generat_metric_by_category(estimator, X_test, y_test, features=features, category='state_id')
    perf_df_state.to_csv(os.path.join(ws, "4_performance_by_state_{}.csv".format(basename)), index=False)

    # -----------------------
    # regional performance maps
    # -----------------------
    huc2_shp_file = files_info['huc2_shp_file']
    usa_conus_file = files_info['usa_conus_file']
    for err_type in ['rmse', 'r2', 'mape']:
        figfile = os.path.join(fig_folder, "5_validation_{}_by_huc2_{}.pdf".format(err_type, basename))
        df_shp = figures.plot_huc2_map(shp_file=huc2_shp_file, usa_map=usa_conus_file, info_df=perf_df_huc2,
                                       legend_column=err_type,
                                       log_scale=False, epsg=5070, cmap='cool', title=err_type, figfile=figfile)

    figfile = os.path.join(fig_folder, "6_validation_by_huc2_{}.pdf".format(basename))
    df_shp.to_file(figfile + ".shp")

    figfile = os.path.join(fig_folder, "7_map_val_error_{}.pdf".format(basename))
    df_shp = figures.plot_scatter_map(X_test['LONG'], X_test['LAT'], err,
                                      legend_column='per_capita', cmap='jet', title="Per Capita WU", figfile=figfile,
                                      log_scale=False)
    figfile = os.path.join(fig_folder, "6_map_val_error_{}.shp".format(basename))
    df_shp.to_file(figfile)

    # -----------------------
    # plot importance
    # -----------------------
    importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    for type in importance_types:
        figfile = os.path.join(fig_folder, "8_feature_importance_{}_{}.pdf".format(type, basename))
        if isinstance(estimator, Pipeline):
            estm = estimator.named_steps["estimator"]
            estimator.named_steps["preprocess"].get_feature_names()
        else:
            estm = estimator
        figures.feature_importance(estm, max_num_feature=15, type=type, figfile=figfile)

    # -----------------------
    # error plots
    # -----------------------
    figfile = os.path.join(fig_folder, "9_resid_err_{}.pdf".format(basename))
    figures.residual_vs_fitted(model, estimator, figfile)


def compute_temporal_change(model):
    # states
    all_states = model.df_pred[['state_id', 'Year', 'est_per_capita', 'pop']]
    all_states['total_wu'] = all_states['pop'] * all_states['est_per_capita']
    all_states = all_states.groupby(by=['state_id', 'Year']).sum()
    all_states.reset_index(inplace=True)
    all_states['est_per_capita'] = all_states['total_wu'] / all_states['pop']
    states = all_states['state_id'].unique()
    temp = []
    for state in states:
        curr_state = all_states[all_states['state_id'] == state]
        curr_state.sort_values(by='Year', inplace=True)
        curr_state['del_pcgd'] = (curr_state['est_per_capita'].diff() / curr_state['est_per_capita']) * curr_state[
            'est_per_capita']
        temp.append(curr_state.copy())
    all_states = pd.concat(temp)

    # huc2
    # states
    all_huc2s = model.df_pred[['HUC2', 'Year', 'est_per_capita', 'pop']]
    all_huc2s['total_wu'] = all_huc2s['pop'] * all_huc2s['est_per_capita']
    all_huc2s = all_huc2s.groupby(by=['HUC2', 'Year']).sum()
    all_huc2s.reset_index(inplace=True)
    all_huc2s['est_per_capita'] = all_huc2s['total_wu'] / all_huc2s['pop']
    huc2s = all_huc2s['HUC2'].unique()
    temp = []
    for h in huc2s:
        curr_state = all_huc2s[all_huc2s['HUC2'] == h]
        curr_state.sort_values(by='Year', inplace=True)
        curr_state['del_pcgd'] = (curr_state['est_per_capita'].diff() / curr_state['est_per_capita']) * curr_state[
            'est_per_capita']
        temp.append(curr_state.copy())
    all_huc2s = pd.concat(temp)

    # counties
    all_counties = model.df_pred[['county_id', 'Year', 'est_per_capita', 'pop']]
    all_counties['total_wu'] = all_counties['pop'] * all_counties['est_per_capita']
    all_counties = all_counties.groupby(by=['county_id', 'Year']).sum()
    all_counties.reset_index(inplace=True)
    all_counties['est_per_capita'] = all_counties['total_wu'] / all_counties['pop']
    counties = all_counties['county_id'].unique()
    temp = []
    for h in counties:
        curr_state = all_counties[all_counties['county_id'] == h]
        curr_state.sort_values(by='Year', inplace=True)
        curr_state['del_pcgd'] = (curr_state['est_per_capita'].diff() / curr_state['est_per_capita']) * curr_state[
            'est_per_capita']
        temp.append(curr_state.copy())
    all_counties = pd.concat(temp)

    return all_counties, all_states, all_huc2s


def complete_model_evaluate(model, estimator=None, basename="initial"):
    ws = model.model_ws
    fig_folder = os.path.join(ws, "predictions")
    if not (os.path.isdir(fig_folder)):
        os.mkdir(fig_folder)
    files_info = model.files_info
    df_pred = model.df_pred
    df_pred['tot_wu'] = df_pred['est_per_capita'] * df_pred['pop']

    #AWUDS
    awuds_file = model.files_info['AWUDS_file']
    awuds_df = pd.read_csv(awuds_file)
    awuds_df = awuds_df[awuds_df['YEAR'] > 2000]
    awuds_df['STATECODE'] = awuds_df['STATECODE'].astype(str).str.zfill(2)
    awuds_df['COUNTYCODE'] = awuds_df['COUNTYCODE'].astype(str).str.zfill(3)
    awuds_df['county_id'] = awuds_df['STATECODE'].astype(str) + awuds_df['COUNTYCODE']
    awuds_df_ = awuds_df[['county_id', 'PS-WTotl', 'PS-DelDO', 'PS-TOPop']]
    awuds_df_['PS-WTotl'] = awuds_df_['PS-WTotl']*1e6
    awuds_df_['PS-DelDO'] = awuds_df_['PS-DelDO'] * 1e6
    awuds_df_['PS-TOPop'] = awuds_df_['PS-TOPop'] * 1e3
    awuds_df_['awuds_pc'] =  awuds_df_['PS-WTotl']/ awuds_df_['PS-TOPop']
    awuds_county_pc =awuds_df_[['county_id', 'awuds_pc' ]].groupby(by = 'county_id').mean()
    model_county_pc = df_pred[['county_id', 'pop', 'tot_wu']].groupby(by=['county_id']).sum()
    model_county_pc.reset_index(inplace=True)
    model_county_pc['county_id'] = model_county_pc['county_id'].astype(str)
    model_county_pc['county_id'] = model_county_pc['county_id'].str.zfill(5)
    model_county_pc['pc'] = model_county_pc['tot_wu'] / model_county_pc['pop']
    model_county_pc = model_county_pc[['county_id', 'pc']].groupby(['county_id']).mean()
    model_county_pc['awuds_pc'] = awuds_county_pc['awuds_pc']

    #---------------
    # nation level
    #---------------

    us_annual = df_pred[['Year', 'pop', 'tot_wu']].groupby(by='Year').sum()
    us_annual.reset_index(inplace=True)
    figfile = os.path.join(fig_folder, "1_total_wu_change_with_time_{}.pdf".format(basename))
    figures.time_bars(us_annual, x='Year', y='tot_wu', figfile=figfile)

    us_annual['pc'] = us_annual['tot_wu'] / us_annual['pop']
    figfile = os.path.join(fig_folder, "2_national_per_capita_change_with_time_{}.pdf".format(basename))
    figures.plot_national_pc_change(us_annual, x='Year', y='pc', figfile=figfile)

    # change with time analysis
    all_counties, all_states, all_huc2s = compute_temporal_change(model)

    # -----------------------
    # Plot per capita change with time for every HUC2
    # -----------------------
    figfile = os.path.join(fig_folder, "3_per_capita_change_with_time_for_each_huc2_{}.pdf".format(basename))
    figures.plot_timeseries_per_huc2(df=all_huc2s, x='Year', y='est_per_capita', figfile=figfile)

    # -----------------------
    # Plot average per capita hist for every HUC2
    # -----------------------
    figfile = os.path.join(fig_folder, "4_per_capita_hist_each_huc2_{}.pdf".format(basename))
    figures.plot_pc_hist_per_huc2(df=model.df_pred, figfile = figfile)


    # -----------------------
    # service_area average per capita scatter map
    # -----------------------
    av_per_capita = df_pred.groupby(by='sys_id').mean()[['LONG', 'LAT', 'est_per_capita']]
    figfile = os.path.join(fig_folder, "5_mean_service_area_per_capita_{}.pdf".format(basename))
    df_shp = figures.plot_scatter_map(av_per_capita['LONG'], av_per_capita['LAT'], av_per_capita['est_per_capita'],
                                      legend_column='est_per_capita', cmap='jet', title="Per Capita WU",
                                      figfile=figfile,
                                      log_scale=False)
    figfile = os.path.join(fig_folder, "5_mean_service_area_per_capita_{}.shp".format(basename))
    df_shp.to_file(figfile)

    # -----------------------
    # HUC2  average capita map ploygon map
    # -----------------------
    huc2_shp_file = files_info['huc2_shp_file']
    usa_conus_file = files_info['usa_conus_file']
    summary_df = model.df_pred[['HUC2', 'est_per_capita']].groupby(by='HUC2').describe()
    cols = summary_df.columns
    renamed_cols = []
    for c in cols:
        renamed_cols.append(c[1])
    summary_df.columns = renamed_cols

    df_temp = model.df_pred[['HUC2', 'pop', 'tot_wu']].groupby(by=['HUC2']).sum()
    df_temp['wpc'] = df_temp['tot_wu'] / df_temp['pop']
    summary_df['wmean'] = df_temp['wpc']

    summary_df.reset_index(inplace=True)
    figfile = os.path.join(fig_folder, "6_mean_huc2_per_capita_{}.pdf".format(basename))
    df_shp = figures.plot_huc2_map(shp_file=huc2_shp_file, usa_map=usa_conus_file, info_df=summary_df,
                                   legend_column='wmean',
                                   log_scale=False, epsg=5070, cmap='Greens',
                                   title="Average Per Capita by HUC2", figfile=figfile)
    figfile = os.path.join(fig_folder, "6_huc2_stats_per_capita_{}.pdf".format(basename))
    df_shp.to_file(figfile + ".shp")

    # -----------------------
    # state per average capita map ploygon map
    # -----------------------
    state_shp_file = files_info['state_shp_file']
    usa_conus_file = files_info['usa_conus_file']

    summary_df = model.df_pred[['state_id', 'est_per_capita']].groupby(by='state_id').describe()
    cols = summary_df.columns
    renamed_cols = []
    for c in cols:
        renamed_cols.append(c[1])
    summary_df.columns = renamed_cols

    df_temp = all_states[['state_id', 'pop', 'total_wu']].groupby(by=['state_id']).mean()
    df_temp['wpc'] = df_temp['total_wu'] / df_temp['pop']
    summary_df['wmean'] = df_temp['wpc']
    summary_df['total_wu'] = df_temp['total_wu']
    summary_df.reset_index(inplace=True)

    figfile = os.path.join(fig_folder, "7_mean_state_per_capita_{}.pdf".format(basename))
    df_shp = figures.plot_state_map(shp_file=state_shp_file, usa_map=usa_conus_file, info_df=summary_df,
                                    legend_column='wmean',
                                    log_scale=False, epsg=5070, cmap='Purples', title="Average Per Capita by State",
                                    figfile=figfile, polygon_col_id='GEOID')
    figfile = os.path.join(fig_folder, "7__state_stats_per_capita_{}.shp".format(basename))
    df_shp.to_file(figfile)

    figfile = os.path.join(fig_folder, "8_mean_state_total_wu_{}.pdf".format(basename))
    df_shp = figures.plot_state_map(shp_file=state_shp_file, usa_map=usa_conus_file, info_df=summary_df,
                                    legend_column='total_wu',
                                    log_scale=True, epsg=5070, cmap='winter', title="Average Total Water Use by State",
                                    figfile=figfile, polygon_col_id='GEOID')

    # -----------------------
    # county per average capita map ploygon map
    # -----------------------
    county_shp_file = files_info['county_shp_file']
    usa_conus_file = files_info['usa_conus_file']
    figfile = os.path.join(fig_folder, "9_mean_county_per_capita_{}.pdf".format(basename))
    summary_df = model.df_pred[['county_id', 'est_per_capita']].groupby(by='county_id').describe()
    cols = summary_df.columns
    renamed_cols = []
    for c in cols:
        renamed_cols.append(c[1])
    summary_df.columns = renamed_cols
    df_temp = model.df_pred[['county_id', 'pop', 'tot_wu']].groupby(by=['county_id']).sum()
    df_temp['wpc'] = df_temp['tot_wu'] / df_temp['pop']
    summary_df['wmean'] = df_temp['wpc']
    summary_df.reset_index(inplace=True)
    df_shp = figures.plot_county_map(shp_file=county_shp_file, usa_map=usa_conus_file, info_df=summary_df,
                                     legend_column='wmean',
                                     log_scale=False, epsg=5070, cmap='Oranges', title="Average Per Capita by State",
                                     figfile=figfile, polygon_col_id='GEOID')
    figfile = os.path.join(fig_folder, "9__county_stats_per_capita_{}.shp".format(basename))
    df_shp.to_file(figfile)

    # -----------------------
    # state change map ploygon map
    # -----------------------
    state_shp_file = files_info['state_shp_file']
    usa_conus_file = files_info['usa_conus_file']
    figfile = os.path.join(fig_folder, "10_pc_average_change_by_state_{}.pdf".format(basename))

    summary_df = all_states.groupby('state_id').sum()
    summary_df.reset_index(inplace=True)

    df_shp = figures.plot_state_map(shp_file=state_shp_file, usa_map=usa_conus_file, info_df=summary_df,
                                    legend_column='del_pcgd',
                                    log_scale=False, epsg=5070, cmap='spring',
                                    title="Average Change of  Per Capita by State in the Period 2000-2020",
                                    figfile=figfile, polygon_col_id='GEOID')
    figfile = os.path.join(fig_folder, "10_pc_average_change_by_state_{}.shp".format(basename))
    df_shp.to_file(figfile)
