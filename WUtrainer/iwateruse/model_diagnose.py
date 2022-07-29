import os

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, max_error, mean_squared_error, r2_score
try :
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
except:
    from sklearn.metrics import mean_absolute_error,  median_absolute_error
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

def generat_metric_by_category(estimator, X, y_true, features, category = 'HUC2'):

    ids = X[category].unique()
    all_metrics = []
    for id in ids:
        mask = X[category] == id
        X_ = X[mask].copy()
        y_hat = estimator.predict(X_[features])
        df_ = generate_metrics(y_true[mask],y_hat)
        df_[category] = id
        df_['n_samples'] = len(X_)
        all_metrics.append(df_)
    all_metrics = pd.concat(all_metrics)
    all_metrics.reset_index(inplace = True)
    all_metrics.drop(columns=['index'], inplace=True)

    return all_metrics

def complete_model_diagnose(model, estimator = None,  basename = "initial"):
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
    perf_df_state = generat_metric_by_category(estimator, X_test, y_test,features=features, category='state_id')
    perf_df_state.to_csv(os.path.join(ws, "4_performance_by_state_{}.csv".format(basename)), index=False)

    # -----------------------
    # regional performance maps
    # -----------------------
    huc2_shp_file = files_info['huc2_shp_file']
    usa_conus_file = files_info['usa_conus_file']
    for err_type in ['rmse', 'r2', 'mape']:
        figfile = os.path.join(fig_folder, "5_validation_{}_by_huc2_{}.pdf".format(err_type, basename))
        figures.plot_huc2_map(shp_file=huc2_shp_file, usa_map=usa_conus_file, info_df=perf_df_huc2, legend_column=err_type,
                              log_scale=False, epsg=5070, cmap='cool', title=err_type, figfile=figfile)


    figfile = os.path.join(fig_folder, "6_map_val_error_{}.pdf".format(basename))
    figures.plot_scatter_map(X_test['LONG'], X_test['LAT'], err,
                             legend_column='per_capita', cmap='jet', title="Per Capita WU", figfile=figfile,
                             log_scale=False)
    # -----------------------
    # plot importance
    # -----------------------
    importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    for type in importance_types:
        figfile = os.path.join(fig_folder, "feature_importance_{}_{}.pdf".format(type, basename))
        if isinstance(estimator, Pipeline):
            estm = estimator.named_steps["estimator"]
            estimator.named_steps["preprocess"].get_feature_names()
        else:
            estm = estimator
        figures.feature_importance(estm, max_num_feature=15, type=type, figfile=figfile)

    # -----------------------
    # error_with predictions
    # -----------------------

    # histograms

    # error maps (1) scatter, (2) per HUC2, state, and county

    # feature importance plots

    # temporal total change and per capita change for each state and HUC2

    # shap plots





    x = 1








