# ===============****===================
"""
Monthly Water Use Model
Date: 7/14/2022
"""
# ===============****===================
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json
import xgboost as xgb
from sklearn.pipeline import Pipeline
from iwateruse.model import Model
from iwateruse import targets, weights, pipelines, outliers_utils, estimators, featurize
from iwateruse import selection
from iwateruse import data_cleaning, report, splittors, pre_train_utils, make_dataset, figures
from iwateruse import denoise, model_diagnose

warnings.filterwarnings('ignore')
xgb.set_config(verbosity=0)

# =============================
# Flags
# ==============================
train_initial_model = False
run_boruta = False
use_boruta_results = True
run_permutation_selection = False
run_chi_selection = False
run_RFECV_selection = False
make_prediction = False
interpret_model = False
quantile_regression = False

# files
usa_conus_file = r"C:\work\water_use\mldataset\gis\nation_gis_files\conus_usa2.shp"
huc2_shp_file = r'C:\work\water_use\mldataset\gis\nation_gis_files\conus_huc2.shp'
state_shp_file = r'C:\work\water_use\mldataset\gis\nation_gis_files\conus_states.shp'
county_shp_file = r'C:\work\water_use\mldataset\gis\nation_gis_files\conus_county.shp'

# =============================
# Setup Training
# =============================
figures_folder = "figs"
model = Model(name='Monthly_frac', log_file='train_log_monthly.log', feature_status_file=r"features_status.xlsx",
              model_type='monthly')
model.raw_target = 'monthly_wu'
model.target = 'monthly_fraction'
datafile = r"clean_train_db_monthly.csv"  # clean_train_db_backup_6_24_2022.csv
df_train = pd.read_csv(datafile)
model.add_training_df(df_train=df_train)

# add water use
seed1 = 123
seed2 = 456

opts = ['monthly_fraction>0.18', 'monthly_fraction<0.03', 'simil_stat<0.7']
model.apply_func(func=outliers_utils.drop_values, type='outliers_func', opts=opts)
model.apply_func(func=outliers_utils.drop_na_target, type='outliers_func')

# =============================
# Feature Engineering
# =============================
# Target summary
tr_df, cat_df = featurize.summary_encode(model, cols=model.categorical_features)
df_concat = pd.concat([tr_df, cat_df], axis=1)
model.add_feature_to_skip_list(model.categorical_features)
model.add_training_df(df_train=df_concat)
del (tr_df);
del (cat_df);
del (df_concat)

# =============================
# Prepare the initial estimator
# =============================

params = {
    'objective': "reg:squarederror",
    'tree_method': 'hist',
    'colsample_bytree': 0.9,  #
    'learning_rate': 0.06,  #
    'max_depth': 10,  #
    'alpha': 0.0,  #
    'n_estimators': 600,  #
    'rate_drop': 0.1,  #
    'skip_drop': 0.5,  #
    'subsample': 0.85,  #
    'reg_lambda': 1,
    'min_child_weight': 1,
    'gamma': 0,  #
    'max_delta_step': 0,
    'seed': seed2,
    'importance_type': 'total_gain'  # this is used in boruta
}

gb = estimators.xgb_estimator(params)

encode_cat_features = False
if encode_cat_features:
    # pipeline
    main_pipeline = pipelines.make_pipeline(model)
    main_pipeline.append(('estimator', gb))
    gb = Pipeline(main_pipeline)

features = model.features
target = model.target

final_dataset = model.df_train
model.log.to_table(final_dataset[features].describe(percentiles=[0, 0.05, 0.5, 0.95, 1]), "Features", header=-1)
model.log.to_table(final_dataset[[target]].describe(percentiles=[0, 0.05, 0.5, 0.95, 1]), "Target", header=-1)

X_train, X_test, y_train, y_test = splittors.stratified_split(model, test_size=0.3, id_column='HUC2', seed=123)
# X_train, X_test, y_train, y_test = splittors.split_by_id(model, args = {'frac' : 0.7, 'seed':547, 'id_column':'sys_id' })
# X_train, X_test, y_train, y_test = train_test_split(final_dataset[features],  final_dataset[target],
#                                                               test_size=0.2, random_state=123)#, shuffle = False
if train_initial_model:
    vv = gb.fit(X_train[features], y_train)
    y_hat = gb.predict(X_test[features])
    err = pd.DataFrame(y_test - y_hat)

    model_file_name = r".\models\monthly\1_initial_model.json"
    model.log.info("\n\n Initial Model saved at ")
    vv.save_model(model_file_name)

    # =============================
    # initial diagnose
    # =============================
    heading = "Scatter Plot for Month Water Use Fractions"
    xlabel = "Actual Water Use Monthly Fraction - Unitless"
    ylabel = "Estimated Water Use Monthly Fraction - Unitless"
    figfile = os.path.join(figures_folder, "monthly_one_2_one_initial.pdf")
    figures.one_to_one(y_test, y_hat, heading=heading, xlabel=xlabel, ylabel=ylabel, figfile=figfile)
    model.log.info("\n\n\n ======= Initial model performance ==========")
    df_metric = model_diagnose.generate_metrics(y_true=y_test, y_pred=y_hat)
    model.log.info("See initial model perfromance at {}".format(figfile))
    model.log.to_table(df_metric)

    # regional performance
    perf_df = model_diagnose.generat_metric_by_category(gb, X_test, y_test, features=features, category='HUC2')

    figfile = os.path.join(figures_folder, "monthly_validation_error_map_initial.pdf")
    figures.plot_scatter_map(X_test['LONG'], X_test['LAT'], err,
                             legend_column='monthly_fraction', cmap='jet', title="Monthly Fractions", figfile=figfile,
                             log_scale=False)

    # plot importance
    importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    for type in importance_types:
        figfile = os.path.join(figures_folder, "annual_feature_importance_{}_v_initial.pdf".format(type))
        if isinstance(gb, Pipeline):
            estm = gb.named_steps["estimator"]
            gb.named_steps["preprocess"].get_feature_names()
        else:
            estm = gb

        figures.feature_importance(estm, max_num_feature=15, type=type, figfile=figfile)

# =============================
# Simplify model by dropping
# features with little importance
# =============================
drop_features = {}
feat_selec_file = 'confirmed_selected_features_monthly.json'
if run_boruta:
    rf = xgb.XGBRFRegressor(n_estimators=500, subsample=0.8, colsample_bynode=0.5, max_depth=7, )

    # impute by mean for boruta
    final_dataset_ = final_dataset.copy()
    for feat in features:
        nanMask = final_dataset_[feat].isna()
        feat_mean = final_dataset_[feat].mean()
        final_dataset_.loc[nanMask, feat] = feat_mean

    importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    for imtype in importance_types:
        rf.set_params(importance_type=imtype)
        confirmed_features = selection.boruta(X=final_dataset_[features], y=final_dataset_[target], estimator=rf)
        keyname = "rf_{}".format(imtype)
        drop_features[keyname] = confirmed_features.values.tolist()

        gb.set_params(importance_type=imtype)
        confirmed_features = selection.boruta(X=final_dataset_[features], y=final_dataset_[target], estimator=gb)
        keyname = "xgb_{}".format(imtype)
        drop_features[keyname] = confirmed_features.values.tolist()

    seeds = [123, 5795, 2136, 5214]
    perm_confirm_features = []
    for sd in seeds:
        X_train, X_test, y_train, y_test = splittors.stratified_split(model, test_size=0.3, id_column='HUC2', seed=sd)
        estm_ = gb.fit(X_train[features], y_train)
        scoring = ['r2']  # , 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
        feature_importance = selection.permutation_selection(X_test[features], y_test, estimator=estm_, scoring=scoring,
                                                             n_repeats=5, features=features)

        best_r2 = feature_importance[feature_importance['metric'].isin(['r2'])]
        confirmed_features = best_r2[(best_r2['mean_reduction'] / best_r2['mean_reduction'].max()) > 0.015]['feature']
        perm_confirm_features = perm_confirm_features + confirmed_features.values.tolist()

    keyname = "permutation_r2"
    drop_features[keyname] = list(set(perm_confirm_features))
    with open(feat_selec_file, 'w') as convert_file:
        convert_file.write(json.dumps(drop_features))

if use_boruta_results:
    # produced by runing "run_boruta"
    confirmed_features = ['Year', 'pop_density', 'gini', 'n_industry', 'median_income',
                          'median_h_year', 'HUC2_50', 'HUC2_75', 'LAT', 'LONG', 'county_id_5',
                          'county_id_25', 'county_id_50', 'county_id_75', 'county_id_95',
                          'awuds_pop_cnt', 'zill_nhouse', 'LotSizeSquareFeet_sum',
                          'YearBuilt_mean', 'BuildingAreaSqFt_sum', 'TaxAmount_mean',
                          'NoOfStories_mean', 'bdg_ftp_count', 'bdg_gt_2median', 'bdg_lt_2median',
                          'bdg_lt_4median', 'av_house_age', 'n_houses', 'average_income',
                          'state_id_5', 'state_id_25', 'state_id_50', 'state_id_95', 'Commercial',
                          'Domestic', 'Urban_Misc', 'Production', 'Water', 'WSA_SQKM',
                          'KG_climate_zone_5', 'KG_climate_zone_75', 'KG_climate_zone_95',
                          'prc_n_lt_ninth_gr', 'prc_n_ninth_to_twelth_gr', 'prc_n_hs_grad',
                          'prc_n_bachelors', 'prc_n_masters_phd', 'pop', 'cii_frac', 'etr', 'pr',
                          'pr_cumdev', 'tmmn', 'tmmx']

    features_to_drop = set(features).difference(confirmed_features)
    model.add_feature_to_skip_list(list(features_to_drop))
    features = model.features

    vv = gb.fit(X_train[features], y_train)
    y_hat = gb.predict(X_test[features])
    err = pd.DataFrame(y_test - y_hat)

    model_file_name = r".\models\monthly\2_model_after_dropping_unimportant_features.json"
    model.log.info("\n\n Initial Model saved at ")
    vv.save_model(model_file_name)

    # =============================
    # model diagnose after droping unimportant features
    # =============================
    heading = "Scatter Plot for Month Water Use Fractions"
    xlabel = "Actual Water Use Monthly Fraction - Unitless"
    ylabel = "Estimated Water Use Monthly Fraction - Unitless"
    figfile = os.path.join(figures_folder, "monthly_one_2_one_after_boruta.pdf")
    figures.one_to_one(y_test, y_hat, heading=heading, xlabel=xlabel, ylabel=ylabel, figfile=figfile)
    model.log.info("\n\n\n ======= model performance ==========")
    df_metric = model_diagnose.generate_metrics(y_true=y_test, y_pred=y_hat)
    model.log.info("See initial model perfromance at {}".format(figfile))
    model.log.to_table(df_metric)

    # regional performance
    perf_df = model_diagnose.generat_metric_by_category(gb, X_test, y_test, features=features, category='HUC2')

    figfile = os.path.join(figures_folder, "monthly_validation_error_map_boruta.pdf")
    figures.plot_scatter_map(X_test['LONG'], X_test['LAT'], err,
                             legend_column='monthly_fraction', cmap='jet', title="Monthly Fractions", figfile=figfile,
                             log_scale=False)

    # plot importance
    importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    for type in importance_types:
        figfile = os.path.join(figures_folder, "annual_feature_importance_{}_v_boruta.pdf".format(type))
        if isinstance(gb, Pipeline):
            estm = gb.named_steps["estimator"]
            gb.named_steps["preprocess"].get_feature_names()
        else:
            estm = gb

        figures.feature_importance(estm, max_num_feature=15, type=type, figfile=figfile)

if run_permutation_selection:
    scoring = ['r2']  # , 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
    feature_importance = selection.permutation_selection(X_test[features], y_test, estimator=gb, scoring=scoring,
                                                         n_repeats=10, features=features)
    metrics = feature_importance['metric'].unique()
    for m in metrics:
        curr_ = feature_importance[feature_importance['metric'].isin([m])]

if run_chi_selection:
    from sklearn.feature_selection import chi2

    chi_test_df = selection.chi_square_test(X=final_dataset[features], y=final_dataset[target], nbins=20)

if run_RFECV_selection:
    from sklearn.feature_selection import RFE, RFECV

    selector = RFECV(vv, verbose=10, scoring='r2')
    selector.fit(final_dataset[features], final_dataset[target])
    features_selected = selector.get_feature_names_out()
    feature_rank = selector.ranking_

# =============================
# Interpret the model
# =============================
if interpret_model:
    import shap

    shap.partial_dependence_plot('BuildingAreaSqFt_sum', gb.predict, X_test, ice=
    False, model_expected_value=True, feature_expected_value=True)

    # X100 = shap.utils.sample(X_train, 100)
    # X500 = shap.utils.sample(X_train, 5000)
    # explainer = shap.Explainer(model.predict, X100)
    X100 = X_train[X_train['HUC2'] == 18]
    X100 = shap.utils.sample(X100, 2000)
    explainer = shap.Explainer(vv, X100)
    # explainer = shap.TreeExplainer(gb, X100)
    shap_values = explainer(X100)
    # shap.plots.waterfall(shap_values[150], max_display=14)
    # shap.summary_plot(shap_values, X100)
    shap.plots.beeswarm(shap_values, max_display=14)
    shap.plots.scatter(shap_values[:, "etr"], color=shap_values)
    # PDPs
    from sklearn.inspection import PartialDependenceDisplay

    common_params = {
        "subsample": 50,
        "n_jobs": -1,
        "grid_resolution": 20,
        "random_state": 0,
    }
    interp_feat = ['bdg_lt_2median']
    PartialDependenceDisplay.from_estimator(gb, X_test, interp_feat)

# =============================
# Quantile regression
# =============================
if quantile_regression:
    from lightgbm import LGBMRegressor

    lgb_params = {'boosting_type': 'gbdt',
                  'class_weight': None,
                  'colsample_bytree': 1.0,
                  'importance_type': 'split',
                  'learning_rate': 0.0789707832086975,
                  'max_depth': -1,
                  'min_child_samples': 1,
                  'min_child_weight': 0.001,
                  'min_split_gain': 0.09505847801910139,
                  'n_estimators': 300,
                  'n_jobs': -1,
                  'num_leaves': 256,
                  'objective': 'quantile',
                  'random_state': 1880,
                  'reg_alpha': 0.7590311640897429,
                  'reg_lambda': 1.437857111206781e-10,
                  'silent': 'warn',
                  'subsample': 1.0,
                  'subsample_for_bin': 200000,
                  'subsample_freq': 0,
                  'bagging_fraction': 1.0,
                  'bagging_freq': 7,
                  'feature_fraction': 0.6210738953447875}
    quantile_alphas = [0.25, 0.5, 0.75]

    lgb_quantile_alphas = {}
    for quantile_alpha in quantile_alphas:
        lgb = LGBMRegressor(alpha=quantile_alpha, **lgb_params)
        lgb.fit(X_train, y_train)
        lgb_quantile_alphas[quantile_alpha] = lgb

    plt.figure()
    for quantile_alpha, lgb in lgb_quantile_alphas.items():
        ypredict = lgb.predict(X_test)
        plt.scatter(y_test, ypredict, s=4, label="{}".format
        (quantile_alpha))
    plt.legend()
    lim = [min(y_test), max(y_test)]
    plt.plot(lim, lim, 'k')
    plt.xlabel("Actual Water Use")
    plt.ylabel("Estimated Water Use")
