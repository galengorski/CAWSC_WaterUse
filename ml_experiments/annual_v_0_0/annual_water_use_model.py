# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Annual Water Use 

# %%

import os, sys
import pandas as pd
from datetime import datetime

from iwateruse.featurize import MultiOneHotEncoder
from iwateruse import data_cleaning, report, splittors, pre_train_utils, make_dataset, figures
from iwateruse import denoise, model_diagnose

import matplotlib.pyplot as plt
from xgboost import plot_importance
import xgboost as xgb

#
import numpy as np
from flopy.plot import styles

# sklearn
from sklearn.metrics import r2_score, mean_squared_error

# %%
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

# %%
# %matplotlib widget
# %matplotlib inline
# %matplotlib ipympl
import warnings

warnings.filterwarnings('ignore')
xgb.set_config(verbosity=0)

# %%
from iwateruse.model import Model
from iwateruse import targets, weights, pipelines, outliers_utils, estimators, featurize
from iwateruse import selection


# %%
# =============================
# Setup Training
# =============================
figures_folder = "figs"
model = Model(name='annual_pc', log_file = 'train_log.log',  feature_status_file= r"features_status.xlsx")
model.raw_target = 'wu_rate'
model.target = 'per_capita'

datafile = r"clean_train_db.csv" #clean_train_db_backup_6_24_2022.csv
df_train = pd.read_csv(datafile)
del(df_train['dom_frac'])
del(df_train['cii_frac'])
#df_train = df_train[['pop', 'wu_rate', 'dom_frac']]
model.add_training_df( df_train = df_train)

# make_dataset.make_ds_per_capita_basic(model, datafile=datafile)
# model.df_train['pop_density']  = model.df_train['pop']/model.df_train['WSA_SQKM']
# model.df_train.loc[model.df_train['WSA_SQKM']==0, 'pop_density'] = np.NaN
# add water use
seed1 = 123
seed2 = 456

# %%
model.apply_func(func=targets.compute_per_capita, type='target_func', args=None)

opts = ['pop<=100', 'per_capita>=500', 'per_capita<=25']
model.apply_func(func=outliers_utils.drop_values, type='outliers_func', opts = opts )
model.apply_func(func = outliers_utils.drop_na_target, type='outliers_func')

# =============================
# Feature Engineering
# =============================
# Target summary
tr_df, cat_df = featurize.summary_encode(model, cols=model.categorical_features)
df_concat = pd.concat([tr_df, cat_df], axis=1)
model.add_feature_to_skip_list( model.categorical_features)
model.add_training_df( df_train = df_concat)
del(tr_df); del(cat_df); del(df_concat)

# =============================
# Prepare the initial estimator
# =============================
params = {
    'objective': "reg:squarederror",
    'tree_method': 'hist',
    'colsample_bytree': 0.8,
    'learning_rate': 0.20,
    'max_depth': 7,
    'alpha': 100,
    'n_estimators': 500,
    'rate_drop': 0.9,
    'skip_drop': 0.5,
    'subsample': 0.8,
    'reg_lambda': 10,
    'min_child_weight': 50,
    'gamma': 10,
    'max_delta_step': 0,
    'seed': seed2
}
params2 = {
    'objective': "reg:squarederror",
    'tree_method': 'hist',
    'colsample_bytree': 0.5972129389586888,
    'learning_rate': 0.04371772054983907,
    'max_depth': 11,
    'alpha': 100,
    'n_estimators': 300,
    'subsample': 1,
    'reg_lambda': 1.2998981941716606e-08,
    'min_child_weight': 4,
    'gamma': 10,
    'max_delta_step': 0,
    'seed': seed2
}

gb = estimators.xgb_estimator(params2)

encode_cat_features = False
if encode_cat_features:
    # pipeline
    main_pipeline = pipelines.make_pipeline(model)
    main_pipeline.append(('estimator', gb))
    gb = Pipeline(main_pipeline)



#rf = xgb.XGBRFRegressor(n_estimators=500, subsample=0.8, colsample_bynode=0.5,  max_depth= 7,)
features = model.features
target = model.target

final_dataset = model.df_train
model.log.to_table(final_dataset[features].describe(percentiles = [0,0.05,0.5,0.95,1]), "Features", header=-1)
model.log.to_table(final_dataset[[target]].describe(percentiles = [0,0.05,0.5,0.95,1]), "Target", header=-1)



X_train, X_test, y_train, y_test = splittors.stratified_split(model, test_size = 0.3,  id_column = 'HUC2', seed = 123)
#X_train, X_test, y_train, y_test = splittors.split_by_id(model, args = {'frac' : 0.7, 'seed':547, 'id_column':'sys_id' })
# X_train, X_test, y_train, y_test = train_test_split(final_dataset[features],  final_dataset[target],
#                                                               test_size=0.2, random_state=123)#, shuffle = False

#w_train, w_test = weights.generate_weights_ones(model)
vv = gb.fit(X_train[features], y_train)
y_hat = gb.predict(X_test[features])
err = pd.DataFrame(y_test - y_hat)

# =============================
# initial diagnose
# =============================

heading = "Scatter Plot for Annual Water Use"
xlabel = "Actual Per Capita Water Use - Gallons"
ylabel = "Estimated Per Capita Water Use - Gallons"
figfile = os.path.join(figures_folder, "annual_one_2_one_v_0.pdf")
figures.one_to_one(y_test, y_hat, heading= heading, xlabel= xlabel, ylabel = ylabel,figfile=figfile )
model.log.info("\n\n\n ======= Initial model performance ==========")
df_metric = model_diagnose.generate_metrics(y_true=y_test, y_pred=y_hat)
model.log.info("See initial model perfromance at {}".format(figfile))
model.log.to_table(df_metric)

# regional performance
perf_df = model_diagnose.generat_metric_by_category(gb, X_test[features], y_test,features=features, category='HUC2')


figfile = os.path.join(figures_folder, "map_val_error.pdf")
figures.plot_scatter_map(X_test['LONG'], X_test['LAT'], err,
                         legend_column = 'per_capita', cmap = 'jet', title = "Per Capita WU" , figfile = figfile,
                         log_scale = False)


# plot importance
importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
for type in importance_types:
    figfile = os.path.join(figures_folder, "annual_feature_importance_{}_v_0.pdf".format(type))
    if isinstance(gb, Pipeline):
        estm = gb.named_steps["estimator"]
        gb.named_steps["preprocess"].get_feature_names()
    else:
        estm = gb

    figures.feature_importance(estm, max_num_feature = 15, type = type, figfile = figfile)

# =============================
# Simplify model by dropping features with little importance
# =============================
if 0:
    confirmed_features = selection.boruta(X=final_dataset[features], y =  final_dataset[target], estimator = rf)
if 1:
    scoring = ['r2']#, 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
    feature_importance = selection.permutation_selection(X_test, y_test, estimator = gb, scoring = scoring,
                                                n_repeats = 10, features = features)
    metrics =feature_importance['metric'].unique()
    for m in metrics:
        curr_ = feature_importance[feature_importance['metric'].isin([m])]



if 0:
    from sklearn.feature_selection import chi2
    chi_test_df = selection.chi_square_test(X = final_dataset[features],  y = final_dataset[target], nbins=20)
if 0:
    from sklearn.feature_selection import RFE, RFECV
    selector = RFECV(vv, verbose=10, scoring= 'r2')
    selector.fit(final_dataset[features], final_dataset[target])
    features_selected = selector.get_feature_names_out()
    feature_rank = selector.ranking_



confirmed_features = ['pop_density', 'gini', 'n_occupation', 'n_occ_service',
       'n_occ_sales_office', 'n_occ_farm_fish_forest', 'n_ind_construction',
       'n_ind_manufacturing', 'n_ind_wholesale_trade', 'n_ind_retail_trade',
       'n_ind_information', 'n_ind_prof_sci_admin_waste', 'households2',
       'income_35k_40k', 'income_125k_150k', 'median_income',
       'h_age_1990_1999', 'h_age_1980_1989', 'h_age_older_1939',
       'median_h_year', 'etr_warm', 'etr_cool', 'etr', 'pr_warm', 'tmmn_warm',
       'tmmn_cool', 'tmmn', 'tmmx_cool', 'HUC2', 'LAT', 'LONG', 'county_id',
       'awuds_totw_cnt', 'awuds_dom_cnt', 'awuds_pop_cnt', 'zill_nhouse',
       'LotSizeSquareFeet_sum', 'YearBuilt_mean', 'BuildingAreaSqFt_sum',
       'TaxAmount_mean', 'NoOfStories_mean', 'bdg_ftp_count', 'bdg_gt_2median',
       'bdg_gt_4median', 'bdg_lt_2median', 'bdg_lt_4median', 'av_house_age',
       'average_income', 'county_tot_pop_2010', 'pov_2019', 'income_cnty',
       'n_jobs_cnty', 'indus_cnty', 'rur_urb_cnty', 'unemployment_cnty',
       'state_id', 'Num_establishments_2012', 'Num_establishments_2017',
       'Commercial', 'Domestic', 'Industrial', 'Institutional',
       'Recreation_Misc', 'Urban_Misc', 'Production', 'Urban_Parks', 'Water',
       'WSA_SQKM', 'KG_climate_zone', 'prc_n_lt_ninth_gr',
       'prc_n_ninth_to_twelth_gr', 'prc_n_hs_grad', 'prc_n_some_college',
       'prc_n_associates', 'prc_n_bachelors', 'prc_n_masters_phd', 'pop']

# =============================
# Interpret the model
# =============================





import shap
shap.partial_dependence_plot('BuildingAreaSqFt_sum', gb.predict, X_test,ice =
            False, model_expected_value = True, feature_expected_value = True)

#X100 = shap.utils.sample(X_train, 100)
#X500 = shap.utils.sample(X_train, 5000)
#explainer = shap.Explainer(model.predict, X100)
X100 = X_train[X_train['HUC2']==18]
X100 = shap.utils.sample(X100, 2000)
explainer = shap.Explainer(vv, X100)
#explainer = shap.TreeExplainer(gb, X100)
shap_values = explainer(X100)
#shap.plots.waterfall(shap_values[150], max_display=14)
#shap.summary_plot(shap_values, X100)
shap.plots.beeswarm(shap_values, max_display=14)
shap.plots.scatter(shap_values[:,"etr"], color = shap_values)
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
from lightgbm import LGBMRegressor
lgb_params = {
    'n_jobs': 1,
    'max_depth': 8,
    'min_data_in_leaf': 10,
    'subsample': 0.8,
    'n_estimators': 500,
    'learning_rate': 0.1,
    'colsample_bytree': 0.8,
    'boosting_type': 'gbdt'
}

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
    # to train a quantile regression, we change the objective parameter and
    # specify the quantile value we're interested in
    lgb = LGBMRegressor(alpha=quantile_alpha, **lgb_params)
    lgb.fit(X_train, y_train)
    lgb_quantile_alphas[quantile_alpha] = lgb
plt.figure()
for quantile_alpha, lgb in lgb_quantile_alphas.items():
    ypredict = lgb.predict(X_test)
    plt.scatter(y_test, ypredict, s = 4, label = "{}".format
                (quantile_alpha))
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.legend()
lim = [min(y_test), max(y_test)]
plt.plot(lim, lim, 'k')
plt.xlabel("Actual Water Use")
plt.ylabel("Estimated Water Use")
