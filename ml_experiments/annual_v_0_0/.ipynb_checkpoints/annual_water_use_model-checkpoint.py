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
import matplotlib
matplotlib.use('Qt5Agg')
import warnings

warnings.filterwarnings('ignore')
xgb.set_config(verbosity=0)

# %%
from iwateruse.model import Model
from iwateruse import targets, weights, pipelines, outliers_utils, estimators
from iwateruse import selection


# %%
# =============================
# Setup Training
# =============================
figures_folder = "figs"
model = Model(name='annual_pc', log_file = 'train_log.log',  feature_status_file= r"features_status.xlsx")
model.raw_target = 'wu_rate'
model.target = 'per_capita'

datafile = r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv"
make_dataset.make_ds_per_capita_basic(model, datafile=datafile)
model.df_train['pop_density']  = model.df_train['pop']/model.df_train['WSA_SQKM']
model.df_train.loc[model.df_train['WSA_SQKM']==0, 'pop_density'] = 0
# add water use
seed1 = 123
seed2 = 456

# %%
model.apply_func(func=targets.compute_per_capita, type='target_func', args=None)

opts = ['pop<=100', 'per_capita>=500', 'per_capita<=25']
model.apply_func(func=outliers_utils.drop_values, type='outliers_func', opts = opts )
model.apply_func(func = outliers_utils.drop_na_target, type='outliers_func')
model.apply_func(func=None, type='add_features_func', args=None)

# split
model.apply_func(func=splittors.random_split, args={'frac': 0.70, 'seed': seed1})


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
gb = estimators.xgb_estimator(params)
rf = xgb.XGBRFRegressor(n_estimators=500, subsample=0.8, colsample_bynode=0.5,  max_depth= 7,)
features = model.features
target = model.target

final_dataset = model.df_train
final_dataset = final_dataset.drop_duplicates(subset = ['sys_id', 'Year'], keep = 'first')
model.log.to_table(final_dataset[features], "Features")
model.log.to_table(final_dataset[[target]], "Target")
X_train, X_test, y_train, y_test = train_test_split(final_dataset[features],  final_dataset[target],
                                                            test_size=0.3, random_state=123)
vv = gb.fit(X_train, y_train)
y_hat = gb.predict(X_test)

# =============================
# initial diagnose
# =============================

heading = "Scatter Plot for Annual Water Use"
xlabel = "Actual Per Capita Water Use - Gallons"
ylabel = "Estimated Per Capita Water Use - Gallons"
figfile = os.path.join(figures_folder, "an_1_1")
figures.one_to_one(y_test, y_hat, heading= heading, xlabel= xlabel, ylabel = ylabel,figfile=figfile )
model.log.info("\n\n\n ======= Initial model performance ==========")
df_metric = model_diagnose.generate_metrics(y_true=y_test, y_pred=y_hat)
model.log.info("See initial model perfromance at {}".format(figfile))
model.log.to_table(df_metric)


# =============================
# Simplify model by dropping features with little importance
# =============================
if 0:
    confirmed_features = selection.boruta(X=final_dataset[features], y =  final_dataset[target], estimator = rf)
if 1:
    scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
    feature_importance = selection.permutation_selection(X_test, y_test, estimator = gb, scoring = scoring,
                                                n_repeats = 10, features = features)
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
