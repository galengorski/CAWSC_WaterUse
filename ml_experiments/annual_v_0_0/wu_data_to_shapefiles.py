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


df_full = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\wu_annual_training3.csv")
# =============================
# Setup Training
# =============================
figures_folder = "figs"
model = Model(name='annual_pc', log_file = 'train_log.log',  feature_status_file= r"features_status.xlsx")
model.raw_target = 'wu_rate'
model.target = 'per_capita'

datafile = r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv" #clean_train_db_backup_6_24_2022.csv
df_train = pd.read_csv(datafile)
dfXY = df_train[['sys_id', 'LAT', 'LONG', 'wu_rate']]
dfXY = dfXY.drop_duplicates(subset = ['sys_id'])
dfXY.to_csv("annual_wu_XY.csv")

# =============================
# Monthly
# =============================
dataset = pd.read_csv(r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv")
monthly_wu = pd.read_csv(r"C:\work\water_use\CAWSC_WaterUse\etc\monthly_wu.csv")
monthly_wu = monthly_wu[monthly_wu['inWSA']>0]
monthly_sys = monthly_wu['WSA_AGIDF'].unique()
monthly_sysXY = df_full[df_full['sys_id'].isin(monthly_sys)]
monthly_sysXY = monthly_sysXY[['sys_id', 'LAT', 'LONG']]
monthly_sysXY = monthly_sysXY.drop_duplicates(subset = ['sys_id'])
monthly_sysXY.to_csv("monthly_wu_XY.csv")
xx = 1