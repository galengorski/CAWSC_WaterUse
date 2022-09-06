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
from iwateruse import (
    data_cleaning,
    report,
    splittors,
    pre_train_utils,
    make_dataset,
    figures,
)
from iwateruse import denoise, model_diagnose

import matplotlib.pyplot as plt
from xgboost import plot_importance
import xgboost as xgb

#
import numpy as np
from flopy.plot import styles


import warnings

warnings.filterwarnings("ignore")
xgb.set_config(verbosity=0)

# %%
from iwateruse.model import Model
from iwateruse import targets, weights, pipelines, outliers_utils, estimators
from iwateruse import selection
from pycaret.regression import *


# %%
# =============================
# Setup Training
# =============================
figures_folder = "figs"
model = Model(
    name="annual_pc",
    log_file="train_log.log",
    feature_status_file=r"..\features_status.xlsx",
)
model.raw_target = "wu_rate"
model.target = "per_capita"

datafile = r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv"
make_dataset.make_ds_per_capita_basic(model, datafile=datafile)
model.df_train["pop_density"] = (
    model.df_train["pop"] / model.df_train["WSA_SQKM"]
)
model.df_train.loc[model.df_train["WSA_SQKM"] == 0, "pop_density"] = 0
# add water use
seed1 = 123
seed2 = 456

# %%
model.apply_func(
    func=targets.compute_per_capita, type="target_func", args=None
)

opts = ["pop<=100", "per_capita>=500", "per_capita<=25"]
model.apply_func(
    func=outliers_utils.drop_values, type="outliers_func", opts=opts
)
model.apply_func(func=outliers_utils.drop_na_target, type="outliers_func")
model.apply_func(func=None, type="add_features_func", args=None)

# split
model.apply_func(
    func=splittors.random_split, args={"frac": 0.70, "seed": seed1}
)


# =============================
# Prepare the initial estimator
# =============================

features = model.features
target = model.target
final_dataset = model.df_train
final_dataset = final_dataset.drop_duplicates(
    subset=["sys_id", "Year"], keep="first"
)
ignore_features = list(
    set(final_dataset.columns).difference(set(features + [target]))
)

reg1 = setup(
    data=final_dataset,
    target=target,
    ignore_features=ignore_features,
    fold=5,
    fold_shuffle=True,
    train_size=0.7,
)

# compare models
learning_algorithems = ["xgboost", "rf"]  # , 'lightgbm', 'et'
best = compare_models(include=learning_algorithems, n_select=3)

xx = 1

tuned_dt = tune_model(xgb)
