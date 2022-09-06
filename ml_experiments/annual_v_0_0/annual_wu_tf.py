import pandas as pd
from numpy import sqrt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import tensorflow as tf

tf.random.set_seed(11)
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

# # load dataset
# df_feat =  pd.read_csv(r"C:\work\water_use\ml_experiments\annual_v_0_0\wu_annual_training.csv")
#
# cols_to_del = ['is_swud', 'Num_establishments_2017', 'NoData',  'Ecode', 'sys_id', 'fips']
# cols_to_del2 = ['population', 'swud_pop', 'small_gb_pop', 'pc_swud', 'pc_tract_data',
# 				'pop_swud_corrected',	'swud_corr_factor',	'pc_swud_corrected',
# 				'pc_gb_data', 'pop_swud_gb_correction',	'pc_swud_gb_corrected',
# 				'bg_pop_2010',	'bg_usgs_correction_factor',	'bg_theo_correction_factor',
# 				'ratio_lu',	'pop_urb',	'ratio_2010', 'swud_pop_ratio']
# cat_feat = ['Ecode_num', 'HUC2', 'county_id']
#
# # for now delete all
# for f in cols_to_del:
# 	del(df_feat[f])
# for f in cols_to_del2:
# 	del(df_feat[f])
# for f in cat_feat:
# 	del(df_feat[f])
#
#
# df_feat = df_feat[df_feat['wu_rate']>0]
#
#
# ##### outliears
# df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
# df_feat.dropna(axis=1, how='all')
# df_feat = df_feat.dropna(axis=0)
# if 0:
#
# 	from sklearn.ensemble import IsolationForest
# 	import numpy as np
# 	X = df_feat.values
# 	clf = IsolationForest(n_estimators=10, warm_start=True)
# 	clf.fit(X)  # fit 10 trees
# 	clf.set_params(n_estimators=20)  # add 10 more trees
# 	mask1 = clf.fit_predict(X)  # fit the added trees
# ################33
# if 0:
# 	from sklearn.neighbors import LocalOutlierFactor
# 	clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# 	# use fit_predict to compute the predicted labels of the training samples
# 	# (when LOF is used for outlier detection, the estimator has no predict,
# 	# decision_function and score_samples methods).
# 	X = df_feat.values
# 	mask2 = clf.fit_predict(X)
# 	##################333
# 	XX = df_feat[['wu_rate', 'LUpop_Swudpop', 'median_income',
# 				  'median_h_year', 'tot_h_age', 'Commercial',
# 				  'Conservation',	'Domestic',	'Industrial',	'Institutional',
# 				  'Recreation_Misc',	'Urban_Misc',	'Production']]
# 	XX = XX.values
#
# 	from sklearn.svm import OneClassSVM
# 	clf = OneClassSVM(gamma='auto')
# 	flg = clf.fit_predict(XX)
#
# 	#mask3 = clf.score_samples(X)
#
#
# df_feat = df_feat[df_feat['LUpop_Swudpop']>5000]
#
#
# pc = df_feat['wu_rate']/df_feat['LUpop_Swudpop']
# #df_feat['LUpop_Swudpop'] = np.log10(df_feat['LUpop_Swudpop'])
# df_feat = df_feat[pc<pc.quantile(0.90)]
# df_feat = df_feat[pc>pc.quantile(0.1)]
# df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
# df_feat.dropna(axis=1, how='all')
# df_feat = df_feat.dropna(axis=0)
#
#
#
# pc = df_feat['wu_rate']/df_feat['LUpop_Swudpop']
#
#
# # split into input and output columns
# y = df_feat['wu_rate'].values
# #y = pc.values
# #y = np.log10(df_feat['wu_rate'].values)
# del(df_feat['wu_rate'])
# X = df_feat.values
# # split into train and test datasets
# X_train, X_test, y_train, y_test = train_test_split(X, y,  random_state=123, test_size=0.33)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# #scaler = RobustScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# # determine the number of input features
# n_features = X_train.shape[1]
# # define model

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

warnings.filterwarnings("ignore")
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
model = Model(
    name="annual_pc",
    log_file="train_log.log",
    feature_status_file=r"features_status.xlsx",
)
model.raw_target = "wu_rate"
model.target = "per_capita"

datafile = r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv"
make_dataset.make_ds_per_capita_basic(model, datafile=datafile)

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
final_dataset["id"] = final_dataset.index.values
if 0:  # todo
    final_dataset = final_dataset.drop_duplicates(
        subset=["sys_id", "Year"], keep="first"
    )
model.log.to_table(final_dataset[features], "Features")
model.log.to_table(final_dataset[[target]], "Target")
if 1:
    outlier_info = pd.read_csv(
        r"C:\work\water_use\ml_experiments\annual_v_0_0\Outliers_6_1.csv"
    )
    ids = []
    for col in outlier_info.columns:
        if col.isdigit():
            ids.append(col)
    if 1:
        iter = 200
        sig_ids = outlier_info.loc[outlier_info["iter"] == iter, ids]
        sig_ids = sig_ids.T
        final_dataset = final_dataset[sig_ids[iter].values == 1]
X_train, X_test, y_train, y_test = train_test_split(
    final_dataset[features],
    final_dataset[target],
    test_size=0.3,
    random_state=123,
)
# gb.fit(X_train, y_train)
# y_hat = gb.predict(X_test)

model = Sequential()
act1 = "relu"
act2 = "sigmoid"
act3 = "linear"
act4 = "tanh"
# act = tf.keras.layers.LeakyReLU(alpha=0.3)

model.add(
    Dense(
        200,
        activation=act4,
        kernel_initializer="he_normal",
        input_shape=(len(features),),
    )
)
model.add(Dense(200, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(100, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(100, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(50, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(50, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(5, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(50, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(50, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(100, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(100, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(200, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(200, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(5, activation=act1, kernel_initializer="he_normal"))
model.add(Dense(1, activation="linear"))
# compile the model
loss = tf.keras.losses.MeanSquaredError()
solver = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)
import tensorflow_probability as tfp


def custom_loss_function(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    p95 = tfp.stats.percentile(squared_difference, 95)
    squared_difference = squared_difference[squared_difference < p95]
    return tf.reduce_mean(squared_difference)


# loss = 'mse'
model.compile(loss=loss, metrics=["accuracy"], optimizer=solver)
# fit the model
history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=100,
    validation_split=0.2,
    verbose=2,
)


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["loss", "val"], loc="upper left")


# evaluate the model
error = model.evaluate(X_test, y_test, verbose=2)
# make a prediction

yhat = model.predict([X_test])
plt.figure()
plt.scatter(y_test, yhat)
from sklearn.metrics import r2_score

accuracy = r2_score(y_test, yhat)
c = [min(y_test), max(y_test)]
plt.plot(c, c, "r")
plt.title(str(accuracy))
plt.show()
# print('Predicted: %.3f' % yhat)
Pxx = 1
