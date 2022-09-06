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
from matplotlib import pyplot as plt

# load dataset
df_feat = pd.read_csv(
    r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\wu_annual_training.csv"
)
if 0:
    cols_to_del = [
        "is_swud",
        "Num_establishments_2017",
        "NoData",
        "Ecode",
        "sys_id",
        "fips",
    ]
    cols_to_del2 = [
        "population",
        "swud_pop",
        "small_gb_pop",
        "pc_swud",
        "pc_tract_data",
        "pop_swud_corrected",
        "swud_corr_factor",
        "pc_swud_corrected",
        "pc_gb_data",
        "pop_swud_gb_correction",
        "pc_swud_gb_corrected",
        "bg_pop_2010",
        "bg_usgs_correction_factor",
        "bg_theo_correction_factor",
        "ratio_lu",
        "pop_urb",
        "ratio_2010",
        "swud_pop_ratio",
    ]
    cat_feat = ["Ecode_num", "HUC2", "county_id"]

    # for now delete all
    for f in cols_to_del:
        del df_feat[f]
    for f in cols_to_del2:
        del df_feat[f]
    for f in cat_feat:
        del df_feat[f]

# 2010 ->
# 2015 -> 4 states in swud
# OH - unit issues
df_feat = df_feat[df_feat["wu_rate"] > 0]
county_awuds = df_feat.groupby(by=["county_id"]).mean()
county_wsa = df_feat.groupby(by=["sys_id", "county_id"]).mean()
county_wsa.reset_index(inplace=True)
county_wsa = county_wsa.groupby(by=["county_id"]).sum()
xx = 1
