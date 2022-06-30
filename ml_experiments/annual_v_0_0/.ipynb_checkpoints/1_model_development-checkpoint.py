#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os, sys
import pandas as pd
from pycaret.regression import *


# %%


dataset = pd.read_csv(r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv")
dataset.shape


# %%


data = dataset.sample(frac=0.8, random_state=786)
data_unseen = dataset.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# %%


exp_reg101 = setup(data = data, target = 'wu_rate', fold_shuffle= True, data_split_shuffle=True, session_id=123 )
xx = 1

# %%




