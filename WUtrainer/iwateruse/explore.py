import os, sys
import configparser
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

config = configparser.ConfigParser()
config.read(r"C:\work\water_use\ml_experiments\annual_v_0_0\config_file.ini")

workspace = config.get("Files", "Workspace")
train_file = config.get("Files", "Train_file")
target = config.get("Target", "target_field")

df_main = pd.read_csv(train_file)
df_train = df_main[df_main['wu_rate'] > 0]

# Generale look into the data
df_train.describe()
df_train[target].describe()

sns.distplot(np.log10(df_train[target]))
sns.distplot(np.log10(df_train[target]/df_train['TPOPSRV']))

# Relationship with categorical features
var = 'HUC2'
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y=target, data=df_train)

var = 'state_id'
f, ax = plt.subplots(figsize=(8, 6))
df_train['_pc_'] = df_train[target]/df_train['TPOPSRV']
fig = sns.boxplot(x=var, y='_pc_', data=df_train)

var = 'HUC2'
f, ax = plt.subplots(figsize=(8, 6))
df_train['_pc_'] = df_train[target]/df_train['TPOPSRV']
fig = sns.boxplot(x=var, y='_pc_', data=df_train)

var = 'KG_climate_zone'
f, ax = plt.subplots(figsize=(8, 6))
df_train['_pc_'] = df_train[target]/df_train['TPOPSRV']
fig = sns.boxplot(x=var, y='_pc_', data=df_train)

# Find NaN in features. Todo: also find NaN in df_main
df_train['_log_pc_'] = np.log10(df_train['_pc_'])

pc25 = df_train[df_train['KG_climate_zone'] == 25]['_log_pc_']
pc14 = df_train[df_train['KG_climate_zone'] == 14]['_log_pc_']
pc7 = df_train[df_train['KG_climate_zone'] == 7]['_log_pc_']
pc4 = df_train[df_train['KG_climate_zone'] == 4]['_log_pc_']
pc8 = df_train[df_train['KG_climate_zone'] == 8]['_log_pc_']
pc5 = df_train[df_train['KG_climate_zone'] == 5]['_log_pc_']
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(pc25, label= 'Cold, no dry season, hot summer', rug=True, hist=False)
sns.distplot(pc7, label= 'Arid, steppe, cold', rug=True, hist=False)
sns.distplot(pc4,  label='Arid, desert, hot', rug=True, hist=False)
sns.distplot(pc5, label = 'Arid, desert, cold ', rug=True, hist=False)
sns.distplot(pc8,  label= 'Temperate, dry summer, hot summer ', rug=True, hist=False)
sns.distplot(pc14,  label= 'Temperate, no dry season, hot summer', rug=True, hist=False)

corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#
k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, '_pc_')['_pc_'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# socio economic data

sns.set()
cols = [val for val in df_train.columns if "income" in val]
sns.pairplot(df_train[cols], size = 2.5)
plt.show();

# gaps
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


#histogram and normal probability plot
plt.figure()
sns.distplot(np.log10(df_train[target]), fit=norm);
fig = plt.figure()
res = stats.probplot(np.log10(df_train[target]), plot=plt)

plt.figure()
sns.distplot(np.log10(df_train["_pc_"]), fit=norm);
fig = plt.figure()
res = stats.probplot(np.log10(df_train["_pc_"]), plot=plt)

plt.figure()
sns.distplot(np.log10(df_train["TPOPSRV"]), fit=norm);
fig = plt.figure()
res = stats.probplot(np.log10(df_train["TPOPSRV"]), plot=plt)


#house age
data = df_train[['h_age_newer_2005','h_age_2000_2004','h_age_1990_1999',	'h_age_1980_1989',
                 'h_age_1970_1979',	'h_age_1960_1969','h_age_1950_1959',	'h_age_1940_1949',	'h_age_older_1939', 'wu_rate']]
sns.set()
sns.pairplot(data, size = 2.5)
plt.show();

x= 1



