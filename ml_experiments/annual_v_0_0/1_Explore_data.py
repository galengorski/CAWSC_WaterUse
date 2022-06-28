# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Exploring Raw Water Use data
# ## Annual Water Use

# +
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import configparser
from scipy import stats
from scipy.stats import norm


# %matplotlib widget
# %matplotlib inline
# %matplotlib ipympl

# +
#m = leafmap.Map(center=(40, -100), zoom=5)
#m
# -

# This notebook is used to summarize data exploration for the water use project. To simplify the reproducability of the result, the external files needed to run the notebook correctly are saved in a configuration file.

# + tags=[]
config = configparser.ConfigParser()
config_file = r"C:\work\water_use\ml_experiments\annual_v_0_0\config_file2.ini"
config.read(config_file)


# -

# read files
workspace = config.get("Files", "Workspace")
train_file = config.get("Files", "Train_file")
target = config.get("Target", "target_field")

df_main = pd.read_csv(train_file)
df_train = df_main[df_main['wu_rate'] > 0]
df_train

# Generale look into the data
df_train.describe()

# Multiple steps were followed to clean population. So the final population feild is defined here
pop_field = 'population'

plt.figure()
sns.distplot(np.log10(df_train[target]), label = 'Total Water Use')
plt.legend()
plt.figure()
sns.distplot(np.log10(df_train[target]/df_train[pop_field]), label = 'Per Capita Water Use')
plt.legend()

# Some of the "per capita values" are very extreme, which might indicate some outliers in WU or population. For now, let us remove exteme 1% values. 

trimed_per_capita = np.log10(df_train[target]/df_train[pop_field])
low_0_01 = trimed_per_capita.quantile(0.01)
top_0_98 = trimed_per_capita.quantile(0.98)
trimed_per_capita = trimed_per_capita[trimed_per_capita >= low_0_01]
trimed_per_capita = trimed_per_capita[trimed_per_capita <=top_0_98]
sns.distplot(trimed_per_capita, label = 'Per Capita Water Use',  hist_kws={ "linewidth": 1,
                            "alpha": 0.5, "color": "g", "edgecolor" : 'black'})
plt.legend()
np.power(10,trimed_per_capita).describe()

# ### Water Use vs. Population
# Notice that some population values are less than 1. This is becuase the census data collector issue.

# + slideshow={"slide_type": "slide"}
db_ = df_train
db_ = db_[db_['population']>0]
db_["Log10 Population"] = np.log10(db_['population'])
db_["Log10 Water Use (Gallon)"] = np.log10(db_['wu_rate'])
ax1 = sns.jointplot(data=db_, x="Log10 Population", y="Log10 Water Use (Gallon)",  marginal_kws=dict(bins=100), kind="hist")#.set_title('lalala')
ax1.fig.suptitle("Water Use vs Population from Census Data Dollector")
ax1.fig.tight_layout()

#local population data
db_ = df_train
db_ = db_[db_[pop_field]>0]
db_["Log10 Population"] = np.log10(db_[pop_field])
db_["Log10 Water Use (Gallon)"] = np.log10(db_['wu_rate'])
ax2 = sns.jointplot(data=db_, x="Log10 Population", y="Log10 Water Use (Gallon)",  marginal_kws=dict(bins=100), kind="hist")
ax2.fig.suptitle("Water Use vs Population from local analysis")
ax1.fig.tight_layout()

# -

# ### Exploring population density
# * Errors in population can stem from errors in the service area boundary. 
# * The following figure indicate that population density can reach unreasonable values (1 million in km2), the error can be in population value or service area.
# * An average population density is $900$ persons in $km^2$. Densities above 2500 ( greater than Q95) is likely outliers.
# * Very low density is possible. low 1% percentile is equivelent to 8 persons in km2.  

db_ =df_train
plt.figure()
plt.scatter(np.log10(db_[pop_field]), np.log10(db_['wu_rate']), c = np.log10(db_[pop_field]/db_['WSA_SQKM']), s = 5)
plt.colorbar()
plt.show()


plt.figure()
pop_density = np.log10(db_[pop_field]/db_['WSA_SQKM'])
#plt.hist(pop_density, bins = 100);
sns.histplot(pop_density)
plt.xlabel("Log10 of Population Density")
plt.ylabel("Frequency")
plt.title("Variability of Population Density")
np.power(10,pop_density).describe()




# ### Categorical Variables
#

# #### Water use by HUC2

var = 'HUC2'
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y=target, data=df_train)
ax.set_yscale('log')

var = 'HUC2'
f, ax = plt.subplots(figsize=(8, 6))
df_train['_pc_'] = df_train[target]/df_train[pop_field]
fig = sns.boxplot(x=var, y='_pc_', data=df_train)
ax.set_yscale('log');
plt.xticks(rotation=90);

# #### Per Capita Water use by State

var = 'state_id'
f, ax = plt.subplots(figsize=(8, 6))
df_train['_pc_'] = df_train[target]/df_train[pop_field];
fig = sns.boxplot(x=var, y='_pc_', data=df_train);
ax.set_yscale('log');
plt.xticks(rotation=90);

# #### Effect of climate on Per Capita WU

var = 'KG_climate_zone'
f, ax = plt.subplots(figsize=(8, 6))
df_train['_pc_'] = df_train[target]/df_train['population']
fig = sns.boxplot(x=var, y='_pc_', data=df_train)
ax.set_yscale('log');
plt.xticks(rotation=90);

# + tags=[]
df_train['_log_pc_'] = np.log10(df_train['_pc_']);

pc25 = df_train[df_train['KG_climate_zone'] == 25]['_log_pc_'];
pc14 = df_train[df_train['KG_climate_zone'] == 14]['_log_pc_'];
pc7 = df_train[df_train['KG_climate_zone'] == 7]['_log_pc_'];
pc4 = df_train[df_train['KG_climate_zone'] == 4]['_log_pc_'];
pc8 = df_train[df_train['KG_climate_zone'] == 8]['_log_pc_'];
pc5 = df_train[df_train['KG_climate_zone'] == 5]['_log_pc_'];
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(pc25, label= 'Cold, no dry season, hot summer', rug=True, hist=False)
sns.distplot(pc7, label= 'Arid, steppe, cold', rug=True, hist=False)
sns.distplot(pc4,  label='Arid, desert, hot', rug=True, hist=False)
sns.distplot(pc5, label = 'Arid, desert, cold ', rug=True, hist=False)
sns.distplot(pc8,  label= 'Temperate, dry summer, hot summer ', rug=True, hist=False)
sns.distplot(pc14,  label= 'Temperate, no dry season, hot summer', rug=True, hist=False)
plt.xlim([-1, 3.5])
plt.legend()

# +

corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()
# -

k = 20 #number of variables for heatmap
cols = corrmat.nlargest(k, '_pc_')['_pc_'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


plt.figure()
sns.distplot(np.log10(df_train[target]), fit = norm);
fig = plt.figure()
res = stats.probplot(np.log10(df_train[target]), plot=plt)

# +
plt.figure()
sns.distplot(np.log10(df_train[target]/df_train[pop_field]), fit = norm);
fig = plt.figure()
per_Capita = np.log10(df_train[target]/df_train[pop_field])
per_Capita = per_Capita[~per_Capita.isna()]

res = stats.probplot(per_Capita, plot=plt)
# -

# ## Exploring Monthly Data

# ## Exploring Daily Data


