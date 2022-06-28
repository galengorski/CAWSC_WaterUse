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

from iwateruse.featurize import MultiOneHotEncoder
from iwateruse import data_cleaning, report, splittors, pre_train_utils

import matplotlib.pyplot as plt
from xgboost import plot_importance
import xgboost as xgb

#
import numpy as np

# sklearn
from sklearn.metrics import r2_score

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
from iwateruse import targets, weights, pipelines, outliers_utils, estimators


# %%
raw_train_df = pd.read_csv(r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv")
model = Model(name='annual_pc', df_train= raw_train_df, feature_status_file= r"features_status.xlsx")


#model.df_train = model.df_train_bk.copy()
model.raw_target = 'wu_rate'
model.target = 'per_capita'
#model.categorical_features = []  # KG_climate_zone', 'HUC2', 'state_id', 'Ecode_num', 'county_id'


pc_50_swud = pd.read_csv(r"spatial_pc_statistics_apply_max_min_pc.csv")
annual_wu = pd.read_csv(r"annual_wu.csv")
annual_wu['wu_rate_mean'] = annual_wu[['annual_wu_G_swuds', 'annual_wu_G_nonswuds']].mean(axis=1)
annual_wu['wu_rate_mean'] = annual_wu['wu_rate_mean'] / annual_wu['days_in_year']
avg_wu = annual_wu[['WSA_AGIDF', 'YEAR', 'wu_rate_mean']].copy()
avg_wu.rename(columns={'WSA_AGIDF': 'sys_id', 'YEAR': 'Year', 'wu_rate_mean': 'wu_rate'}, inplace=True)

# %%


# add water use
del (model.df_train['wu_rate'])
model.df_train = model.df_train.merge(avg_wu, on=['sys_id', 'Year'], how='left')
model.df_train_bk = model.df_train[model.df_train['wu_rate'] > 0]

model.df_train = model.df_train_bk

if 0:
    # upscale
    from upscale_dataset import up_scale_data

    agg_rules = pd.read_csv(r"aggregation_roles.csv")
    df_agg = up_scale_data(df=model.df_train, groupby_col=["county_id", "Year"], skip_cols=[], agg_rules=agg_rules)
    model.df_train = df_agg
    model.df_train_bk = df_agg

# %%

model.apply_func(func=targets.compute_per_capita, type='target_func', args=None)
model.apply_func(func=outliers_utils.outliers_func, type='outliers_func', args=None)
model.apply_func(func=None, type='add_features_func', args=None)

# split
model.apply_func(func=splittors.random_split, args={'frac': 0.7, 'seed': 123})
# model.apply_func(func=splittors.split_by_id,
#                  args={'id_column': 'sys_id', 'frac': 0.7, 'seed': 123})
# model.apply_func(func=splittors.stratified_split,
#                      args={'id_column': 'state_id', 'frac': 0.7, 'seed': 123})


main_pipeline = pipelines.make_pipeline(model)
gb = estimators.xgb_estimator()
# gb = lightGB_estimator()
# gb = xgb_estimator_deep()

main_pipeline.append(('estimator', gb))
estimator = Pipeline(main_pipeline)

model.apply_func(func=pre_train_utils.pre_train, type='', args=None)

w_train, w_test = weights.generate_weights_ones(model)
kwargs = {estimator.steps[-1][0] + '__sample_weight': w_train}
estimator.fit(model.X_train, model.y_train, **kwargs)



# use purified data


# Available importance_types = [‘weight’, ‘gain’, ‘cover’, ‘total_gain’, ‘total_cover’]
# ***************************


plt.figure()
ypredict = estimator.predict(model.X_test)

accuracy = r2_score(model.y_test, ypredict, sample_weight=w_test)
accuracy

plt.scatter(model.y_test, ypredict, s=4, c=np.log10(model.X_test['pop']), cmap='jet')
plt.plot([min(model.y_test), max(model.y_test)], [min(model.y_test), max(model.y_test)], 'r')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# interpolation
from sklearn.preprocessing import PolynomialFeatures

XY_train = model.X_train[['LONG', 'LAT']]
XY_test = model.X_test[['LONG', 'LAT']]
poly_reg = PolynomialFeatures(degree=3, include_bias=True)
X_train = poly_reg.fit_transform(XY_train)
X_test = poly_reg.transform(XY_test)

xgb_params = {
    'objective': "reg:squarederror",
    # 'tree_method': 'hist',
    # 'colsample_bytree': 0.5,
    'learning_rate': 0.1,
    'max_depth': 7,
    # 'alpha': 5,
    'n_estimators': 500,
    # 'rate_drop': 0.9,
    # 'skip_drop': 0.5,
    'subsample': 0.5,
    # 'reg_lambda': 0,
    'min_child_weight': 10,
    # 'gamma': 0,
    # 'max_delta_step': 0
}
# xgb_params = {}
gb_interpolate = xgb.XGBRegressor(seed=123, **xgb_params)
gb_interpolate.fit(X_train, model.y_train)
ypredict = gb_interpolate.predict(X_test)
accuracy = r2_score(model.y_test, ypredict, sample_weight=w_test)
accuracy

plt.scatter(model.y_test, ypredict, s=4, c=np.log10(model.X_test['pop']), cmap='jet')
plt.plot([min(model.y_test), max(model.y_test)], [min(model.y_test), max(model.y_test)], 'r')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

plt.figure()
plt.scatter(np.log10(y_test * X_test['pop']), np.log10(ypredict * X_test['pop']), s=4)
accuracy = r2_score(np.log10(y_test * X_test['pop']), np.log10(ypredict * X_test['pop']))
# plt.plot([min(y_test*X_test['pop']), max(ypredict*X_test['pop'])], [min(y_test*X_test['pop']), max(y_test)*X_test['pop']], 'r')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()

plt.colorbar()
plt.show()

# %%
plt.figure()
plt.scatter(X_train['LONG'], X_train['LAT'], s=5)
plt.scatter(X_test['LONG'], X_test['LAT'], s=5)

# %%
e = np.abs(y_test - ypredict)
mask = e > 150

# %%
plt.figure()
plt.scatter(X_test[mask]['LONG'], X_test[mask]['LAT'], c=X_test[mask]['pop'], cmap='jet', s=4)
plt.colorbar()

# %%
X_test.LONG

# %%
features = X_test.columns
imp = []
for feature in features:
    X_test_permutated = X_test.copy()
    val = X_test_permutated[feature].sample(frac=1)
    val = val.values
    X_test_permutated[feature] = val
    y_predicted_per = model.predict(X_test_permutated)
    accuracy = r2_score(y_test, y_predicted_per)
    print(accuracy)
    imp.append(accuracy - 0.7714)

# %%

# %%

# %%

# %%
from statsmodels.distributions.empirical_distribution import ECDF

# %%

# %%
x = df['pc'].values
ec = ECDF(x)

# %%
ec(x)

# %%
ytrain_predicted = model.predict(X_train)
plt.figure()

accuracy = r2_score(y_train, ytrain_predicted)
accuracy

plt.scatter(y_train, ytrain_predicted, s=4)
plt.plot([min(y_train), max(y_train)], [min(ytrain_predicted), max(ytrain_predicted)], 'r')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()

plt.colorbar()
plt.show()

# %% [markdown]
# ## Meaure Importance 

# %%
features = X_test.columns
imp = []
for feature in features:
    X_test_permutated = X_test.copy()
    val = X_test_permutated[feature].sample(frac=1)
    val = val.values
    X_test_permutated[feature] = val
    y_predicted_per = model.predict(X_test_permutated)
    accuracy = r2_score(y_test, y_predicted_per)
    print(accuracy)
    imp.append(accuracy - 0.7714)

imp = np.array(imp)
plt.figure()
plt.bar(features[np.argsort(imp)][0:50], imp[np.argsort(imp)[0:50]])

# %%
ordered_features = features[np.argsort(imp)]

# %%
features = np.array(features)
feat_to_drop = features[np.argsort(imp)][70:]
feat_to_drop = feat_to_drop.tolist()

# %%
feat_to_drop = feat_to_drop.tolist()

# %%
feat_to_drop

# %%
feat_to_drop.remove('KG_climate_zone')

# %%
plt.figure()
plt.bar(features[np.argsort(imp)][0:50], imp[np.argsort(imp)[0:50]])

# %%
imp[np.argsort(imp)[0:50]]

# %%
len(features) - 60

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.decomposition import PCA

pipeline = Pipeline([('disc', KBinsDiscretizer(n_bins=1000, encode='ordinal', strategy='uniform')),
                     ('model', gb)])
kwargs = {pipeline.steps[-1][0] + '__sample_weight': w / w}
pipeline.fit(X_train, y_train, **kwargs)

plt.figure()
ypredict = pipeline.predict(X_test)

w2 = 1.0 / Test_freq
# w2 = w2/np.sum(w)
w2 = w2 / np.max(w2)

accuracy = r2_score(y_test, ypredict, sample_weight=w2 / w2)
accuracy

plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# %% [markdown]
# ## Pipeline
# ### Categorical encoding

# %%
# (A.2) transformation of categorical features
categorical_features = ['HUC2', 'state_id', 'KG_climate_zone', 'county_id']
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# %%
# drop_before_preprocess = ['sys_id']
# categorical_features = ['HUC2', 'state_id',  'KG_climate_zone', 'county_id' ]
# ohc1 = MultiOneHotEncoder(catfeatures = categorical_features )
# dataset = ohc1.transform(dataset)

# %%
# for i in dataset.columns:
#     print(i)

# %% [markdown]
# # Estimating Water Use without Log transformation

# %%


# %%
# columns to drop
columns_to_drop = ['population', 'sys_id', 'pc']
df = df_train.copy()
df['pc'] = df['wu_rate'] / df['pop']
df = df[df['pop'] > 100]
mask = (df['pc'] > 20) & (df['pc'] < 500)
df = df[mask]

df = df.drop(columns_to_drop, axis=1)
features = list(df.columns)
features.remove('wu_rate')
X = df[features]
y = df['wu_rate']

# %%
df.columns

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# %%
# squaredlogerror
# squarederror
gb = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.8, learning_rate=0.1,
                      max_depth=7, alpha=0.01, n_estimators=500, rate_drop=0.9, skip_drop=0.5, subsample=0.8,
                      seed=123, reg_lambda=0.0)

# %%
gb.fit(X_train, y_train)

# %%
ypredict = gb.predict(X_test)
accuracy = r2_score(y_test, ypredict)

# %%
plt.figure()
accuracy = int(accuracy * 100) / 100.0
plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual Water Use")
plt.ylabel("Estimated Water Use")
plt.grid()
plt.show()

# %% [markdown]
# ## Log water use & population 

# %%
# columns to drop
columns_to_drop = ['population', 'sys_id', 'pc']
df = df_train.copy()
df = df[df['wu_rate'] > 100]
df['pc'] = df['wu_rate'] / df['pop']
df = df[df['pop'] > 100]
mask = (df['pc'] > 20) & (df['pc'] < 500)
df = df[mask]

df = df.drop(columns_to_drop, axis=1)
df['pop'] = np.log10(df['pop'])
df['wu_rate'] = np.log10(df['wu_rate'])
features = list(df.columns)
features.remove('wu_rate')

X = df[features]
y = df['wu_rate']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# %%
gb = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.8, learning_rate=0.01,
                      max_depth=7, alpha=0.01, n_estimators=500, rate_drop=0.9, skip_drop=0.5, subsample=0.8,
                      seed=123, reg_lambda=0.0)

# %%
gb.fit(X_train, y_train)

# %%
ypredict = gb.predict(X_test)
accuracy = r2_score(y_test, ypredict)
accuracy

# %%
plt.figure()
accuracy = int(accuracy * 100) / 100.0
plt.scatter(10 ** y_test, 10 ** ypredict, s=4)
plt.plot([min(10 ** y_test), max(10 ** y_test)], [min(10 ** y_test), max(10 ** y_test)], 'r')
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual Water Use")
plt.ylabel("Estimated Water Use")
plt.grid()
plt.show()

# %%
plt.figure()
pc_test = (10 ** y_test) / (10 ** X_test['pop'])
pc_predict = (10 ** ypredict) / (10 ** X_test['pop'])
accuracy = r2_score(pc_test, pc_predict)
plt.scatter(pc_test, pc_predict, s=4)

plt.plot([min(pc_test), max(pc_test)], [min(pc_test), max(pc_test)], 'r')
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual Water Use")
plt.ylabel("Estimated Water Use")
plt.grid()
plt.show()


# %% [markdown]
# ## Quantile Regression

# %% [markdown]
# ## Use Xgboost

# %%
# # columns to drop
# columns_to_drop = ['population', 'sys_id', 'pc']
# df = dataset.copy()
# df = df[df['wu_rate']>0]
# df['pc'] = df['wu_rate']/df['pop']
# df = df[df['pop']>1000]
# mask = (df['pc']>20) & (df['pc']<400)
# df = df[mask]

# df = df.drop(columns_to_drop, axis=1)
# df['pop'] = np.log10(df['pop'])
# df['wu_rate'] = np.log10(df['wu_rate'])
# features = list(df.columns)
# features.remove('wu_rate')


# X =df[features]
# y = df['wu_rate']

# %%
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 123)

# %%
def myfunc(quantile):
    def _myfunc(y_true, y_pred):
        errors = y_pred - y_true
        left_mask = errors < 0
        right_mask = errors > 0

        grad = -quantile * left_mask + (1 - quantile) * right_mask
        hess = np.ones_like(y_pred)

        return grad, hess

    return _myfunc


# %%
# log cosh quantile is a regularized quantile loss function
def log_cosh_quantile(alpha):
    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err) ** 2
        hess[hess < 0.01] = 0.01
        # hess = np.ones_like(hess)

        return grad, hess

    return _log_cosh_quantile


# %%
1 / np.cosh(-1000) ** 2


# %%
def original_quantile_loss(alpha, delta):
    def _original_quantile_loss(y_true, y_pred):
        x = y_true - y_pred
        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
                (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta
        return grad, hess

    return _original_quantile_loss


# %%
quantile_alphas = [0.1, 0.5, 0.90]

xgb_quantile_alphas = {}
for quantile_alpha in quantile_alphas:
    # to train a quantile regression, we change the objective parameter and
    # specify the quantile value we're interested in
    gb = xgb.XGBRegressor(objective=log_cosh_quantile(quantile_alpha), colsample_bytree=0.8, learning_rate=0.05,
                          max_depth=5, alpha=0.01, n_estimators=500, rate_drop=0.9, skip_drop=0.5, subsample=0.8,
                          seed=323, reg_lambda=0.01)
    gb.fit(X_train, y_train)
    xgb_quantile_alphas[quantile_alpha] = gb

# %%
plt.figure()
for quantile_alpha, lgb in xgb_quantile_alphas.items():
    ypredict = lgb.predict(X_test)
    plt.scatter(10 ** y_test, 10 ** ypredict, s=4, label="{}".format
    (quantile_alpha))
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.legend()
lim = [min(10 ** y_test), max(10 ** y_test)]
plt.plot(lim, lim, 'k')
plt.xlabel("Actual Water Use")
plt.ylabel("Estimated Water Use")

# %%

# %%


# %%

# %% [markdown]
# ## Use Ligh GBM

# %%

# %%
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

# %%
quantile_alphas = [0.1, 0.5, 0.90]

lgb_quantile_alphas = {}
for quantile_alpha in quantile_alphas:
    # to train a quantile regression, we change the objective parameter and
    # specify the quantile value we're interested in
    lgb = LGBMRegressor(objective='quantile', alpha=quantile_alpha, **lgb_params)
    lgb.fit(X_train, y_train)
    lgb_quantile_alphas[quantile_alpha] = lgb

# %%
plt.figure()
for quantile_alpha, lgb in lgb_quantile_alphas.items():
    ypredict = lgb.predict(X_test)
    plt.scatter(y_test, ypredict, s=4, label="{}".format
    (quantile_alpha))
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.legend()
lim = [min(y_test), max(y_test)]
plt.plot(lim, lim, 'k')
plt.xlabel("Actual Water Use")
plt.ylabel("Estimated Water Use")

# %% [markdown]
# # Dimension-reduction

# %%
columns_to_drop = ['population', 'sys_id', 'wu_rate', 'pop', 'households2', 'n_employed', 'n_houses']
df = df_train.copy()

df['pc'] = df['wu_rate'] / df['pop']
df = df[df['pop'] > 100]
mask = (df['pc'] > 20) & (df['pc'] < 500)
df = df[mask]

df = df.drop(columns_to_drop, axis=1)
features = list(df.columns)
features.remove('pc')
X = df[features]
y = df['pc']

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=10, whiten=True)
pca.fit(X)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

# %%
pca.transform(X)


# %% [markdown]
# # Testing more ideas for estimating Per Capita 

# %%
def make_dataset(dataset=df_train, filters=[], drop_cols=[]):
    pass


# %% [markdown]
# ## No action

# %%

columns_to_drop = ['population', 'sys_id', 'wu_rate']
df = df_train.copy()

df['pc'] = df['wu_rate'] / df['pop']
df = df[df['pop'] > 100]
mask = (df['pc'] > 20) & (df['pc'] < 500)
df = df[mask]

df = df.drop(columns_to_drop, axis=1)
features = list(df.columns)
features.remove('pc')
X = df[features]
y = df['pc']

# squarederror
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
gb = xgb.XGBRegressor(objective="reg:squarederror", colsample_bytree=0.8, learning_rate=0.1,
                      max_depth=7, alpha=0.01, n_estimators=500, rate_drop=0.9, skip_drop=0.5, subsample=0.8,
                      seed=123, reg_lambda=0.0)

w = np.exp(1e-2 * np.power((y_train - 200) / 10, 2.0))
w = w / np.sum(w)
w = w.values
w = w / np.max(w)
gb.fit(X_train, y_train, sample_weight=w / w)
plt.figure()
ypredict = gb.predict(X_test)

w2 = np.exp(1e-3 * np.power((y_test - 200) / 10, 2.0))
w2 = w2 / np.sum(w)
w2 = w2.values
w2 = w2 / np.max(w2)

accuracy = r2_score(y_test, ypredict, sample_weight=w2 / w2)
accuracy

plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# %%

# %%

columns_to_drop = ['population', 'sys_id', 'wu_rate']
df = df_train.copy()
del (df['pc_median'])
df['pc'] = df['wu_rate'] / df['pop']
df = df[df['pop'] > 1000]
mask = (df['pc'] > 20) & (df['pc'] < 500)
df = df[mask]

df = df.drop(columns_to_drop, axis=1)
features = list(df.columns)
features.remove('pc')
X = df[features]
y = df['pc']

# 
plt.figure()
v_freq = plt.hist(df['pc'].values, bins=30)
freq = np.interp(df['pc'].values, v_freq[1][1:], v_freq[0], left=None, right=None, period=None)
X['freq'] = freq
# squarederror
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
Train_freq = X_train['freq']
Test_freq = X_test['freq']
del (X_test['freq'])
del (X_train['freq'])

# squarederror
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 123)
gb = xgb.XGBRegressor(objective="reg:squarederror", tree_method='hist', colsample_bytree=0.8, learning_rate=0.20,
                      max_depth=7, alpha=100, n_estimators=500, rate_drop=0.9, skip_drop=0.5, subsample=0.8,
                      seed=123, reg_lambda=10, min_child_weight=1, gamma=10, max_delta_step=0,
                      )
# gb = xgb.XGBRegressor(objective="reg:squarederror" )


w = 1.0 / Train_freq
# w = w/np.sum(w)
w = w / np.max(w)
gb.fit(X_train, y_train, sample_weight=w / w)
plt.figure()
ypredict = gb.predict(X_test)

w2 = 1.0 / Test_freq
# w2 = w2/np.sum(w)
w2 = w2 / np.max(w2)

accuracy = r2_score(y_test, ypredict, sample_weight=w2 / w2)
accuracy

plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# %% [markdown]
# # Use Pipeline 
#

# %%


# (A.1) transformation of numeric features
numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# (A.2) transformation of categorical features
categorical_features = ["embarked", "sex", "pclass"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# (A.3)do all preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

#  (B.1) complete pipeline
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)

# %%
from sklearn import set_config

set_config(display="diagram")
clf

# %%
## Scaling

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pipeline = Pipeline([('robost_scaler', RobustScaler()),
                     ('model', gb)])
kwargs = {pipeline.steps[-1][0] + '__sample_weight': w / w}
pipeline.fit(X_train, y_train, **kwargs)

plt.figure()
ypredict = pipeline.predict(X_test)

w2 = 1.0 / Test_freq
# w2 = w2/np.sum(w)
w2 = w2 / np.max(w2)

accuracy = r2_score(y_test, ypredict, sample_weight=w2 / w2)
accuracy

plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

from sklearn.decomposition import PCA

pipeline = Pipeline([('uniform_dist', PowerTransformer()),
                     ('model', gb)])
kwargs = {pipeline.steps[-1][0] + '__sample_weight': w / w}
pipeline.fit(X_train, y_train, **kwargs)

plt.figure()
ypredict = pipeline.predict(X_test)

w2 = 1.0 / Test_freq
# w2 = w2/np.sum(w)
w2 = w2 / np.max(w2)

accuracy = r2_score(y_test, ypredict, sample_weight=w2 / w2)
accuracy

plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.decomposition import PCA

pipeline = Pipeline([('disc', KBinsDiscretizer(n_bins=1000, encode='ordinal', strategy='uniform')),
                     ('model', gb)])
kwargs = {pipeline.steps[-1][0] + '__sample_weight': w / w}
pipeline.fit(X_train, y_train, **kwargs)

plt.figure()
ypredict = pipeline.predict(X_test)

w2 = 1.0 / Test_freq
# w2 = w2/np.sum(w)
w2 = w2 / np.max(w2)

accuracy = r2_score(y_test, ypredict, sample_weight=w2 / w2)
accuracy

plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# %%
# plot feature importance
# ‘weight’: the number of times a feature is used to split the data across all trees.

# ‘gain’: the average gain across all splits the feature is used in.

# ‘cover’: the average coverage across all splits the feature is used in.

# ‘total_gain’: the total gain across all splits the feature is used in.

# ‘total_cover’: the total coverage across all splits the feature is used in.
plot_importance(gb, xlabel='total_gain', importance_type='total_gain', max_num_features=10)
from sklearn.feature_selection import SelectFromModel

plt.tight_layout()

# %%
# Another look at importance

# %%
plt.figure()
ypredict = gb.predict(X_test)

w2 = 1.0 / Test_freq
# w2 = w2/np.sum(w)
w2 = w2 / np.max(w2)

accuracy = r2_score(y_test, ypredict, sample_weight=w2 / w2)
accuracy

plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# %%
# Tuning
gb.get_params()
# params = {'colsample_bylevel': scipy.stats.uniform (loc = 0.1, scale=0.9),
#     'colsample_bynode': scipy.stats.uniform (loc = 0.1, scale=0.9),
#     'colsample_bytree': scipy.stats.uniform (loc = 0.1, scale=0.9),
#     'gamma': scipy.stats.uniform (loc = 0, scale=100),
#     'learning_rate': scipy.stats.uniform (loc = 0.001, scale=0.4),
#     'max_depth': [3,4,5,6,7,8,9, 10],
#     'min_child_weight' : scipy.stats.uniform (loc = 0 scale=1000),
#     'reg_alpha': 100,
#      'reg_lambda': 10,
#      'subsample': 0.8
#     }

# %%
from pycaret.regression import *

columns_to_drop = ['population', 'sys_id', 'wu_rate']
df = df_train.copy()
del (df['pc_median'])
df['pc'] = df['wu_rate'] / df['pop']
df = df[df['pop'] > 1000]
mask = (df['pc'] > 20) & (df['pc'] < 500)
df = df[mask]

df = df.drop(columns_to_drop, axis=1)
data = df.sample(frac=0.9, random_state=786)
data_unseen = df.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

exp0 = setup(data=data2, target='pc', train_size=0.8,
             fold_shuffle=True, data_split_shuffle=True, session_id=123)
xgb = create_model('xgboost', fold=3)
tuned_ada = tune_model(xgb)

# %%

# %%
tuned_ada = tune_model(xgb)

# %%
columns_to_drop = ['population', 'sys_id', 'wu_rate']
df = df_train.copy()

df['pc'] = df['wu_rate'] / df['pop']
df = df[df['pop'] > 100]
mask = (df['pc'] > 20) & (df['pc'] < 500)
df = df[mask]

df = df.drop(columns_to_drop, axis=1)
features = list(df.columns)
features.remove('pc')
X = df[features]
y = df['pc']

# %%
columns_to_drop = ['population', 'sys_id', 'wu_rate']
df = df_train.copy()

df['pc'] = df['wu_rate'] / df['pop']
df = df[df['pop'] > 100]
mask = (df['pc'] > 20) & (df['pc'] < 500)
df = df[mask]

df = df.drop(columns_to_drop, axis=1)
features = list(df.columns)
features.remove('pc')
X = df[features]
y = df['pc']

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=20, whiten=True)
pca.fit(X)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)
X = pca.transform(X)

# %%
# squarederror
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
gb = xgb.XGBRegressor(objective='reg:"squarederror"', colsample_bytree=0.8, learning_rate=0.1,
                      max_depth=7, alpha=0.01, n_estimators=500, rate_drop=0.9, skip_drop=0.5, subsample=0.8,
                      seed=123, reg_lambda=0.0)
gb.fit(X_train, y_train)

# %%
plt.figure()
ypredict = gb.predict(X_test)
accuracy = r2_score(y_test, ypredict)
accuracy
accuracy = int(accuracy * 100) / 100.0
plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# %% [markdown]
# ## Quantile transformation of the target PC
# ### Uniform Distribution

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform
from scipy import stats

x = pd.DataFrame(np.random.rand(100), columns=['x'])
stats.percentileofscore(x['x'], 0.47219)

# %%

# %%
from scipy import stats

# %%
# quantile transform
# columns to drop
columns_to_drop = ['population', 'sys_id', 'wu_rate']
df = df_train.copy()

df['pc'] = df['wu_rate'] / df['pop']
df = df[df['pop'] > 100]
mask = (df['pc'] > 20) & (df['pc'] < 500)
df = df[mask]
pc_max = df['pc'].max()
pc_min = df['pc'].min()

df = df.drop(columns_to_drop, axis=1)
wu = quantile_transform(df['pc'].values.reshape(len(df), 1), n_quantiles=5000,
                        random_state=0)  # , output_distribution = 'normal'
df['pc'] = wu.flatten()
features = list(df.columns)
features.remove('pc')
X = df[features]
y = df['pc']

# %%

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# %%
# squarederror
gb = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.8, learning_rate=0.1,
                      max_depth=7, alpha=0.01, n_estimators=500, rate_drop=0.9, skip_drop=0.5, subsample=0.8,
                      seed=123, reg_lambda=0.0)

# %%
gb.fit(X_train, y_train)

# %%
plt.figure()
ypredict = gb.predict(X_test)
accuracy = r2_score(y_test, ypredict)
accuracy
accuracy = int(accuracy * 100) / 100.0
plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual Water Use")
plt.ylabel("Estimated Water Use")
plt.grid()
plt.show()

# %% [markdown]
#

# %%
plt.figure()
ypredict = gb.predict(X_test)
ypredict = pc_min + ypredict * (pc_max - pc_min)
y_test = pc_min + y_test * (pc_max - pc_min)
accuracy = r2_score(y_test, ypredict)
accuracy
accuracy = int(accuracy * 100) / 100.0
plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# %% [markdown]
# ## Normal Distribution

# %%
# quantile transform
# columns to drop
columns_to_drop = ['population', 'sys_id', 'wu_rate']
df = df_train.copy()

df['pc'] = df['wu_rate'] / df['pop']
df = df[df['pop'] > 100]
mask = (df['pc'] > 20) & (df['pc'] < 500)
df = df[mask]
pc_mean = df['pc'].mean()
pc_std = df['pc'].std()

df = df.drop(columns_to_drop, axis=1)
wu = quantile_transform(df['pc'].values.reshape(len(df), 1), n_quantiles=5000,
                        random_state=0, output_distribution='normal')  # , output_distribution = 'normal'
df['pc'] = wu.flatten()
features = list(df.columns)
features.remove('pc')
X = df[features]
y = df['pc']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# %%
gb = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.8, learning_rate=0.1,
                      max_depth=7, alpha=0.01, n_estimators=500, rate_drop=0.9, skip_drop=0.5, subsample=0.8,
                      seed=123, reg_lambda=0.0)

# %%
gb.fit(X_train, y_train)

# %%
plt.figure()
ypredict = gb.predict(X_test)
accuracy = r2_score(y_test, ypredict)
accuracy
accuracy = int(accuracy * 100) / 100.0
plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# %%
pc_std

# %%
plt.figure()
ypredict = gb.predict(X_test)

ypredict = 180 + pc_mean + pc_std * ypredict
y_test = 180 + pc_mean + pc_std * y_test
accuracy = r2_score(y_test, ypredict)

accuracy = int(accuracy * 100) / 100.0
plt.scatter(y_test, ypredict, s=4)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')

plt.title("$R^2 = ${}".format(accuracy))
plt.xlabel("Actual PC Water Use")
plt.ylabel("Estimated PC Water Use")
plt.grid()
plt.show()

# %%

# %%

# %%

# %%
