import os, sys
import pandas as pd
import numpy as np


def make_ds_per_capita_basic(model):
    # ======= Load =======
    model.df_train_bk = pd.read_csv(r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv")
    pc_50_swud = pd.read_csv(r"spatial_pc_statistics_apply_max_min_pc.csv")
    annual_wu = pd.read_csv(r"annual_wu.csv")
    model.df_train = model.df_train_bk.copy()

    # ====== Add water use =======
    annual_wu['wu_rate_mean'] = annual_wu[['annual_wu_G_swuds', 'annual_wu_G_nonswuds']].mean(axis=1)
    annual_wu['wu_rate_mean'] = annual_wu['wu_rate_mean'] / annual_wu['days_in_year']
    avg_wu = annual_wu[['WSA_AGIDF', 'YEAR', 'wu_rate_mean']].copy()
    avg_wu.rename(columns={'WSA_AGIDF': 'sys_id', 'YEAR': 'Year', 'wu_rate_mean': 'wu_rate'}, inplace=True)
    del (model.df_train['wu_rate'])
    model.df_train = model.df_train.merge(avg_wu, on=['sys_id', 'Year'], how='left')
    model.df_train_bk = model.df_train.copy()


def make_ds_per_capita_system_average(model):
    # ======= Load =======
    features_info = pd.read_excel(r"C:\work\water_use\ml_experiments\annual_v_0_0\features_status.xlsx", sheet_name='annual')
    model.df_train_bk = pd.read_csv(r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv")
    pc_50_swud = pd.read_csv(r"spatial_pc_statistics_apply_max_min_pc.csv")
    annual_wu = pd.read_csv(r"annual_wu.csv")

    #custom fixes
    model.df_train_bk['rur_urb_cnty'] = model.df_train_bk['rur_urb_cnty'].astype(int)

    # Extract data
    model.categorical_features = features_info[features_info['Type'].isin(['categorical'])]['Feature_name'].values.tolist()
    model.df_train = model.df_train_bk.copy()

    # ====== Add water use =======
    annual_wu['wu_rate_mean'] = annual_wu[['annual_wu_G_swuds', 'annual_wu_G_nonswuds']].mean(axis=1)
    annual_wu['wu_rate_mean'] = annual_wu['wu_rate_mean'] / annual_wu['days_in_year']
    avg_wu = annual_wu[['WSA_AGIDF', 'YEAR', 'wu_rate_mean']].copy()
    avg_wu.rename(columns={'WSA_AGIDF': 'sys_id', 'YEAR': 'Year', 'wu_rate_mean': 'wu_rate'}, inplace=True)
    del (model.df_train['wu_rate'])
    model.df_train = model.df_train.merge(avg_wu, on=['sys_id', 'Year'], how='left')

    #
    temp_catfeat = []
    for cat_feat in model.categorical_features:
        if cat_feat in model.df_train.columns:
            temp_catfeat.append(cat_feat)
    model.categorical_features = temp_catfeat

    #
    cat_df = model.df_train[model.categorical_features]
    cat_df = cat_df.groupby('sys_id').agg(pd.Series.mode)

    numeric_feat = set(model.df_train.columns).difference(set(model.categorical_features))
    numeric_feat = ['sys_id'] + list(numeric_feat)
    numeric_df = model.df_train[numeric_feat]
    numeric_df = numeric_df.groupby('sys_id').median()

    df = pd.concat([numeric_df, cat_df], axis=1)
    df.reset_index(inplace=True)
    model.df_train = df
    model.df_train_bk = model.df_train.copy()

def make_ds_per_capita_system_average_neutral(model):
    # ======= Load =======
    features_info = pd.read_excel(r"C:\work\water_use\ml_experiments\annual_v_0_0\features_status.xlsx", sheet_name='annual')
    model.df_train_bk = pd.read_csv(r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv")
    pc_50_swud = pd.read_csv(r"spatial_pc_statistics_apply_max_min_pc.csv")
    annual_wu = pd.read_csv(r"annual_wu.csv")

    #custom fixes
    model.df_train_bk = model.df_train_bk[model.df_train_bk['Ecode'].isin(['N'])]
    model.df_train_bk['rur_urb_cnty'] = model.df_train_bk['rur_urb_cnty'].astype(int)
    del(model.df_train_bk['Ecode'])

    # Extract data
    model.categorical_features = features_info[features_info['Type'].isin(['categorical'])]['Feature_name'].values.tolist()
    model.df_train = model.df_train_bk.copy()

    # ====== Add water use =======
    annual_wu['wu_rate_mean'] = annual_wu[['annual_wu_G_swuds', 'annual_wu_G_nonswuds']].mean(axis=1)
    annual_wu['wu_rate_mean'] = annual_wu['wu_rate_mean'] / annual_wu['days_in_year']
    avg_wu = annual_wu[['WSA_AGIDF', 'YEAR', 'wu_rate_mean']].copy()
    avg_wu.rename(columns={'WSA_AGIDF': 'sys_id', 'YEAR': 'Year', 'wu_rate_mean': 'wu_rate'}, inplace=True)
    del (model.df_train['wu_rate'])
    model.df_train = model.df_train.merge(avg_wu, on=['sys_id', 'Year'], how='left')

    #
    temp_catfeat = []
    for cat_feat in model.categorical_features:
        if cat_feat in model.df_train.columns:
            temp_catfeat.append(cat_feat)
    model.categorical_features = temp_catfeat

    #
    cat_df = model.df_train[model.categorical_features]
    cat_df = cat_df.groupby('sys_id').agg(pd.Series.mode)

    numeric_feat = set(model.df_train.columns).difference(set(model.categorical_features))
    numeric_feat = ['sys_id'] + list(numeric_feat)
    numeric_df = model.df_train[numeric_feat]
    numeric_df = numeric_df.groupby('sys_id').median()

    df = pd.concat([numeric_df, cat_df], axis=1)
    df.reset_index(inplace=True)
    model.df_train = df
    model.df_train_bk = model.df_train.copy()

