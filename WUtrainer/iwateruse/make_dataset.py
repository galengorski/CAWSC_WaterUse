import os, sys
import pandas as pd
import numpy as np



def get_annual_wu(model, annual_wu):

    model.log.info("\n\n\n ======= Preparing Annual Water Use Data ==========")


    cols = ['fswud', 'fnonswud', 'fswuds_pc', 'fnonswuds_pc']
    cleanned_wu = annual_wu.copy()
    mask = cleanned_wu['pop'].isna() | cleanned_wu['pop']==0
    cleanned_wu.loc[mask, 'pop'] = 1.0
    cleanned_wu['fswud'] = cleanned_wu['annual_wu_G_swuds'].abs()
    cleanned_wu['fnonswud'] = cleanned_wu['annual_wu_G_nonswuds'].abs()
    cleanned_wu['nonswuds_pc'] = cleanned_wu['nonswuds_pc'].abs()
    cleanned_wu['fswuds_pc'] = cleanned_wu['fswud'] / (cleanned_wu['days_in_year']* cleanned_wu['pop'])
    cleanned_wu['fnonswuds_pc'] = cleanned_wu['fnonswud'] / (cleanned_wu['days_in_year']* cleanned_wu['pop'])
    #model.log.to_table(cleanned_wu[cols].describe(percentiles=[0.05,0.5,0.95]), header = -1)


    cleanned_wu.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in cols:
        cleanned_wu.loc[cleanned_wu[c] == 0, c] = np.NAN


    model.log.to_table(cleanned_wu[cols].describe(percentiles=[0.05,0.5,0.95]), header = -1)

    # mask system not in WSA
    model.log.info("\n\n *** Drop systems outside WSA ...")
    for c in cols:
        cleanned_wu.loc[cleanned_wu['inWSA'] == 0, c] = np.NAN
    cleanned_wu['fswuds_pc'] = cleanned_wu['fswud'] / (cleanned_wu['days_in_year'] * cleanned_wu['pop'])
    cleanned_wu['fnonswuds_pc'] = cleanned_wu['fnonswud'] / (cleanned_wu['days_in_year'] * cleanned_wu['pop'])
    model.log.to_table(cleanned_wu[cols].describe(percentiles=[0.05,0.5,0.95]), header = -1)

    mask = cleanned_wu['flg_no_annual'] == 1
    model.log.info("\n\n *** Use monthly data to fill missing annual data")
    cleanned_wu.loc[mask, 'fswud'] = cleanned_wu.loc[mask, 'fnonswud']
    cleanned_wu['fswuds_pc'] = cleanned_wu['fswud'] / (cleanned_wu['days_in_year']* cleanned_wu['pop'])
    cleanned_wu['fnonswuds_pc'] = cleanned_wu['fnonswud'] / (cleanned_wu['days_in_year']* cleanned_wu['pop'])
    model.log.to_table(cleanned_wu[cols].describe(percentiles=[0.05,0.5,0.95]), header = -1)

    model.log.info("\n\n *** Resolve difference between SWUDS monthly and annual water use ")
    mask = cleanned_wu['flg_annual_isdiff_annualize_month_swud'] == 2
    monthly_pc = cleanned_wu['annual_wu_from_monthly_swuds_G']/(365 * cleanned_wu['pop'])
    annual_pc = cleanned_wu['fswud'] / (365 * cleanned_wu['pop'])
    annual_issues_mask = (annual_pc<=25) | (annual_pc>=300)
    monthly_ok_maks = (monthly_pc>=25) & (monthly_pc < 300)
    mask = mask & annual_issues_mask & monthly_ok_maks
    cleanned_wu.loc[mask, 'fswud'] = cleanned_wu.loc[mask, 'annual_wu_from_monthly_swuds_G']
    cleanned_wu.loc[mask, 'swuds_pc'] =  cleanned_wu.loc[mask, 'fswud']/(365*cleanned_wu.loc[mask, 'pop'])
    cleanned_wu['fswuds_pc'] = cleanned_wu['fswud'] / (cleanned_wu['days_in_year']* cleanned_wu['pop'])
    cleanned_wu['fnonswuds_pc'] = cleanned_wu['fnonswud'] / (cleanned_wu['days_in_year']* cleanned_wu['pop'])
    model.log.to_table(cleanned_wu[cols].describe(percentiles=[0.05, 0.5, 0.95]), header=-1)

    model.log.info("\n\n *** Resolve issues with large temporal change  ")
    model.log.info("Drop systems with temporal change greater than 50% and have extreme per capita --out side (20,600)")
    mask = cleanned_wu['prc_time_change_swud'] > 50
    maskpc = (cleanned_wu['fswuds_pc']<20) |  (cleanned_wu['fswuds_pc']>600)
    cleanned_wu.loc[(mask & maskpc), ['fswud']] = np.NAN

    mask = cleanned_wu['prc_time_change_nonswud'] > 50
    maskpc = (cleanned_wu['fnonswuds_pc'] < 20) | (cleanned_wu['fnonswuds_pc'] > 600)
    cleanned_wu.loc[(mask & maskpc), ['fnonswud', 'fnonswuds_pc']] = np.NAN
    cleanned_wu['fswuds_pc'] = cleanned_wu['fswud'] / (cleanned_wu['days_in_year'] * cleanned_wu['pop'])
    cleanned_wu['fnonswuds_pc'] = cleanned_wu['fnonswud'] / (cleanned_wu['days_in_year'] * cleanned_wu['pop'])
    model.log.to_table(cleanned_wu[cols].describe(percentiles=[0.05,0.5,0.95]), header = -1)

    # correct population
    cleanned_wu['pop_enh'] = cleanned_wu['pop'].copy()
    swud16_pc = cleanned_wu['fswud']/ (cleanned_wu['days_in_year'] * cleanned_wu['pop_swud16'])
    tpopsrv_pc = cleanned_wu['fswud'] / (cleanned_wu['days_in_year'] * cleanned_wu['TPOPSRV'])

    model.log.info("\n\n *** Correct Population using SWUD16 population ... ")
    badswud =  (cleanned_wu['fswuds_pc'] <= 20) | (cleanned_wu['fswuds_pc'] > 500) |(cleanned_wu['fnonswuds_pc'] <= 20) | (cleanned_wu['fnonswuds_pc'] > 500)
    goodswud16 = (swud16_pc > 20) & (swud16_pc <= 500)
    cleanned_wu.loc[badswud & goodswud16, 'pop_enh'] = cleanned_wu.loc[badswud & goodswud16, 'pop_swud16']

    cleanned_wu['fswuds_pc'] = cleanned_wu['fswud'] / (cleanned_wu['days_in_year'] * cleanned_wu['pop_enh'])
    cleanned_wu['fnonswuds_pc'] = cleanned_wu['fnonswud'] / (cleanned_wu['days_in_year'] * cleanned_wu['pop_enh'])
    model.log.to_table(cleanned_wu[cols].describe(percentiles=[0.05, 0.5, 0.95]), header=-1)

    model.log.info("\n\n *** Correct Population using TPOPSRV population ... ")
    badswud = (cleanned_wu['fswuds_pc'] <= 20) | (cleanned_wu['fswuds_pc'] > 500) | (
                cleanned_wu['fnonswuds_pc'] <= 20) | (cleanned_wu['fnonswuds_pc'] > 500)
    goodtpop16 = (tpopsrv_pc > 20) & (tpopsrv_pc <= 500)
    cleanned_wu.loc[badswud & goodtpop16, 'pop_enh'] = cleanned_wu.loc[badswud & goodtpop16, 'TPOPSRV']

    cleanned_wu['fswuds_pc'] = cleanned_wu['fswud'] / (cleanned_wu['days_in_year'] * cleanned_wu['pop_enh'])
    cleanned_wu['fnonswuds_pc'] = cleanned_wu['fnonswud'] / (cleanned_wu['days_in_year'] * cleanned_wu['pop_enh'])
    model.log.to_table(cleanned_wu[cols].describe(percentiles=[0.05, 0.5, 0.95]), header=-1)

    model.log.info("\n\n *** Compare SWUD and NonSWUD and use the best ... ")
    cleanned_wu['final_wu'] = cleanned_wu['fswud'].copy()
    badswud = (cleanned_wu['fswuds_pc'] <= 20) | (cleanned_wu['fswuds_pc'] > 500) | cleanned_wu['fswuds_pc'].isna()
    gooNonSWUD = (cleanned_wu['fnonswuds_pc'] > 20) & (cleanned_wu['fnonswuds_pc'] <= 500)
    cleanned_wu.loc[badswud & gooNonSWUD, 'final_wu'] = cleanned_wu.loc[badswud & gooNonSWUD, 'fnonswud']
    cleanned_wu['final_wu_pc'] = cleanned_wu['final_wu'] / (cleanned_wu['days_in_year'] * cleanned_wu['pop_enh'])
    model.log.to_table(cleanned_wu[['final_wu', 'final_wu_pc']].describe(percentiles=[0.05, 0.5, 0.95]), header=-1)


    return cleanned_wu

def make_ds_per_capita_basic(model, datafile = None):
    # ======= Load =======
    df_train_bk = pd.read_csv(datafile)
    #pc_50_swud = pd.read_csv(r"spatial_pc_statistics_apply_max_min_pc.csv")
    annual_wu = pd.read_csv(r"C:\work\water_use\CAWSC_WaterUse\etc\annual_wu.csv")
    cleanned_wu = get_annual_wu(model, annual_wu)
    df_train = df_train_bk.copy()

    # ====== Add water use =======
    annual_wu['wu_rate_mean'] = annual_wu[['annual_wu_G_swuds', 'annual_wu_G_nonswuds']].mean(axis=1)
    annual_wu['wu_rate_mean'] = annual_wu['wu_rate_mean'] / annual_wu['days_in_year']
    avg_wu = annual_wu[['WSA_AGIDF', 'YEAR', 'wu_rate_mean']].copy()
    avg_wu.rename(columns={'WSA_AGIDF': 'sys_id', 'YEAR': 'Year', 'wu_rate_mean': 'wu_rate'}, inplace=True)
    del (df_train['wu_rate'])
    df_train = df_train.merge(avg_wu, on=['sys_id', 'Year'], how='left')

    model.add_training_df( df_train = df_train)


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

