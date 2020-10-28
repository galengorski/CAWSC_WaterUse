import os, sys
import dask.dataframe as pdd
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy import interpolate

def get_features(huc2s = [], database_root = ""):

    cs_features = ['population', 'households2', 'median_income', 'pop_density']
    cs_features = cs_features + ['h_age_newer_2005', 'h_age_2000_2004', 'h_age_1990_1999', 'h_age_1980_1989',
                                 'h_age_1970_1979', 'h_age_1960_1969', 'h_age_1950_1959',
                                 'h_age_1940_1949', 'h_age_older_1939']
    if len (huc2s) == 0:
        huc2s = range(1,23)
    huc2_folders = os.listdir(database_root)
    for huc2 in huc2s:
        # loop over all huc2s
        if not(str(huc2) in huc2_folders):
            print("There is no HUC2 = {} in {} folder".format(huc2, database_root))
            continue

        # extract census
        census_folder = os.path.join(database_root,
                                     os.path.join(str(huc2), "census"))
        census_df = aggregate_census_data(census_folder, census_prefix="cs_wsa_",
                                          cs_features=cs_features, skip_wkey="_yearly")
        census_df.to_csv(os.path.join(census_folder, "daily_census_{}.csv".format(huc2)))

        # extract climate
        climate_folder = os.path.join(database_root,
                                     os.path.join(str(huc2), "climate"))
        climate_df = aggregate_climate_date(climate_folder=climate_folder)
        climate_df.to_csv(os.path.join(climate_folder, "daily_climate_{}.csv".format(huc2)))
        xxx = 1
        # extract wu
        # combine

    # reduce to annual
    # reduce to monthly

    pass

def get_all_annual_db(database_root, wu_file, wsa_file):

    cs_features = ['population', 'households2', 'median_income', 'pop_density']
    cs_features = cs_features + ['h_age_newer_2005', 'h_age_2000_2004', 'h_age_1990_1999', 'h_age_1980_1989',
                                 'h_age_1970_1979', 'h_age_1960_1969', 'h_age_1950_1959',
                                 'h_age_1940_1949', 'h_age_older_1939']

    huc2s = range(1,23)
    huc2_folders = os.listdir(database_root)
    for huc2 in huc2s:
        # loop over all huc2s
        if not(str(huc2) in huc2_folders):
            print("There is no HUC2 = {} in {} folder".format(huc2, database_root))
            continue

        # extract census
        census_folder = os.path.join(database_root,
                                     os.path.join(str(huc2), "census"))
        # extract climate
        climate_folder = os.path.join(database_root,
                                     os.path.join(str(huc2), "climate"))

        df_annual =assemble_annual_training_dataset(wu_file=wu_file, wsa_file=wsa_file, year_field='YEAR', wu_field='TOT_WD_MGD'
                                         , sys_id_field='WSA_AGIDF',
                                         shp_sys_id_field='WSA_AGIDF', output_file='',
                                         census_folder=census_folder, climate_folder=climate_folder,
                                         func_to_process_sys_name=None, to_galon=1e6,
                                         census_file_prefex="cs_wsa_", climate_file_prefix="",
                                         cs_features=cs_features)
        df_annual['HUC2'] = huc2
        # check if folder assemble exist
        feature_folder = os.path.join(database_root,
                                     os.path.join(str(huc2)))
        if not("assemble" in os.listdir(feature_folder)):
            os.mkdir(os.path.join(feature_folder, "assemble"))
        assemble_folder = os.path.join(feature_folder, "assemble")
        df_annual.to_csv(os.path.join(assemble_folder, "train_db_{}.csv".format(huc2)))

    # now loop over assemble folders to assemble the final file

def aggregate_climate_date(climate_folder):
    li = []
    climate_vars = ["etr", "pr", "tmmn", "tmmx"]
    prefix = [var + "_" for var in climate_vars]

    sys_ids = get_water_system_ids_from_folder(climate_folder, prefix)
    for sys_id in sys_ids:
        print(sys_id)
        for ivar, var in enumerate(climate_vars):
            clim_fn = "{}_{}.csv".format(var, sys_id)
            clim_fn = os.path.join(climate_folder, clim_fn)
            if os.path.isfile(clim_fn):
                cc_df = pd.read_csv(clim_fn, parse_dates=['day'])

                if ivar == 0:
                    curr_cm = pd.DataFrame(columns=['Date', var])
                    curr_cm['Date'] = cc_df['day']
                    curr_cm[var] = cc_df[sys_id]
                    curr_cm.set_index(keys=['Date'], inplace=True)
                    continue

                cc_df.set_index(keys=['day'], inplace=True)
                curr_cm[var] = cc_df[sys_id]
        li.append(curr_cm.copy())
    li = pd.concat(li)
    return li


def aggregate_census_data(census_folder, census_prefix, cs_features, skip_wkey=''):
    li = []
    files = os.listdir(census_folder)
    for file in files:
        print(file)

        if skip_wkey in file:
            continue

        if census_prefix in file:
            sys_id = os.path.splitext(file)[0]
            sys_id = sys_id.replace(census_prefix, "")
            df_ = pd.read_csv(os.path.join(census_folder, file))
            df_['date'] = df_['dyear'].apply(frac_to_date)
            df_['sys_id'] = sys_id.lower()
            li.append(df_)
    li = pd.concat(li, ignore_index=True)

    return li


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def extrapolate_census_data(df, fields_to_interpolate, index_type='year'):
    x = df.index.values
    if index_type == 'year':
        pass
    elif index_type == 'year_month':
        x_1 = []
        for x_ in x:
            x_1.append(x_[0] + (x_[1] - 1) / 12)
        x = np.array(x_1)

    for var in fields_to_interpolate:
        y = df[var].values
        maskNaN = np.isnan(y)
        mask = np.logical_not(maskNaN)
        x_ = x[mask]
        y_ = y[mask]
        mean_value = np.mean(y_)
        x_ = moving_average(x_, n=3)
        y_ = moving_average(y_, n=3)

        f = interpolate.interp1d(x_, y_, fill_value='extrapolate', kind='linear')
        xnew = x[maskNaN]
        ynew = f(xnew)
        ynew[ynew <= 0] = mean_value

        y[maskNaN] = ynew
        df[var] = y

    return df


def frac_to_date(frac_date):
    yyear = int(frac_date)
    days = frac_date - yyear
    if yyear % 4 == 0:
        days = round(366 * days)
    else:
        days = round(365 * days)
    ddate = pd.to_datetime('{}-1-1'.format(yyear)) + pd.to_timedelta(days, unit='D')
    return ddate


def get_water_system_ids_from_folder(folder, prefix='cs_'):
    if not (type(prefix) is list):
        prefix = [prefix]

    files = os.listdir(folder)
    ws_list = []
    for file in files:
        fn = os.path.splitext(file)[0]
        for pre in prefix:
            fn = fn.replace(pre, "")
        ws_list.append(fn.upper())
    ws_list = set(ws_list)  # remove duplicates
    return list(ws_list)


def assemble_annual_training_dataset(wu_file, wsa_file='', year_field='year', wu_field='wu', sys_id_field='x',
                                     shp_sys_id_field='x', output_file='', census_folder='.', climate_folder='.',
                                     func_to_process_sys_name=None, to_galon=1.0,
                                     census_file_prefex="", climate_file_prefix="",
                                     cs_features=[], maxYear=2020, minYear=2000):
    wsa_shp = gpd.read_file(wsa_file)
    wu_df = pd.read_csv(wu_file)[[year_field, sys_id_field, wu_field]]

    # extract intersection of wsa shapefile and wu_df
    sys_ids = set(wsa_shp[shp_sys_id_field]).intersection(set(wu_df[sys_id_field]))
    wu_df = wu_df[wu_df[sys_id_field].isin(sys_ids)]

    wu_df[wu_field] = wu_df[wu_field] * to_galon
    sys_in_census = get_water_system_ids_from_folder(census_folder, prefix='cs_wsa_')
    # wu_df = wu_df.set_index(keys=[date_field])
    index = 0
    all_wu = []
    for sys_id in sys_in_census:

        if not (sys_id in sys_ids):
            continue

        index = index + 1
        # get current wu
        sa_mask = wu_df[sys_id_field] == sys_id
        curr_wu = wu_df[sa_mask]
        curr_wu = curr_wu.groupby(by=year_field).sum()
        wu_ = pd.DataFrame(columns=curr_wu.columns, index=np.arange(minYear, maxYear + 1))
        for cc in curr_wu.columns:
            wu_[cc] = curr_wu[cc]

        if len(curr_wu) == 0:
            continue

        print(sys_id)

        if not (func_to_process_sys_name is None):
            sys_id = func_to_process_sys_name(sys_id)

        # census
        fn = "{}{}.csv".format(census_file_prefex, sys_id)
        fn_cs = os.path.join(census_folder, fn)
        if os.path.isfile(fn_cs):
            cs_features_ = ['dyear'] + cs_features
            cs_df = pd.read_csv(fn_cs)[cs_features_]
            cs_df['date'] = cs_df['dyear'].apply(frac_to_date)
            cs_df['sys_id'] = sys_id.lower()
            cs_df['year'] = cs_df['date'].dt.year
            cs_df = cs_df.groupby(by=['year']).mean()
            cs_df = extrapolate_census_data(cs_df,
                                            fields_to_interpolate=cs_features)
            for cs_var in cs_features:
                wu_[cs_var] = cs_df[cs_var]

        # climate
        for var in ["etr", "pr", "tmmn", "tmmx"]:
            clim_fn = "{}_{}.csv".format(var, sys_id)
            clim_fn = os.path.join(climate_folder, clim_fn)
            if os.path.isfile(clim_fn):
                cc_df = pd.read_csv(clim_fn, parse_dates=['day'])
                field_names = list(cc_df.columns)
                field_names.remove('day')
                cc_df['sys_id'] = sys_id
                cc_df['year'] = cc_df['day'].dt.year

                cc_df = cc_df.groupby(by=['year']).mean()

                wu_[var] = cc_df
        # wu_.reset_index(inplace=True)
        wu_['sys_id'] = sys_id
        all_wu.append(wu_.copy())
    all_wu = pd.concat(all_wu)
    all_wu['Year'] = all_wu.index.values
    all_wu['wu_rate'] = all_wu[wu_field]
    all_wu.reset_index(inplace=True)

    del (all_wu['index'])
    del (all_wu[wu_field])
    return all_wu


def assemble_monthly_training_dataset(wu_file, wsa_file='', year_field='year', wu_field='wu', sys_id_field='x',
                                      shp_sys_id_field='x', output_file='', census_folder='.', climate_folder='.',
                                      func_to_process_sys_name=None, to_galon=1.0,
                                      census_file_prefex="", climate_file_prefix="",
                                      cs_features=[], maxYear=2020, minYear=2000):
    mon = {1: 'JAN_MGD', 2: 'FEB_MGD', 3: 'MAR_MGD', 4: 'APR_MGD', 5: 'MAY_MGD', 6: 'JUN_MGD',
           7: 'JUL_MGD', 8: 'AUG_MGD', 9: 'SEP_MGD', 10: 'OCT_MGD', 11: 'NOV_MGD', 12: 'DEC_MGD'}
    cols_to_load = [year_field, sys_id_field] + list(mon.values())
    wsa_shp = gpd.read_file(wsa_file)
    wu_df = pd.read_csv(wu_file)[cols_to_load]

    for key in mon.keys():
        wu_df[key] = wu_df[mon[key]]
        del (wu_df[mon[key]])
    Qann = wu_df[range(1, 13)].sum(axis=1)
    wu_df = wu_df[Qann > 0]
    # extract intersection of wsa shapefile and wu_df
    sys_ids = set(wsa_shp[shp_sys_id_field]).intersection(set(wu_df[sys_id_field]))
    wu_df = wu_df[wu_df[sys_id_field].isin(sys_ids)]

    for mi in range(1, 13):
        wu_df[mi] = wu_df[mi] * to_galon

    wu_df = wu_df.groupby(['YEAR', 'WSA_AGIDF']).sum()
    wu_df = wu_df.stack()
    wu_df = pd.DataFrame(wu_df, columns=['wu_rate'])
    wu_df.reset_index(inplace=True)
    wu_df['month'] = wu_df['level_2']
    del (wu_df['level_2'])

    sys_in_census = get_water_system_ids_from_folder(census_folder, prefix='cs_wsa_')
    # wu_df = wu_df.set_index(keys=[date_field])
    index = 0
    all_wu = []
    for sys_id in sys_in_census:

        if not (sys_id in sys_ids):
            continue

        index = index + 1
        # get current wu
        sa_mask = wu_df[sys_id_field] == sys_id
        curr_wu = wu_df[sa_mask]
        curr_wu = curr_wu.groupby(by=[year_field, 'month']).sum()

        if len(curr_wu) == 0:
            continue

        print(sys_id)

        if not (func_to_process_sys_name is None):
            sys_id = func_to_process_sys_name(sys_id)

        # census
        fn = "{}{}.csv".format(census_file_prefex, sys_id)
        fn_cs = os.path.join(census_folder, fn)
        if os.path.isfile(fn_cs):
            cs_features_ = ['dyear'] + cs_features
            cs_df = pd.read_csv(fn_cs)[cs_features_]
            cs_df['date'] = cs_df['dyear'].apply(frac_to_date)
            cs_df['sys_id'] = sys_id.lower()
            cs_df['year'] = cs_df['date'].dt.year
            cs_df['month'] = cs_df['date'].dt.month
            cs_df = cs_df.groupby(by=['year', 'month']).mean()
            cs_df = extrapolate_census_data(cs_df,
                                            fields_to_interpolate=cs_features, index_type='year_month')

            cs_df['wu_rate'] = curr_wu['wu_rate']
            curr_wu = cs_df
        # climate
        for var in ["etr", "pr", "tmmn", "tmmx"]:
            clim_fn = "{}_{}.csv".format(var, sys_id)
            clim_fn = os.path.join(climate_folder, clim_fn)
            if os.path.isfile(clim_fn):
                cc_df = pd.read_csv(clim_fn, parse_dates=['day'])
                field_names = list(cc_df.columns)
                field_names.remove('day')
                # cc_df['sys_id'] = sys_id
                cc_df['year'] = cc_df['day'].dt.year
                cc_df['month'] = cc_df['day'].dt.month
                cc_df = cc_df.groupby(by=['year', 'month']).mean()

                curr_wu[var] = cc_df
        # wu_.reset_index(inplace=True)
        curr_wu['sys_id'] = sys_id
        all_wu.append(curr_wu.copy())
    all_wu = pd.concat(all_wu)
    all_wu.reset_index(inplace=True)
    return all_wu


# only daily dataset
def assemble_training_dataset(wu_file, date_field='date', wu_field='wu', sys_id_field='x',
                              output_file='master_file.csv', census_climate_folder='.',
                              func_to_process_sys_name=None, to_galon=1.0):
    wu_df = pd.read_csv(wu_file, parse_dates=[date_field])
    folder = census_climate_folder
    sys_ids = wu_df[sys_id_field].unique()

    wu_df['population'] = np.NAN
    wu_df['median_income'] = np.NAN
    wu_df['pop_density'] = np.NAN

    wu_df['pr'] = np.NAN
    wu_df['pet'] = np.NAN
    wu_df['tmmn'] = np.NAN
    wu_df['tmmx'] = np.NAN

    wu_df[wu_field] = wu_df[wu_field] * to_galon

    wu_df = wu_df.set_index(keys=[date_field])
    index = 0
    for sys_id in sys_ids:
        index = index + 1
        # get current wu
        sa_mask = wu_df[sys_id_field] == sys_id
        curr_wu = wu_df[sa_mask]
        print(sys_id)

        if not (func_to_process_sys_name is None):
            sys_id = func_to_process_sys_name(sys_id)

        # census
        fn = "{}.csv".format(sys_id)
        fn_cs = os.path.join(folder, fn)
        if os.path.isfile(fn_cs):
            cs_df = pd.read_csv(fn_cs)[['dyear', 'population', 'median_income', 'pop_density']]
            cs_df['date'] = cs_df['dyear'].apply(frac_to_date)
            cs_df['sys_id'] = sys_id.lower()
            cs_df = cs_df.set_index(keys=['date'])
            for cs_var in ['population', 'median_income', 'pop_density']:
                wu_df.loc[sa_mask, cs_var] = cs_df[cs_var]

        # climate
        for var in ["pet", "pr", "tmmn", "tmmx"]:
            clim_fn = "{}_{}.csv".format(var, sys_id)
            clim_fn = os.path.join(folder, clim_fn)
            if os.path.isfile(clim_fn):
                cc_df = pd.read_csv(clim_fn, parse_dates=['day'])
                field_names = list(cc_df.columns)
                field_names.remove('day')
                cc_df['sys_id'] = sys_id
                cc_df = cc_df.set_index(keys=['day'])
                wu_df.loc[sa_mask, var] = cc_df[field_names[0]]

    wu_df.reset_index(inplace=True)
    columns = {date_field: "Date", wu_field: "wu_rate", sys_id_field: "sys_id"}
    for field in columns.keys():
        wu_df[columns[field]] = wu_df[field]

    subcolumns = ["sys_id", "Date", 'wu_rate', 'population', 'median_income', 'pop_density', 'pr',
                  'pet', 'tmmn', 'tmmx']
    wu_df = wu_df[subcolumns]
    wu_df.to_csv(output_file)


if __name__ == '__main__':
    test_daily = False
    test_monthly = False
    test_annually = False

    if False:
        db_root = r"C:\work\water_use\mldataset\ml\training\features"
        get_features(huc2s=[6], database_root=db_root)

    if test_daily:
        wu_file = r"C:\work\water_use\dataset\dailywu\wi\wi_raw_data.csv"
        output_file = r"C:\work\water_use\dataset\dailywu\wi\wi_master.csv"
        assemble_training_dataset(wu_file, date_field='PUMP_DATE', wu_field='PUMP_Kgal', sys_id_field='PWS_ID',
                                  output_file=output_file, census_climate_folder=os.path.dirname(wu_file),
                                  to_galon=1000)

    test_assemble_all_annual = True
    if test_assemble_all_annual:
        wu_file = r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\SWUDS v13.csv"
        wsa_file = r"C:\work\water_use\mldataset\gis\wsa\WSA_v2_1_alb83_attrib.shp"
        database_root = r"C:\work\water_use\mldataset\ml\training\features"
        get_all_annual_db(database_root, wu_file, wsa_file)
        pass

    if test_annually:
        wu_file = r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\SWUDS v13.csv"
        wsa_file = r"C:\work\water_use\mldataset\gis\wsa\WSA_v2_1_alb83_attrib.shp"
        census_folder = r"C:\work\water_use\mldataset\ml\training\features\6\census"
        climate_folder = r"C:\work\water_use\mldataset\ml\training\features\6\climate"
        cs_features = ['population', 'households2', 'median_income', 'pop_density']
        cs_features = cs_features + ['h_age_newer_2005', 'h_age_2000_2004', 'h_age_1990_1999', 'h_age_1980_1989',
                                     'h_age_1970_1979', 'h_age_1960_1969', 'h_age_1950_1959',
                                     'h_age_1940_1949', 'h_age_older_1939']
        assemble_annual_training_dataset(wu_file=wu_file, wsa_file=wsa_file, year_field='YEAR', wu_field='TOT_WD_MGD'
                                         , sys_id_field='WSA_AGIDF',
                                         shp_sys_id_field='WSA_AGIDF', output_file='',
                                         census_folder=census_folder, climate_folder=climate_folder,
                                         func_to_process_sys_name=None, to_galon=1e6,
                                         census_file_prefex="cs_wsa_", climate_file_prefix="",
                                         cs_features=cs_features)
    if False: # test aggregate_climate
        climate_folder = r"C:\work\water_use\mldataset\ml\training\features\6\climate"
        aggregate_climate_date(climate_folder)

    if False:
        wu_file = r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\SWUDS v13.csv"
        wsa_file = r"C:\work\water_use\mldataset\gis\wsa\WSA_v2_1_alb83_attrib.shp"
        census_folder = r"C:\work\water_use\mldataset\ml\training\features\6\census"
        climate_folder = r"C:\work\water_use\mldataset\ml\training\features\6\climate"
        cs_features = ['population', 'households2', 'median_income', 'pop_density']
        cs_features = cs_features + ['h_age_newer_2005', 'h_age_2000_2004', 'h_age_1990_1999', 'h_age_1980_1989',
                                     'h_age_1970_1979', 'h_age_1960_1969', 'h_age_1950_1959',
                                     'h_age_1940_1949', 'h_age_older_1939']
        assemble_monthly_training_dataset(wu_file=wu_file, wsa_file=wsa_file, year_field='YEAR', wu_field='TOT_WD_MGD'
                                          , sys_id_field='WSA_AGIDF',
                                          shp_sys_id_field='WSA_AGIDF', output_file='',
                                          census_folder=census_folder, climate_folder=climate_folder,
                                          func_to_process_sys_name=None, to_galon=1e6,
                                          census_file_prefex="cs_wsa_", climate_file_prefix="",
                                          cs_features=cs_features)
    if False:
        cs_features = ['population', 'households2', 'median_income', 'pop_density']
        cs_features = cs_features + ['h_age_newer_2005', 'h_age_2000_2004', 'h_age_1990_1999', 'h_age_1980_1989',
                                     'h_age_1970_1979', 'h_age_1960_1969', 'h_age_1950_1959',
                                     'h_age_1940_1949', 'h_age_older_1939']
        census_folder = r"C:\work\water_use\mldataset\ml\training\features\6\census"
        aggregate_census_data(census_folder=census_folder, census_prefix="cs_wsa_", cs_features=cs_features,
                              skip_wkey='_yearly')