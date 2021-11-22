import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas


db_root = r"C:\work\water_use\mldataset"

add_places_pop_to_training = False
fix_pop = False


# =========================================
# add places census data
# =========================================
if add_places_pop_to_training:
    wsa_places_shp = geopandas.read_file(r"C:\work\water_use\gis_data\WSA_Census_Places_Maping_intersect.shp")
    place_pop_ts = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\misc_features\population_by_pace_census\SUB-EST2020_ALL.csv")
    agg_pop_df = pd.read_csv(os.path.join(db_root, r"ml\training\misc_features\WSA_V1_fromCheryl.csv"))
    ag_pop_year_df = pd.read_excel(os.path.join(db_root,
                                                r"ml\training\misc_features\V1_polys_with_water_service_08072021_for_Ayman_popsrv_year.xlsx"),
                                   sheet_name="V1_1polyswWS_popsrvyr")
    wsa_places_intersect_density = geopandas.read_file(r"C:\work\water_use\gis_data\wsa_census_places_den.shp")

    wsa_pop_from_places = wsa_places_intersect_density.groupby(by=['WSA_AGIDF']).sum()
    place_pop_ts = place_pop_ts[place_pop_ts['PLACE'] != 99990]
    place_pop_ts = place_pop_ts[place_pop_ts['SUMLEV'] > 50]
    place_pop_ts = place_pop_ts[place_pop_ts['SUMLEV'] == 162]
    place_pop_ts['place_fips'] = place_pop_ts['PLACE'].astype(str).str.zfill(5)
    place_pop_ts['state_fips'] = place_pop_ts['STATE'].astype(str).str.zfill(2)
    place_pop_ts['complte_fips'] = place_pop_ts[['state_fips', 'place_fips']].agg(''.join, axis=1)
    if 1:
        xx = wsa_places_intersect_density.merge(place_pop_ts, left_on='PLACEFIPS', right_on='complte_fips', how='left')
        wsa_area = xx[['WSA_AGIDF', 'wsaPlaceKM']]
        wsa_area = wsa_area.groupby(by=['WSA_AGIDF']).sum()
        wsa_area.reset_index(inplace=True)
        xx = xx.merge(wsa_area, left_on='WSA_AGIDF', right_on='WSA_AGIDF', how='left')
        scaling = xx['wsaPlaceKM_x']/xx['wsaPlaceKM_y']
        cols = ['POPESTIMATE2010',
                 'POPESTIMATE2011', 'POPESTIMATE2012', 'POPESTIMATE2013',
                 'POPESTIMATE2014', 'POPESTIMATE2015', 'POPESTIMATE2016',
                 'POPESTIMATE2017', 'POPESTIMATE2018', 'POPESTIMATE2019',
                 'POPESTIMATE2020']
        for col in cols:
            xx[col] = xx[col] * scaling

        xx = xx.groupby(by=['WSA_AGIDF']).sum()
        xx.reset_index(inplace=True)
        xx = xx[['WSA_AGIDF', 'popPlcwsa', 'POPESTIMATE2010',
                 'POPESTIMATE2011', 'POPESTIMATE2012', 'POPESTIMATE2013',
                 'POPESTIMATE2014', 'POPESTIMATE2015', 'POPESTIMATE2016',
                 'POPESTIMATE2017', 'POPESTIMATE2018', 'POPESTIMATE2019',
                 'POPESTIMATE2020']]

        usgs_pop = agg_pop_df[['WSA_AGIDF', 'TPOPSRV']]
        xx = xx.merge(usgs_pop, left_on='WSA_AGIDF', right_on='WSA_AGIDF', how='left')

        ag_pop_year_df['year'] = ag_pop_year_df['POP_METH'].str[-4:]
        ag_pop_year_df['year'] = ag_pop_year_df['year'].astype(int)
        ag_pop_year_df.loc[ag_pop_year_df['year'] == 19, 'year'] = 2019
        ag_pop_year_df[['WSA_AGIDF', 'year', 'POP_SRV']]
        ag_pop_year_df = ag_pop_year_df.groupby(['WSA_AGIDF']).agg({'POP_SRV': 'sum', 'year': 'mean'})
        ag_pop_year_df.reset_index(inplace=True)

        xx = xx.merge(ag_pop_year_df, left_on='WSA_AGIDF', right_on='WSA_AGIDF', how='left')

        xx.to_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\wsa_places_pop.csv")

    # ag_pop_year_df['year'] = ag_pop_year_df['POP_METH'].str[-4:]
    # ag_pop_year_df['year'] = ag_pop_year_df['year'].astype(int)
    # ag_pop_year_df.loc[ag_pop_year_df['year'] == 19, 'year'] = 2019
    # ag_pop_year_df[['WSA_AGIDF', 'year', 'POP_SRV']]
    # ag_pop_year_df = ag_pop_year_df.groupby(['WSA_AGIDF']).agg({'POP_SRV': 'sum', 'year': 'mean'})
    # ag_pop_year_df.reset_index(inplace=True)
    #
    #
    # def calc_pop(group):
    #     sum_ = group['Shape_Area'].sum()
    #     weight = (group['Shape_Area'].values / sum_)
    #     ave = np.sum(group['POPULATION'].values * weight)
    #     return pd.Series([ave], index=['population'])
    #
    #
    # xx = wsa_places_shp.groupby(['WSA_AGIDF', 'PLACEFIPS']).apply(calc_pop)
    # xx = xx.reset_index()
    #
    # xx = xx.merge(place_pop_ts, left_on='PLACEFIPS', right_on='complte_fips', how='left')
    # xx = xx[['WSA_AGIDF', 'population', 'POPESTIMATE2010',
    #          'POPESTIMATE2011', 'POPESTIMATE2012', 'POPESTIMATE2013',
    #          'POPESTIMATE2014', 'POPESTIMATE2015', 'POPESTIMATE2016',
    #          'POPESTIMATE2017', 'POPESTIMATE2018', 'POPESTIMATE2019',
    #          'POPESTIMATE2020']]
    # xx = xx.groupby('WSA_AGIDF').sum()
    # xx = xx.reset_index()
    # xx = xx.merge(ag_pop_year_df, left_on='WSA_AGIDF', right_on='WSA_AGIDF', how='left')
    # xx.to_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\wsa_places_pop.csv")
    # cc = 1



def filter_out_rapid_pop_changes(_df):
    """
    - Census data before 2010 is not great, try to reinterpolate.
    - some times population growth is abnormal

    :param _df: has two columns year, tract_pop
    :return:
    """
    max_growth_rate = 5.0 / 100
    _df['diff'] = _df['population'].diff().values[1:].tolist() + [0]
    mask = abs(_df['diff']) / _df['population'] < max_growth_rate
    ref_df = _df[mask]

    ref_df = ref_df[ref_df['Year'] < 2020] # census population for 2020 is not great

    ref_df = ref_df[(ref_df['diff'] < ref_df['diff'].quantile(0.9)) & (
            ref_df['diff'] > ref_df['diff'].quantile(0.1))]

    if len(ref_df) == 0:
        ref_df = _df[mask]

    pop_ref = ref_df['population'].mean()
    mean_increase = ref_df['diff'].mean()
    year_ref = np.mean(ref_df['Year'])
    new_pop = (pop_ref - (_df['Year'] - year_ref) * mean_increase)[~mask]

    return new_pop

if fix_pop:

    wsa_pop = pd.read_excel(r"C:\work\water_use\mldataset\ml\training\misc_features\V1_polys_with_water_service_08072021_for_Ayman_popsrv_year.xlsx")
    train_db = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\wu_annual_training.csv")
    place_pop = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\population_by_pace_census\SUB-EST2020_ALL.csv")
    place_wsa_map = geopandas.read_file(r"C:\work\water_use\gis_data\wsa_census_places_den.shp")

    tract_pop = train_db[['sys_id', 'Year', 'population']]
    yy = tract_pop.groupby(['sys_id'], group_keys=False).apply(filter_out_rapid_pop_changes)

    # add years
    wsa_pop['year'] = wsa_pop['POP_METH'].str[-4:]
    wsa_pop['year'] = wsa_pop['year'].astype(int)
    wsa_pop.loc[wsa_pop['year'] == 19, 'year'] = 2019

    wu_train_db = train_db[train_db['wu_rate']>0]
    nwu_train_db = train_db[~(train_db['wu_rate'] > 0)]

    all_sys_id = set(train_db['sys_id'].unique())
    sys_with_wu = set(wu_train_db['sys_id'].unique())
    sys_with_nwu = all_sys_id.difference(sys_with_wu)

    # ====== (1) work on systems that has water use data
    wsa_pop['pp_ratio'] = np.abs(wsa_pop['POP_SRV'] - (wsa_pop['POLY_POP'])) / wsa_pop['POP_SRV']
    sys_above_30 = wsa_pop[wsa_pop['pp_ratio']>0.3]['WSA_AGIDF']
    sys_above_30 = sys_above_30.unique()
    bad_train_db = wu_train_db[wu_train_db['sys_id'].isin(sys_above_30)]
    wsa_pop = wsa_pop[['WSA_AGIDF', 'POP_SRV', 'POLY_POP', 'year']]
    bad_train_db = bad_train_db.merge(wsa_pop, how = 'left', left_on = 'sys_id', right_on='WSA_AGIDF')
    # ====== (2) Work on syste, that has no water use data
    # todo: some large systems has no historical water use






    xx = 1