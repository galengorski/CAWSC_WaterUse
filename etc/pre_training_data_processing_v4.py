import os, sys
import pandas as pd
import geopandas
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as pdd
import join_county_to_wsa
import time

import iwateruse.spatial_feat_utils as spatial_feat

db_root = r"C:\work\water_use\mldataset"
annual_training_file = os.path.join(db_root, r"ml\training\train_datasets\annual\wu_annual_training3.csv")
monthly_training_file = os.path.join(db_root,
                                     r"C:\work\water_use\mldataset\ml\training\train_datasets\Monthly\wu_monthly_training3.csv")
swud_df = pd.read_csv(os.path.join(db_root,
                                   r"ml\training\targets\monthly_annually\SWUDS_v17.csv"),
                      encoding='cp1252')
nonswuds_df = pd.read_csv(os.path.join(db_root,
                                       r"ml\training\targets\monthly_annually\nonswuds_wu_v4_clean.csv"),
                          encoding='cp1252')
joint_db = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\Join_swud_nswud3.csv")
tot_gb_pop = pd.read_csv(r"C:\work\water_use\blockgroup\gb_wsa_pop.csv")  # ?#
wsa_shp = geopandas.read_file(r"C:\work\water_use\mldataset\gis\wsa\WSA_v1.shp")
wsa_county = geopandas.read_file(r"C:\work\water_use\mldataset\gis\wsa_county_map\wsa_county_map.shp")
awuds_df = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\awuds_all_years.csv")
sell_buy_df = pd.read_csv(
    r"C:\work\water_use\mldataset\ml\training\misc_features\selling_buying\water_exchange_data.csv", encoding='cp1252')
land_use_fn = r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\NWALT_landUse_summarized_for_ICITE_033021.xlsx"
koppen_climate_df = geopandas.read_file(
    r"C:\work\water_use\mldataset\gis\koppen_maps\wsa_climate_intersection_proj.shp")
agg_pop_df = pd.read_csv(os.path.join(db_root, r"ml\training\misc_features\WSA_V1_fromCheryl.csv"))  # ?#
ag_pop_year_df = pd.read_csv(
    os.path.join(db_root, r"ml\training\misc_features\V1_polys_with_water_service_06022021_for_GIS.csv"),
    encoding='cp1252')  # ?#
places_pop_df = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\wsa_places_pop.csv")
wsa_local_pop = pd.read_csv(
    r"C:\work\water_use\mldataset\ml\training\misc_features\V1_polys_with_water_service_08072021_for_Ayman_popsrv_year.csv",
    encoding='cp1252')  # ?#
monthly_climate = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\features\monthly_climate.csv")
annual_climate = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\features\annual_climate.csv")
county_shp = geopandas.read_file(r"C:\work\water_use\mldataset\gis\tl_2016_us_county\tl_2016_us_county.shp")

collect_all_annual_data = False
collect_all_monthly_data = False
drop_repeated_records = False
add_full_climate = False
fill_lat_lon_gaps = False
add_AWUS_data = False
add_zillow_data = False
add_building_footprint = False
compute_n_houses = False
add_water_exchange_data = False
use_fractions_of_income_age = False
add_county_data = False
add_state_fips = False
Add_Manufacturing_establishments = False
add_land_use = False
add_area = False
add_climate_classes = False
compute_education_perc = False
add_enhanced_pop = False
update_wu_data = False
make_negative_nan = False

# =======================================
# Collect Annual Data
# =======================================
if collect_all_annual_data:
    wu = []
    fdb_root = os.path.join(db_root, r"ml\training\features")
    huc2_folders = os.listdir(fdb_root)
    for huc2_folder in huc2_folders:
        fn = os.path.join(fdb_root, os.path.join(huc2_folder, "assemble"))
        fn = os.path.join(fn, "train_db2_{}.csv".format(huc2_folder))
        if os.path.isfile(fn):
            wu_ = pd.read_csv(fn).set_index('Unnamed: 0')
            wu.append(wu_)

    wu = pdd.concat(wu)
    train_db = wu.compute()

    # drop dublications
    mask = train_db.duplicated(['sys_id', 'Year'], keep=False)
    dublicated = train_db[mask]
    dublicated = dublicated.groupby(by=['sys_id', 'Year']).mean()
    dublicated.reset_index(inplace=True)
    dublicated['HUC2'] = dublicated['HUC2'].round()
    train_db = train_db[~mask]
    train_db = pd.concat([train_db, dublicated])
    train_db = train_db[train_db['sys_id'].isin(wsa_shp['WSA_AGIDF'])]
    train_db.to_csv(annual_training_file, index=False)
    del (wu)
else:
    try:
        train_db = pd.read_csv(annual_training_file)
    except:
        print("Download Fail... try something else")
        train_db = pd.read_csv(annual_training_file, engine='python')

# =======================================
# Collect Monthly data
# =======================================
"""
Since census data are changing slowly, this code interpolates obtain monthly 
census from annual using linear interpolation. 
"""
if collect_all_monthly_data:
    train_db_monthly = train_db[['sys_id', 'Year']].copy()
    train_db_monthly = train_db_monthly.drop_duplicates(subset=['Year', 'sys_id'], keep='last').reset_index(drop=True)
    for mo in range(1, 13):
        train_db_monthly[mo] = mo
    train_db_monthly = train_db_monthly.melt(id_vars=['sys_id', 'Year'], value_name='month')
    del (train_db_monthly['variable'])

    train_db_monthly = train_db_monthly.merge(train_db, how='left', on=['Year', 'sys_id'])
    #TOdo: THE FOLLOWING VARUIABLES ARE INCORRECT
    climate_variables = ['etr_warm', 'etr_cool', 'etr', 'pr_warm', 'pr_cool',
                         'pr', 'tmmn_warm', 'tmmn_cool', 'tmmn', 'tmmx_warm', 'tmmx_cool',
                         'tmmx', 'wu_rate']

    for var in climate_variables:
        del (train_db_monthly[var])

    monthly_climate.rename(columns={'year': 'Year'}, inplace=True)
    train_db_monthly = train_db_monthly.merge(monthly_climate, how='left', on=['Year', 'sys_id'])
    train_db_monthly.to_csv(monthly_training_file, index=False)

# =======================================
# drop repeated records
# =======================================
if drop_repeated_records:
    """ When a system exists in two huc2s, the census data collector download the data twice"""
    train_db = train_db.drop_duplicates(subset = ['sys_id', 'Year'], keep = 'first')
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# add annual climate data
# =======================================
if add_full_climate:
    climate_variables = ['etr_warm', 'etr_cool', 'etr', 'pr_warm', 'pr_cool',
                             'pr', 'tmmn_warm', 'tmmn_cool', 'tmmn', 'tmmx_warm', 'tmmx_cool',
                             'tmmx']
    train_db.drop(columns = climate_variables, inplace = True)
    annual_climate.rename(columns = {'year':'Year'}, inplace = True)
    train_db = train_db.merge(annual_climate, on = ['sys_id', 'Year'], how = 'left')
    train_db.to_csv(annual_training_file, index=False)




# =======================================
# Fill gaps in LAT/LON data
# =======================================
if fill_lat_lon_gaps:
    sys_coord_df = wsa_shp[['WSA_AGIDF']].copy()
    sys_coord_df['geometry'] = wsa_shp['geometry'].centroid
    sys_coord_df = geopandas.GeoDataFrame(sys_coord_df, geometry='geometry', crs=wsa_shp.crs)
    sys_coord_df['X'] = sys_coord_df['geometry'].geometry.x
    sys_coord_df['Y'] = sys_coord_df['geometry'].geometry.y
    sys_coord_df.to_crs(epsg=4326, inplace=True)
    sys_coord_df['LAT'] = sys_coord_df['geometry'].geometry.y
    sys_coord_df['LONG'] = sys_coord_df['geometry'].geometry.x
    del (train_db['LONG'])
    del (train_db['LAT'])
    train_db = train_db.merge(sys_coord_df[['WSA_AGIDF', 'X', 'Y', 'LAT', 'LONG']], how='left', left_on='sys_id',
                              right_on='WSA_AGIDF')
    del (train_db['WSA_AGIDF'])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add AWUDS data
# =======================================
if add_AWUS_data:
    awuds_df = awuds_df[awuds_df['YEAR'] > 2000]
    awuds_df['STATECODE'] = awuds_df['STATECODE'].astype(str).str.zfill(2)
    awuds_df['COUNTYCODE'] = awuds_df['COUNTYCODE'].astype(str).str.zfill(3)
    awuds_df['county_id'] = awuds_df['STATECODE'].astype(str) + awuds_df['COUNTYCODE']
    county_shp.to_crs(crs=wsa_shp.crs, inplace=True)
    wsa_county_map = geopandas.sjoin(wsa_shp, county_shp, how="left")
    wsa_county_map[['WSA_AGIDF', 'GEOID', 'NAME']]
    wsa_county_map['area'] = wsa_county_map.geometry.area
    wsa_county_map = wsa_county_map.sort_values(by=['area'])
    wsa_county_map = wsa_county_map.sort_values(by=['area'], ascending=False)
    wsa_county_map = wsa_county_map.drop_duplicates(subset='WSA_AGIDF', keep="first")
    wsa_county_map = wsa_county_map[['WSA_AGIDF', 'GEOID']]
    awuds_df_ = awuds_df[['county_id', 'PS-WTotl', 'PS-DelDO', 'PS-TOPop']]
    awuds_df_ = awuds_df_.groupby(by=['county_id']).mean().reset_index()
    wsa_county_map = wsa_county_map.merge(awuds_df_, how='left', left_on='GEOID', right_on='county_id')
    wsa_county_map.rename(
        columns={'PS-WTotl': 'awuds_totw_cnt', 'PS-DelDO': 'awuds_dom_cnt', 'PS-TOPop': 'awuds_pop_cnt',
                 'WSA_AGIDF': 'sys_id'}, inplace=True)
    wsa_county_map.drop(['GEOID'], axis=1, inplace=True)
    train_db = train_db.merge(wsa_county_map, how='left', on='sys_id')
    train_db['awuds_totw_cnt'] = train_db['awuds_totw_cnt'] * 1e6
    train_db['awuds_dom_cnt'] = train_db['awuds_dom_cnt'] * 1e6
    train_db['awuds_pop_cnt'] = train_db['awuds_pop_cnt'] * 1e3
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# add Zillow
# =======================================
if add_zillow_data:
    zillow_folder = r"C:\work\water_use\mldataset\ml\training\misc_features\zillow"
    zillow_files = ["zillow_wsa_2018", "zillow_wsa_2019", "zillow_wsa_2021"]

    all_zillow = []
    for file in zillow_files:
        fn = os.path.join(zillow_folder, file + ".csv")
        zdf = pd.read_csv(fn)
        zdf['zill_nhouse'] = zdf['NoOfHouses_sum']
        cols = ['WSA_AGIDF', 'zill_nhouse', 'LotSizeSquareFeet_sum', 'YearBuilt_mean', 'BuildingAreaSqFt_sum',
                'TaxAmount_mean', 'NoOfStories_mean']
        zdf = zdf[cols]

        for c in cols:
            if 'WSA_AGIDF' in c:
                continue

            zdf.loc[zdf[c] < 1, c] = np.nan
        all_zillow.append(zdf.copy())

    all_zillow = pd.concat(all_zillow)
    all_zillow = all_zillow.groupby(by=['WSA_AGIDF']).median()
    all_zillow.reset_index(inplace=True)
    train_db = train_db.merge(all_zillow, how='left', right_on='WSA_AGIDF', left_on='sys_id')
    del (train_db['WSA_AGIDF'])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# add Building Footprint
# =======================================
if add_building_footprint:
    fn_blg_footprint = r"C:\work\water_use\mldataset\ml\training\misc_features\buildingfootprints\us_building_footprint.csv"
    blg_ft_df = pd.read_csv(fn_blg_footprint)
    blg_ft_df = blg_ft_df[['WSA_AGIDF', 'count', 'gt_2median', 'gt_4median', 'lt_2median', 'lt_4median']].copy()
    blg_ft_df.rename(columns={'count': 'bdg_ftp_count', 'gt_2median': 'bdg_gt_2median', 'gt_4median': 'bdg_gt_4median',
                              'lt_2median': 'bdg_lt_2median', 'lt_4median': 'bdg_lt_4median'},
                     inplace=True)

    train_db = train_db.merge(blg_ft_df, how='left', right_on='WSA_AGIDF', left_on='sys_id')
    del (train_db['WSA_AGIDF'])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add Water Exchange data
# =======================================
if add_water_exchange_data:

    sell_buy_df['WSA_AGIDF'] = sell_buy_df['WSA_AGIDF'].str.replace("'", "")
    sell_buy_df = sell_buy_df.drop_duplicates(subset=['WSA_AGIDF'])
    sell_buy_df = sell_buy_df[['WSA_AGIDF', 'BuySellFlag2']]
    sell_buy_df.rename(columns={'BuySellFlag2': 'Ecode'}, inplace=True)
    train_db = train_db.merge(sell_buy_df, left_on='sys_id', right_on='WSA_AGIDF', how='left')
    categories = train_db['Ecode'].unique()
    for i, categ in enumerate(categories):
        mask = train_db['Ecode'] == categ
        train_db.loc[mask, 'Ecode_num'] = i
    del (train_db['WSA_AGIDF'])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add total number of houses
# =======================================
if compute_n_houses:
    def compute_average_house_age(df, ref_year=2022):
        year_dict = {'h_age_newer_2005': 2005,
                     'h_age_2000_2004': 2002,
                     'h_age_1990_1999': 1994.5,
                     'h_age_1980_1989': 1984.5,
                     'h_age_1970_1979': 1974.5,
                     'h_age_1960_1969': 1964.5,
                     'h_age_1950_1959': 1954.5,
                     'h_age_1940_1949': 1944.5,
                     'h_age_older_1939': 1939

                     }
        cols = year_dict.keys()
        df['av_house_age'] = 0
        df['hhsum'] = 0
        for col in cols:
            df['av_house_age'] = df['av_house_age'] + np.abs(df[col]) * (ref_year - year_dict[col])
            df['hhsum'] = df['hhsum'] + np.abs(df[col])

        df['av_house_age'] = df['av_house_age'] / df['hhsum']
        del (df['hhsum'])
        return df


    def fix_house_negative_age(df):
        "This is the raw train_Df"
        df.loc[df['median_h_year'] < 1930, 'median_h_year'] = 1930
        df.loc[df['median_h_year'] > 2010, 'median_h_year'] = 2010

        ag_cols = [s for s in df.columns if "h_age_" in s]
        for col in ag_cols:
            df.loc[df[col] < 0, col] = 0
        df = compute_average_house_age(df, ref_year=2022)

        # compute % of houses
        house_age = df[ag_cols]
        df['n_houses'] = house_age.sum(axis=1)
        for feat in ag_cols:
            df["prc_" + feat] = 100 * df[feat] / df['n_houses']

        return df


    train_db = fix_house_negative_age(train_db)
    # train_db['pop_house_ratio'] = train_db['population']/train_db['n_houses']
    # train_db['family_size'] = train_db['population'] / train_db['households2']
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# fraction of income and age
# =======================================
if use_fractions_of_income_age:

    def fix_negative_income(df):
        income_cols = [s for s in df.columns if "income_" in s]
        for feat in income_cols:
            df.loc[df[feat] < 0, feat] = 0

        sum_income = df[income_cols].sum(axis=1)

        for feat in income_cols:
            df["prc_" + feat] = 100.0 * df[feat] / sum_income

        df.loc[df['median_income'] < 20000, 'median_income'] = 20000

        # compute mean income
        income_dic = {}
        for feat in income_cols:
            if "_lt_" in feat or "_gt_" in feat:
                v = feat.split("_")[2]
                v = v.upper()
                ave = 1000.0 * int(v.replace("K", ""))

            else:
                vals = feat.split("_")[1:]
                sumv = 0
                for v in vals:
                    v = v.upper()
                    v = 1000.0 * int(v.replace("K", ""))
                    sumv = sumv + v
                ave = sumv / 2.0
            income_dic[feat] = ave
        df['average_income'] = 0
        for feat in income_cols:
            inc = income_dic[feat]
            df['average_income'] = df['average_income'] + inc * df["prc_" + feat]

        return df


    train_db = fix_negative_income(train_db)
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add economic county data
# =======================================
if add_county_data:
    fid = open(r"C:\work\water_use\mldataset\ml\training\misc_features\parameters_eco.txt", 'r')
    content = fid.readlines()
    eco_info = {}
    for line in content:
        if line.strip()[0] == '#':
            continue
        field_key = line.strip().split(":")
        eco_info[field_key[0]] = field_key[1]
    fid.close()

    fid = open(r"C:\work\water_use\mldataset\ml\training\misc_features\parameters.txt", 'w')
    fields = list(eco_info.keys())
    for field in fields:
        fid.write(field)
        fid.write("\n")
    fid.close()
    old_fields = train_db.columns
    train_db = join_county_to_wsa.add_county_data(train_db)

    for field in train_db.columns:
        if field in old_fields:
            continue
        if field in ['fips']:
            continue
        for ff in fields:
            ff2 = ff.lower()
            if field in ff2:
                break
            else:
                continue
        train_db[eco_info[ff]] = train_db[ff2]
        del (train_db[ff2])

    sub_fields = ['income_', 'n_jobs_', 'indus_', 'rur_urb_', 'unemployment_']
    for sub_ in sub_fields:
        curr_set = []
        for yr in range(2000, 2021):
            nm = sub_ + str(yr)
            if nm in train_db.columns:
                curr_set.append(nm)

        df_ = train_db[curr_set]
        fmean = df_.mean(axis=1)
        fnname = sub_ + "cnty"
        train_db[fnname] = fmean

        for ff in curr_set:
            del (train_db[ff])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add state fips from county data
# =======================================
if add_state_fips:
    train_db['state_id'] = (train_db['county_id'] / 1000.0).astype(int)
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add economic county data 2 - number of establishments
# =======================================
if Add_Manufacturing_establishments:
    train_db['county_id'] = train_db['county_id'].astype(int)
    manu_folder = r"C:\work\water_use\mldataset\ml\training\misc_features\ml_county_to_sa\county_data\US_County"
    fn2012 = os.path.join(manu_folder, r"US_MFG_SIC31-33_County_2012_20200911.csv")
    fn2017 = os.path.join(manu_folder, r"US_MFG_SIC31-33_County_2017_20200911.csv")
    train_db['county_ids'] = train_db['county_id'].astype(str).str.zfill(5)

    df = pd.read_csv(fn2012)
    df['countysss'] = df['id'].str[-5:]
    df = df[df['Meaning of 2012 NAICS code'] == 'Manufacturing']
    df = df[['countysss', 'Number of establishments']]
    df['Num_establishments_2012'] = df['Number of establishments']
    del (df['Number of establishments'])
    df['Num_establishments_2012'] = pd.to_numeric(df['Num_establishments_2012'], errors='coerce')
    train_db = train_db.merge(df, left_on='county_ids', right_on='countysss', how='left')
    del (train_db['countysss'])

    df = pd.read_csv(fn2017)
    df['countysss'] = df['id'].str[-5:]
    df = df[df['Meaning of NAICS code'] == 'Manufacturing']
    df = df[['countysss', 'Number of establishments']]
    df['Num_establishments_2017'] = df[['Number of establishments']]
    del (df['Number of establishments'])
    df['Num_establishments_2017'] = pd.to_numeric(df['Num_establishments_2017'], errors='coerce')
    train_db = train_db.merge(df, left_on='county_ids', right_on='countysss', how='left')

    del (train_db['county_ids'])
    del (train_db['countysss'])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add Land Use Data
# =======================================
if add_land_use:
    # Theobald
    df_ = pd.read_excel(
        r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\ICITE_Theobald_zonal_stats_033121.xlsx",
        sheet_name='finalSUMbyICITE')
    train_db = train_db.merge(df_, left_on='sys_id', right_on='WSA_AGIDF', how='left')
    del (train_db['WSA_AGIDF'])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add area
# =======================================
if add_area:
    area_shp = wsa_shp[['WSA_SQKM', 'WSA_AGIDF']]
    train_db = train_db.merge(area_shp, left_on='sys_id', right_on='WSA_AGIDF', how='left')
    del (train_db['WSA_AGIDF'])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# add Koppen Climate classification
# =======================================
if add_climate_classes:
    # there is error in some wsa where 0 is missing fron train_db name
    db_ = set(train_db['sys_id'].unique())
    cl_ = set(koppen_climate_df['WSA_AGIDF'].unique())
    sys_id_with_issues = db_.difference(cl_)
    for ss in sys_id_with_issues:
        ss_ = "0" + ss
        if ss_ in cl_:
            train_db.loc[train_db['sys_id'] == ss, 'sys_id'] = ss_

    idx = koppen_climate_df.groupby(['WSA_AGIDF'])['area_calc'].transform(max) == koppen_climate_df['area_calc']
    koppen_climate_df = koppen_climate_df[idx][['WSA_AGIDF', 'gridcode']]
    train_db = train_db.merge(koppen_climate_df, left_on='sys_id', right_on='WSA_AGIDF', how='left')
    train_db['KG_climate_zone'] = train_db['gridcode']
    del (train_db['WSA_AGIDF'])
    del (train_db['gridcode'])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# education
# =======================================
if compute_education_perc:
    edu_cols = ['n_lt_ninth_gr	n_ninth_to_twelth_gr	n_hs_grad	n_some_college	n_associates	n_bachelors	n_masters_phd']
    edu_cols = edu_cols[0].strip().split()
    del(train_db['n_tot_ed_attain'])
    sumedu = train_db[edu_cols].sum(axis = 1)
    sumedu[sumedu <= 0] = np.nan

    for feat in edu_cols:
        train_db["prc_"+feat] = 100.0* train_db[feat]/sumedu
        del(train_db[feat])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# add enhanced population
# =======================================
if add_enhanced_pop:
    master_pop = pd.read_csv(r"C:\work\water_use\ml_experiments\annual_v_0_0\master_population.csv")
    master_pop.rename(columns = {'WSA_AGIDF':'sys_id'}, inplace = True)
    master_pop = master_pop[['sys_id', 'pop', 'Year']]
    master_pop = master_pop.groupby(by=['sys_id', 'Year']).mean()
    master_pop.reset_index(inplace=True)
    train_db = train_db.merge(master_pop, on=['sys_id', 'Year'], how = 'left')
    mask = train_db['pop'].isna()
    train_db.loc[mask, 'pop'] = train_db[mask]['population']
    train_db.to_csv(annual_training_file, index=False)

if make_negative_nan:
    neg_cols = []
    for col in train_db.columns:
        try:
            if np.any(train_db[col]<0):
                neg_cols.append(col)
        except:
            print("{} not numeric".format(col))
    for col in neg_cols:
        if col in ['X', 'Y', 'LAT', 'LONG', 'pr_cumdev']:
            continue
        train_db.loc[train_db[col]<0, col] = np.nan
    train_db.to_csv(annual_training_file, index=False)

    pass



if 0:

    # =======================================
    # add update water use
    # =======================================
    annual_wu = pd.read_csv(r"annual_wu.csv")

    # =======================================
    # add spatial averages
    # =======================================
    # add population
    pop_master = pd.read_csv(r"C:\work\water_use\ml_experiments\annual_v_0_0\master_population.csv")
    pop_master = pop_master[['WSA_AGIDF', 'Year', 'population_c', 'pop', 'pop_swud16', 'TPOPSRV']]
    pop_master.rename(columns={'WSA_AGIDF': 'sys_id'}, inplace=True)
    train_db = train_db.merge(pop_master, how='left', on=['sys_id', 'Year'])

    annual_wu = pd.read_csv(r"C:\work\water_use\ml_experiments\annual_v_0_0\annual_wu.csv")
    annual_wu = annual_wu[annual_wu['inWSA'] == 1].copy()
    tn_mask = annual_wu['WSA_AGIDF'].str.contains("TN")
    annual_wu.loc[tn_mask, 'annual_wu_G_nonswuds'] = np.nan

    # drop outliers
    annual_wu = annual_wu[annual_wu['flg_annual_isdiff_annualize_month_swud'] == 0]
    annual_wu = annual_wu[annual_wu['flg_no_annual'] == 0]
    annual_wu = annual_wu[(annual_wu['prc_time_change_swud'].isna()) | (annual_wu['prc_time_change_swud'] < 20)]

    # annual_wu['wu_rate'] = annual_wu[['annual_wu_G_swuds', 'annual_wu_G_nonswuds']].mean(axis = 1)/annual_wu['days_in_year']
    annual_wu['wu_rate'] = annual_wu['annual_wu_G_swuds'] / annual_wu['days_in_year']
    annual_wu = annual_wu[annual_wu['wu_rate'] > 0]
    annual_wu = annual_wu[['WSA_AGIDF', 'YEAR', 'wu_rate']]
    annual_wu.rename(columns={'WSA_AGIDF': 'sys_id', 'YEAR': 'Year', 'wu_rate': 'wu_rate2'}, inplace=True)
    train_db = train_db.merge(annual_wu, how='left', on=['sys_id', 'Year'])

    train_db['pc_pop'] = train_db['wu_rate2'] / train_db['pop']
    train_db['pc_swud'] = train_db['wu_rate2'] / train_db['pop_swud16']
    train_db['pc_tpopsrv'] = train_db['wu_rate2'] / train_db['TPOPSRV']

    df_spatial = spatial_feat.generate_local_statistics(train_db, sys_id_col='sys_id',
                                                        max_points=50, raduis=500)

    xx = 1

