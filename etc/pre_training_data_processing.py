import os, sys
import pandas as pd
import geopandas
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as pdd
import join_county_to_wsa

db_root = r"C:\work\water_use\mldataset"
annual_training_file = os.path.join(db_root, r"ml\training\train_datasets\annual\wu_annual_training.csv")
monthly_training_file = os.path.join(db_root, r"C:\work\water_use\mldataset\ml\training\train_datasets\Monthly\wu_monthly_training.csv")
swud_df = pd.read_csv(os.path.join(db_root, r"ml\training\targets\monthly_annually\swud_v15.csv"),  encoding='cp1252')
tot_gb_pop = pd.read_csv(r"C:\work\water_use\blockgroup\gb_wsa_pop.csv")
wsa_shp = geopandas.read_file(r"C:\work\water_use\mldataset\gis\wsa\WSA_v3_02072021\WSA_v3_alb83_02072021.shp")
wsa_county = geopandas.read_file(r"C:\work\water_use\mldataset\gis\wsa_county_map\wsa_county_map.shp")
awuds_df = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\awuds_all_years.csv")
sell_buy_df = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\selling_buying\water_exchange_info.csv")
land_use_fn = r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\NWALT_landUse_summarized_for_ICITE_033021.xlsx"
koppen_climate_df = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\kg_zones\kg_wsa_zones.csv")
agg_pop_df = pd.read_csv(os.path.join(db_root,r"ml\training\misc_features\WSA_V1_fromCheryl.csv"))
ag_pop_year_df = pd.read_excel(os.path.join(db_root, r"ml\training\misc_features\V1_polys_with_water_service_06022021_for_GIS.xlsx"), sheet_name="V1_1polyswWS")

collect_all_annual_data = False
collect_all_monthly_data = False
add_swud_data = False
correct_census_pop_using_swud = False
fill_lat_lon_gaps = False
correct_wGB_population = False
add_AWUS_data = False
add_water_exchange_data = False
compute_n_houses = False
use_fractions_of_income_age = False
compute_n_houses = False
use_fractions_of_income_age = False
add_county_data = False
add_land_use = False
add_area = False
scale_pop_using_land_use = False
More_land_use = False
Add_Manufacturing_establishments = False
add_agg_wsa_pop = False
correct_enhance_pop_using_swud = False
add_climate_classes = False
add_agg_pop_year = True
add_state_fips = False

# =======================================
# Collect Annual Data
# =======================================
if collect_all_annual_data:
    wu = []
    fdb_root = os.path.join(db_root, r"ml\training\features")
    huc2_folders = os.listdir(fdb_root)
    for huc2_folder in huc2_folders:
        fn = os.path.join(fdb_root, os.path.join(huc2_folder, "assemble"))
        fn = os.path.join(fn, "train_db_{}.csv".format(huc2_folder))
        if os.path.isfile(fn):
            wu_ = pdd.read_csv(fn).set_index('Unnamed: 0')
            wu.append(wu_)

    wu = pdd.concat(wu)
    train_db = wu.compute()
    train_db.to_csv(annual_training_file, index = False)
    del(wu)
else:
    try:
        train_db = pd.read_csv(annual_training_file)
    except:
        print("Download Fail... try something else")
        train_db = pd.read_csv(annual_training_file, engine='python')


# =======================================
# Collect Monthly data
# =======================================
if collect_all_monthly_data:
    wu = []
    features_to_use = ['YEAR', 'month', 'wu_rate', 'etr', 'pr', 'tmmn', 'tmmx', 'sys_id']
    fdb_root = os.path.join(db_root, r"ml\training\features")
    huc2_folders = os.listdir(fdb_root)
    for huc2_folder in huc2_folders:
        print(huc2_folder)
        fn = os.path.join(fdb_root, os.path.join(huc2_folder, "assemble"))
        fn = os.path.join(fn, "train_db_monthly_{}.csv".format(huc2_folder))
        if os.path.isfile(fn):
            wu_ = pd.read_csv(fn).set_index('Unnamed: 0')
            wu.append(wu_)

    train_db_monthly = pd.concat(wu)
    train_db_monthly = train_db_monthly[features_to_use]
    #train_db_monthly = wu.compute()

    XY = train_db[['sys_id', 'LAT', 'LONG']]
    XY = XY.drop_duplicates(subset=['sys_id'])
    train_db_monthly = train_db_monthly.merge(XY, left_on='sys_id', right_on='sys_id', how='left')
    train_db_monthly.to_csv(monthly_training_file, index=False)


# =======================================
# Add swud data -- we should not use this per Cheryl B suggestions
# =======================================
add_swud_data = False # should always be false
if add_swud_data:
    train_db = pd.read_csv(annual_training_file)
    sys_ids = train_db['sys_id'].unique()
    train_db['swud_pop'] = 0
    train_db['small_gb_pop'] = 0
    lensys = 1.0* len(sys_ids)
    swud_df[swud_df['POP_SRV']==0] = np.nan
    for i, sys_id in enumerate(sys_ids):
        print(100 * i /lensys )
        pop = swud_df[swud_df['WSA_AGIDF'] == sys_id]['POP_SRV'].mean()
        train_db.loc[train_db['sys_id'] == sys_id, 'swud_pop'] = pop
        try:
            pop2 = tot_gb_pop[tot_gb_pop['WSA_AGIDF'] == sys_id]['pop'].values[0]
        except:
            print("No pop info")
            pop2 = 0
        train_db.loc[train_db['sys_id'] == sys_id, 'small_gb_pop'] = pop2
    train_db.to_csv(annual_training_file, index = False)

# =======================================
# Add Aggrigated Population: this seems to be the best pop estimation
# =======================================
if add_agg_wsa_pop:
    agg_pop_df = agg_pop_df[['WSA_AGIDF', 'TPOPSRV']]
    train_db = train_db.merge(agg_pop_df, left_on='sys_id', right_on= 'WSA_AGIDF', how='left')
    del (train_db['WSA_AGIDF'])
    train_db.to_csv(annual_training_file, index=False)

if add_agg_pop_year:
    def extrat_year(txt):
        new_string = ''
        for s in txt:
            if s.isdigit():
                new_string = new_string + s
            else:
                new_string = new_string + " "

        years = new_string.strip().split()
        years = [int(y) for y in years]
        years = np.array(years)
        years = years[years>1000]
        year = np.mean(years)
        if pd.isna(year):
            return year
        else:
            return int(year)


    ag_pop_year_df['year'] = ag_pop_year_df['POP_METH'].apply(extrat_year)


    pass

# =======================================
# Correct Census Population Using Swud Pop
# =======================================
correct_census_pop_using_swud = False
if correct_census_pop_using_swud:
    train_db['pc_swud'] = train_db['wu_rate']/train_db['swud_pop']
    train_db['pc_tract_data'] = train_db['wu_rate'] / train_db['population']
    mask1 = (train_db['swud_pop']>0)
    mask2 = (train_db['pc_swud']<=200) & (train_db['pc_swud']>=40)
    mask3 = (train_db['pc_tract_data']>200) | (train_db['pc_tract_data']<40)
    mask = mask1 & mask2 & mask3
    sys_ids = train_db.loc[mask, 'sys_id'].unique()
    train_db['pop_swud_corrected'] = train_db['population']
    train_db['swud_corr_factor'] =1.0
    for i, sys_id in enumerate(sys_ids):
        print(i)
        print(i/len(sys_ids))
        mask_sys_id = train_db['sys_id']==sys_id

        curr_df = train_db[mask & mask_sys_id]
        corr_factor = curr_df['swud_pop'].mean() / curr_df['population'].mean()
        if not(corr_factor > 0):
            corr_factor = 1.0

        train_db.loc[ mask_sys_id, 'pop_swud_corrected'] =\
            corr_factor * train_db.loc[ mask_sys_id, 'population'].values
        train_db.loc[mask_sys_id, 'swud_corr_factor'] = corr_factor

    train_db['pc_swud_corrected'] = train_db['wu_rate'] / train_db['pop_swud_corrected']
    train_db['pc_gb_data'] = train_db['wu_rate'] / train_db['small_gb_pop']

    train_db.to_csv(annual_training_file, index = False)

# =======================================
# Fill gaps in LAT/LON data
# =======================================
if fill_lat_lon_gaps:
    sys_ids = wsa_shp['WSA_AGIDF'].unique()
    mask_lat = train_db['LAT'].isna()
    mask_long = train_db['LONG'].isna()
    mask_xy = (mask_lat) | (mask_long)

    sys_no_xy = train_db[mask_xy]['sys_id'].unique()
    counter = 0
    for i, sys_id in enumerate(sys_no_xy):
        if sys_id[0].isdigit():
            if len(sys_id)<9:
                maskcc = train_db['sys_id'] == sys_id
                sys_id = "0" + sys_id
                train_db.loc[maskcc, 'sys_id'] = sys_id

        if not(sys_id in sys_ids):
            print('No XY info')
            continue

        counter = counter + 1
        print((i * 1.0) / len(sys_no_xy))
        mask_id = wsa_shp['WSA_AGIDF']==sys_id
        Lat = wsa_shp.loc[mask_id, 'LAT'].values[0]
        Long = wsa_shp.loc[mask_id, 'LONG'].values[0]
        train_db.loc[train_db['sys_id']==sys_id, 'LAT'] = Lat
        train_db.loc[train_db['sys_id'] == sys_id, 'LONG'] = Long
    mask_lat = train_db['LAT'].isna()
    mask_long = train_db['LONG'].isna()
    mask_xy2 = (mask_lat) | (mask_long)
    fraction_fixed = (1-1.0 * np.sum(mask_xy2))/(1.0 * np.sum(mask_xy))
    print("Fraction of LAT/LONG fixed is {}".format(fraction_fixed*100.0))

    train_db.to_csv(annual_training_file, index = False)

# =======================================
# Try to correct WGB with TGB
# =======================================

if correct_wGB_population:
    train_db['pop_swud_gb_correction'] = train_db['pop_swud_corrected']
    mask_extreme_pc = (train_db['pc_swud_corrected']>700) | ((train_db['pc_swud_corrected']<20) )
    mask2 = (train_db['pc_gb_data']<700) & (train_db['pc_gb_data']>20)
    mask = mask_extreme_pc & mask2
    train_db.loc[mask, 'pop_swud_gb_correction'] = train_db.loc[mask, 'small_gb_pop']
    train_db['pc_swud_gb_corrected'] = train_db['wu_rate']/train_db['pop_swud_gb_correction']
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add AWUDS data
# =======================================
if add_AWUS_data:
    awuds_df = awuds_df[awuds_df['YEAR'] > 2000]
    awuds_df['STATECODE'] = awuds_df['STATECODE'].astype(str).str.zfill(2)
    awuds_df['COUNTYCODE'] = awuds_df['COUNTYCODE'].astype(str).str.zfill(3)
    awuds_df['county_id'] = awuds_df['STATECODE'].astype(str) + awuds_df['COUNTYCODE']

    county_id_df = awuds_df['county_id']
    county_id_shp = wsa_county['GEOID']
    sys_ids = train_db['sys_id'].unique()
    train_db['county_id'] = ''
    train_db['awuds_totw_cnt'] = 0
    train_db['awuds_dom_cnt'] = 0
    train_db['awuds_dom_cnt'] = 0
    train_db['awuds_pop_cnt'] = 0
    for i, sys_id in enumerate(sys_ids):
        print((1.0*i)/len(sys_ids))
        mask = wsa_county['WSA_AGIDF']==sys_id
        try:
            curr_county = wsa_county.loc[mask, 'GEOID' ].values[0]
        except:
            print("No data")
            continue

        mask2 = awuds_df['county_id']==curr_county
        tot_withdrawal = awuds_df.loc[mask2, 'PS-WTotl'].mean() * 1e6
        domestic_del = awuds_df.loc[mask2, 'PS-DelDO'].mean()*1e6
        ps_pop = awuds_df.loc[mask2, 'PS-TOPop'].mean()*1e3

        mask3 = train_db['sys_id'] == sys_id
        train_db.loc[mask3, 'awuds_totw_cnt'] = tot_withdrawal
        train_db.loc[mask3, 'awuds_dom_cnt'] = domestic_del
        train_db.loc[mask3, 'awuds_pop_cnt'] = ps_pop
        train_db.loc[mask3, 'county_id'] = curr_county
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add Water Exchange data
# =======================================
if add_water_exchange_data:

    sell_buy_df['WSA_AGIDF'] = sell_buy_df['WSA_AGIDF'].str.replace("'", "")
    sell_buy_df = sell_buy_df.drop_duplicates(subset = ['WSA_AGIDF'])
    sell_buy_df = sell_buy_df[['WSA_AGIDF', 'Ecode']]
    train_db = train_db.merge(sell_buy_df, left_on = 'sys_id', right_on = 'WSA_AGIDF',  how='left')
    categories = train_db['Ecode'].unique()
    for i, categ in enumerate(categories):
        mask = train_db['Ecode'] == categ
        train_db.loc[mask, 'Ecode_num'] = i
    del(train_db['WSA_AGIDF'])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add total number of houses
# =======================================
if compute_n_houses:
    house_age = train_db[['h_age_newer_2005', 'h_age_2000_2004', 'h_age_1990_1999', 'h_age_1980_1989', 'h_age_1970_1979', 'h_age_1960_1969', 'h_age_1950_1959', 'h_age_1940_1949', 'h_age_older_1939' ]]
    train_db['n_houses'] = house_age.sum(axis = 1)
    train_db['pop_house_ratio'] = train_db['population']/train_db['n_houses']
    train_db['family_size'] = train_db['population'] / train_db['households2']
    train_db.to_csv(annual_training_file, index=False)


# =======================================
# fraction of income and age
# =======================================
if use_fractions_of_income_age:
    income_info = "income_lt_10k	income_10K_15k	income_15k_20k	income_20k_25k	income_25k_30k	income_30k_35k	income_35k_40k" \
    "	income_40k_45k	income_45k_50k	income_50k_60k	income_60k_75k	income_75k_100k	income_100k_125k	income_125k_150k" \
    "	income_150k_200k	income_gt_200k"
    income_feat = income_info.split()
    sum_income = train_db[income_feat].sum(axis = 1)
    for feat in income_feat:
        train_db[feat] = train_db[feat]/sum_income

    hs_age_feats = ['h_age_newer_2005', 'h_age_2000_2004', 'h_age_1990_1999', 'h_age_1980_1989', 'h_age_1970_1979',
                    'h_age_1960_1969', 'h_age_1950_1959', 'h_age_1940_1949', 'h_age_older_1939' ]
    for feat in hs_age_feats:
        train_db[feat] = train_db[feat]/train_db['n_houses']
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add economic county data
# =======================================
if add_county_data:
    fid = open(r"C:\work\water_use\mldataset\ml\training\misc_features\parameters_eco.txt", 'r')
    content = fid.readlines()
    eco_info = {}
    for line in content:
        if line.strip()[0]=='#':
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
    train_db = join_county_to_wsa.add_county_data()

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
        del(train_db[ff2])

    sub_fields = ['income_', 'n_jobs_', 'indus_', 'rur_urb_', 'unemployment_' ]
    for sub_ in sub_fields:
        curr_set = []
        for yr in range(2000,2021):
            nm = sub_+str(yr)
            if nm in train_db.columns:
                curr_set.append(nm)

        df_ = train_db[curr_set]
        fmean = df_.mean(axis = 1)
        fnname = sub_+"cnty"
        train_db[fnname] = fmean

        for  ff in curr_set:
            del(train_db[ff])
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
    manu_folder = r"C:\work\water_use\mldataset\ml\training\misc_features\ml_county_to_sa\county_data\US_County"
    fn2012 = os.path.join(manu_folder, r"US_MFG_SIC31-33_County_2012_20200911.csv")
    fn2017 = os.path.join(manu_folder, r"US_MFG_SIC31-33_County_2017_20200911.csv")
    train_db['county_ids'] = train_db['county_id'].astype(str).str.zfill(5)

    df = pd.read_csv(fn2012)
    df['countysss'] = df['id'].str[-5:]
    df = df[df['Meaning of 2012 NAICS code'] == 'Manufacturing']
    df = df[['countysss', 'Number of establishments']]
    df['Num_establishments_2012'] = df['Number of establishments']
    del(df['Number of establishments'])
    train_db = train_db.merge(df, left_on='county_ids', right_on='countysss', how='left')
    del (train_db['countysss'])

    df = pd.read_csv(fn2017)
    df['countysss'] = df['id'].str[-5:]
    df = df[df['Meaning of NAICS code'] == 'Manufacturing']
    df = df[['countysss', 'Number of establishments']]
    df['Num_establishments_2017'] = df[['Number of establishments']]
    del (df['Number of establishments'])
    train_db = train_db.merge(df, left_on='county_ids', right_on='countysss', how='left')

    del(train_db['county_ids'])
    del (train_db['countysss'])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add Land Use Data
# =======================================
if add_land_use:
    years = [1974, 1982, 1992, 2002, 2012]
    for year in years:
        sheet_name = "Summary LU {}".format(year)
        df_ = pd.read_excel(land_use_fn, sheet_name=sheet_name)
        for col in df_.columns:
            if col in ['WSA_AGIDF']:
                continue
            new_col = col+"_{}".format(year)
            df_[new_col] = df_[col]
            del( df_[col])
        train_db = train_db.merge(df_, left_on='sys_id', right_on='WSA_AGIDF', how='left')
        del(train_db['WSA_AGIDF'])
    # Theobald
    df_ = pd.read_excel(r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\ICITE_Theobald_zonal_stats_033121.xlsx", sheet_name = 'finalSUMbyICITE')
    train_db = train_db.merge(df_, left_on='sys_id', right_on='WSA_AGIDF', how='left')
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add area
# =======================================
if add_area:
    area_shp = wsa_shp[['WSA_SQKM', 'WSA_AGIDF']]
    train_db = train_db.merge(area_shp, left_on='sys_id', right_on='WSA_AGIDF', how='left')
    del(train_db['WSA_AGIDF'])
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Enhance_population usning Landa use
# =======================================
if scale_pop_using_land_use:
    # usgs wall to wall
    lu_usgs = pd.read_excel(r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\NWALT_ZONAL_STATS_Subset_BG_2010_data_042321.xlsx",
                            sheet_name=r"Pivot_table_LU2012")
    lu_theobald = pd.read_excel(r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\ICITE_Theobalt_zonal_stats_BLOCKGROUP_043021.xlsx",
                                sheet_name='finalSUMbyICITE')

    # bg_select_wsa: is the set of BG's that intersect with WSA
    bg_select_wsa = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\bg_intersect_wsa\bg_select_wsa.csv")
    # bg_x_wsa: is the geometrical instersection of SWA and BG maps
    bg_x_wsa = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\bg_intersect_wsa\blc_grp_x_swa.csv")

    #C:\work\water_use\mldataset\gis\subset_BG_2010.shp
    bg_wsa_map = bg_x_wsa[['US_blck_12', 'WSA_AGIDF']]
    pop_tot_bg_map = bg_select_wsa.merge(bg_wsa_map, left_on='GISJOIN', right_on='US_blck_12', how='left')
    pop_tot_bg = pop_tot_bg_map[['GISJOIN', 'WSA_AGIDF', 'H7V001']]
    pop_tot_bg = pop_tot_bg.groupby('WSA_AGIDF').sum()

    # usgs land use corrections
    lu_usgs['Domestic_Total'] = lu_usgs['Sum of Developed'] + lu_usgs['Sum of Semi-Developed']
    lu_usgs = lu_usgs.merge(bg_wsa_map, left_on='Row Labels', right_on='US_blck_12', how='left')
    lu_domestic_total = lu_usgs.groupby(by = 'WSA_AGIDF').sum()
    lu_swa = train_db[['WSA_AGIDF', 'Domestic_Total_2012']]
    lu_swa = lu_swa.set_index('WSA_AGIDF')
    lu_swa['LU_BG'] = lu_domestic_total['Domestic_Total']
    lu_swa['bg_usgs_correction_factor'] = lu_swa['Domestic_Total_2012'] / lu_swa['LU_BG']

    # theobald land use corrections
    lu_theobald = lu_theobald.merge(bg_wsa_map, left_on='GISJOIN', right_on='US_blck_12', how='left')
    lu_domestic_total_theo = lu_theobald.groupby(by='WSA_AGIDF').sum()
    lu_swa_theo = train_db[['WSA_AGIDF', 'Domestic']]
    lu_swa_theo = lu_swa_theo.set_index('WSA_AGIDF')
    lu_swa_theo['LU_BG'] = lu_domestic_total_theo['Domestic'] + lu_domestic_total_theo['Urban_Misc']
    lu_swa_theo['bg_theo_correction_factor'] = lu_swa_theo['Domestic'] / lu_swa_theo['LU_BG']

    train_db = train_db.merge(pop_tot_bg, left_on='sys_id', right_on='WSA_AGIDF', how='left')
    train_db['bg_pop_2010'] = train_db['H7V001']
    del(train_db['H7V001'])
    del(train_db['WSA_AGIDF'])
    del(lu_usgs)
    del(lu_theobald)
    del(bg_select_wsa)
    del(bg_x_wsa)
    del(bg_wsa_map)

    lu_swa.reset_index(inplace=True)
    lu_swa = lu_swa[['WSA_AGIDF', 'bg_usgs_correction_factor']]
    lu_swa_theo.reset_index(inplace=True)
    lu_swa_theo = lu_swa_theo[['WSA_AGIDF', 'bg_theo_correction_factor']]
    lu_swa_theo.drop_duplicates(subset=['WSA_AGIDF'], inplace=True)
    lu_swa.drop_duplicates(subset=['WSA_AGIDF'], inplace=True)

    train_db = train_db.merge(lu_swa[['WSA_AGIDF', 'bg_usgs_correction_factor']],
                              left_on='sys_id', right_on='WSA_AGIDF', how='left')


    train_db = train_db.merge(lu_swa_theo[['WSA_AGIDF', 'bg_theo_correction_factor']],
                              left_on='sys_id', right_on='WSA_AGIDF', how='left')

    cols = train_db.columns
    for col in cols:
        if "WSA_AGIDF" in col:
            print(col)
            del(train_db[col])

    train_db.to_csv(annual_training_file, index=False)

# =======================================
# More Land Use analysis
# =======================================
if More_land_use:

    lu_theobald_BG = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\theo_bg_all_lu.csv")
    lu_theobald_swa = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\Theo_swa_all_lu.csv")
    bg_x_wsa = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\misc_features\bg_intersect_wsa\blc_grp_x_swa.csv")
    bg_wsa_map = bg_x_wsa[['US_blck_12', 'WSA_AGIDF']]
    lu_theobald_BG = lu_theobald_BG.merge(bg_wsa_map, left_on='GISJOIN', right_on='US_blck_12', how='left')
    urban_id = list(range(211,216)) + list(range(221,224))
    density = [0.1, 1, 2.5, 10, 40,1,1,1,1]
    #density = [1, 1, 1, 1, 1]
    urban_col = ["VALUE_{}".format(id) for id in urban_id]
    total_bg_lu = lu_theobald_BG.groupby(by=['WSA_AGIDF']).sum()
    #total_bg_lu = total_bg_lu[urban_col].sum(axis=1)
    total_bg_lu['dua'] = 0
    for ii, uid in enumerate(urban_col):
        total_bg_lu['dua'] = total_bg_lu['dua'] + total_bg_lu[uid] * (1.0/density[ii])
        del(total_bg_lu[uid])

    lu_theobald_swa = lu_theobald_swa.groupby(by = ['WSA_AGIDF']).sum()
    lu_theobald_swa['dua'] = 0
    for ii, uid in enumerate(urban_col):
        lu_theobald_swa['dua'] = lu_theobald_swa['dua'] + lu_theobald_swa[uid] * (1.0/density[ii])
        del(lu_theobald_swa[uid])


    lu_theobald_swa['dua_bg'] = total_bg_lu['dua']
    lu_theobald_swa['ratio_lu'] = lu_theobald_swa['dua'] / lu_theobald_swa['dua_bg']
    lu_theobald_swa = lu_theobald_swa[['ratio_lu']]
    lu_theobald_swa.reset_index(inplace=True)
    train_db = train_db.merge(lu_theobald_swa, left_on='sys_id', right_on='WSA_AGIDF', how='left')
    del (train_db['WSA_AGIDF'])
    train_db['pop_urb'] = train_db['bg_pop_2010'] * train_db['ratio_lu']
    db2010 = train_db[train_db.Year == 2010].copy()
    db2010['ratio_2010'] = db2010['pop_urb'] / db2010['population']
    db2010 = db2010[['sys_id', 'ratio_2010']]

    train_db = train_db.merge(db2010, left_on='sys_id', right_on='sys_id', how='left')
    train_db['pop_enhanced'] = train_db['population'] * train_db['ratio_2010']
    train_db.to_csv(annual_training_file, index=False)


# =======================================
# enhance LU-pop using swud
# =======================================
if correct_enhance_pop_using_swud:
    swud_df_ = swud_df[swud_df['POP_SRV'] > 0]
    swud_df_ = swud_df_[swud_df_['TOT_WD_MGD'] > 0]
    swud_df_ = swud_df_[['WSA_AGIDF', 'YEAR', 'POP_SRV' ]]
    swud_df_['swud_year'] = swud_df_['YEAR']
    del( swud_df_['YEAR'])

    swud_df_ = swud_df_.drop_duplicates(
        subset=['WSA_AGIDF', 'swud_year'],
        keep='first').reset_index(drop=True)

    train_db = train_db.merge(swud_df_, left_on=['sys_id', 'Year'], right_on=['WSA_AGIDF', 'swud_year'], how='left')
    train_db['swud_pop_ratio'] = train_db['POP_SRV'] / train_db['pop_enhanced']

    rratio = train_db.groupby('sys_id').mean()['swud_pop_ratio']
    del(train_db['swud_pop_ratio'])
    del(train_db['swud_year'])
    del(train_db['POP_SRV'])
    del (train_db['WSA_AGIDF'])

    rratio = rratio.reset_index()
    train_db = train_db.merge(rratio, left_on='sys_id', right_on= 'sys_id', how='left')
    train_db.loc[train_db['swud_pop_ratio'].isna(),'swud_pop_ratio'] = 1.0
    train_db.loc[train_db['swud_pop_ratio']==0, 'swud_pop_ratio'] = 1.0
    train_db['LUpop_Swudpop'] = train_db['swud_pop_ratio'] * train_db['pop_enhanced']
    train_db.to_csv(annual_training_file, index=False)
    x = 1

# =======================================
# add Koppen Climate classification
# =======================================
if add_climate_classes:
    koppen_climate_df = koppen_climate_df[['WSA_AGIDF', 'MAJORITY']]
    koppen_climate_df['KG_climate_zone'] = koppen_climate_df['MAJORITY']
    del(koppen_climate_df['MAJORITY'])
    train_db = train_db.merge(koppen_climate_df, left_on='sys_id', right_on='WSA_AGIDF', how='left')
    del (koppen_climate_df['WSA_AGIDF'])
    train_db.to_csv(annual_training_file, index=False)

    xx = 1
xx = 1