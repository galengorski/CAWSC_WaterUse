import os, sys
import pandas as pd
import geopandas
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as pdd
import join_county_to_wsa
import time

db_root = r"C:\work\water_use\mldataset"
annual_training_file = os.path.join(
    db_root, r"ml\training\train_datasets\annual\wu_annual_training2.csv"
)
monthly_training_file = os.path.join(
    db_root,
    r"C:\work\water_use\mldataset\ml\training\train_datasets\Monthly\wu_monthly_training2.csv",
)
swud_df = pd.read_csv(
    os.path.join(
        db_root, r"ml\training\targets\monthly_annually\swud_v15.csv"
    ),
    encoding="cp1252",
)
swud_df16 = pd.read_csv(
    os.path.join(
        db_root,
        r"ml\training\targets\monthly_annually\SWUDS v16 WDs by FROM-sites and by WSA_AGG_ID_04282021.csv",
    ),
    encoding="cp1252",
)
tot_gb_pop = pd.read_csv(r"C:\work\water_use\blockgroup\gb_wsa_pop.csv")
wsa_shp = geopandas.read_file(
    r"C:\work\water_use\mldataset\gis\wsa\WSA_v3_02072021\WSA_v3_alb83_02072021.shp"
)
wsa_county = geopandas.read_file(
    r"C:\work\water_use\mldataset\gis\wsa_county_map\wsa_county_map.shp"
)
awuds_df = pd.read_csv(
    r"C:\work\water_use\mldataset\ml\training\misc_features\awuds_all_years.csv"
)
sell_buy_df = pd.read_csv(
    r"C:\work\water_use\mldataset\ml\training\misc_features\selling_buying\water_exchange_info.csv"
)
land_use_fn = r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\NWALT_landUse_summarized_for_ICITE_033021.xlsx"
koppen_climate_df = geopandas.read_file(
    r"C:\work\water_use\mldataset\gis\koppen_maps\wsa_climate_intersection_proj.shp"
)
agg_pop_df = pd.read_csv(
    os.path.join(db_root, r"ml\training\misc_features\WSA_V1_fromCheryl.csv")
)
ag_pop_year_df = pd.read_excel(
    os.path.join(
        db_root,
        r"ml\training\misc_features\V1_polys_with_water_service_06022021_for_GIS.xlsx",
    ),
    sheet_name="V1_1polyswWS",
)
places_pop_df = pd.read_csv(
    r"C:\work\water_use\mldataset\ml\training\misc_features\wsa_places_pop.csv"
)
wsa_local_pop = pd.read_excel(
    r"C:\work\water_use\mldataset\ml\training\misc_features\V1_polys_with_water_service_08072021_for_Ayman_popsrv_year.xlsx"
)

collect_all_annual_data = True
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
add_agg_pop_year = False
add_state_fips = False
compute_transeint_pop = False
add_places_pop = False  # do not use!
add_zillow_data = False
add_building_footprint = False
add_local_pop = True
add_census_place_data = True
update_annual_wu_from_15_to_16 = True


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
            wu_ = pd.read_csv(fn).set_index("Unnamed: 0")
            wu.append(wu_)

    wu = pdd.concat(wu)
    train_db = wu.compute()
    train_db.to_csv(annual_training_file, index=False)
    del wu
else:
    try:
        train_db = pd.read_csv(annual_training_file)
    except:
        print("Download Fail... try something else")
        train_db = pd.read_csv(annual_training_file, engine="python")


# =======================================
# Collect Monthly data
# =======================================
if collect_all_monthly_data:
    wu = []
    features_to_use = [
        "YEAR",
        "month",
        "wu_rate",
        "etr",
        "pr",
        "tmmn",
        "tmmx",
        "sys_id",
    ]
    fdb_root = os.path.join(db_root, r"ml\training\features")
    huc2_folders = os.listdir(fdb_root)
    for huc2_folder in huc2_folders:
        print(huc2_folder)
        fn = os.path.join(fdb_root, os.path.join(huc2_folder, "assemble"))
        fn = os.path.join(fn, "train_db_monthly_{}.csv".format(huc2_folder))
        if os.path.isfile(fn):
            wu_ = pd.read_csv(fn).set_index("Unnamed: 0")
            wu.append(wu_)

    train_db_monthly = pd.concat(wu)
    train_db_monthly = train_db_monthly[features_to_use]
    # train_db_monthly = wu.compute()

    XY = train_db[["sys_id", "LAT", "LONG"]]
    XY = XY.drop_duplicates(subset=["sys_id"])
    train_db_monthly = train_db_monthly.merge(
        XY, left_on="sys_id", right_on="sys_id", how="left"
    )
    train_db_monthly.to_csv(monthly_training_file, index=False)

# =======================================
# Add aggregate Population: this seems to
# be the best pop estimation
# =======================================
if add_agg_wsa_pop:
    agg_pop_df = agg_pop_df[["WSA_AGIDF", "TPOPSRV"]]
    train_db = train_db.merge(
        agg_pop_df, left_on="sys_id", right_on="WSA_AGIDF", how="left"
    )
    del train_db["WSA_AGIDF"]
    train_db.to_csv(annual_training_file, index=False)

if add_agg_pop_year:

    def extrat_year(txt):
        new_string = ""
        for s in txt:
            if s.isdigit():
                new_string = new_string + s
            else:
                new_string = new_string + " "

        years = new_string.strip().split()
        years = [int(y) for y in years]
        years = np.array(years)
        years = years[years > 1000]
        year = np.mean(years)
        if pd.isna(year):
            return 2013
        else:
            return int(year)

    # we assume if data is missing, we use 2010!. Chyrel will provide clean population with dates
    ag_pop_year_df["year"] = ag_pop_year_df["POP_METH"].apply(extrat_year)

# =======================================
# Correct Census Population Using Swud Pop
# =======================================
if compute_transeint_pop:
    agg_pop_df = agg_pop_df[["WSA_AGIDF", "TPOPSRV"]]
    agg_pop_df.set_index("WSA_AGIDF", inplace=True)
    agg_pop_df = agg_pop_df[~agg_pop_df.index.duplicated(keep="first")]

    ag_pop_year_df = ag_pop_year_df[["POP_SRV", "year", "WSA_AGIDF"]]
    ag_pop_year_df = ag_pop_year_df.groupby("WSA_AGIDF").agg(
        {"POP_SRV": "sum", "year": "mean"}
    )

    ###
    swud_df_ = swud_df[swud_df["POP_SRV"] > 0]
    swud_df_ = swud_df_[swud_df_["TOT_WD_MGD"] > 0]
    swud_df_ = swud_df_[["WSA_AGIDF", "YEAR", "POP_SRV"]]
    swud_df_["swud_year"] = swud_df_["YEAR"]
    del swud_df_["YEAR"]

    swud_df_ = swud_df_.drop_duplicates(
        subset=["WSA_AGIDF", "swud_year"], keep="first"
    ).reset_index(drop=True)

    train_db = train_db.merge(
        swud_df_,
        left_on=["sys_id", "Year"],
        right_on=["WSA_AGIDF", "swud_year"],
        how="left",
    )
    train_db["swud_pop_ratio"] = train_db["POP_SRV"] / train_db["pop_enhanced"]

    rratio = train_db.groupby("sys_id").mean()["swud_pop_ratio"]
    del train_db["swud_pop_ratio"]
    del train_db["swud_year"]
    del train_db["POP_SRV"]
    del train_db["WSA_AGIDF"]

    rratio = rratio.reset_index()
    train_db = train_db.merge(
        rratio, left_on="sys_id", right_on="sys_id", how="left"
    )
    train_db.loc[train_db["swud_pop_ratio"].isna(), "swud_pop_ratio"] = 1.0
    train_db.loc[train_db["swud_pop_ratio"] == 0, "swud_pop_ratio"] = 1.0
    train_db["LUpop_Swudpop"] = (
        train_db["swud_pop_ratio"] * train_db["pop_enhanced"]
    )
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Correct Census Population Using Swud Pop
# =======================================
if correct_census_pop_using_swud:
    train_db["pc_swud"] = train_db["wu_rate"] / train_db["swud_pop"]
    train_db["pc_tract_data"] = train_db["wu_rate"] / train_db["population"]
    mask1 = train_db["swud_pop"] > 0
    mask2 = (train_db["pc_swud"] <= 200) & (train_db["pc_swud"] >= 40)
    mask3 = (train_db["pc_tract_data"] > 200) | (
        train_db["pc_tract_data"] < 40
    )
    mask = mask1 & mask2 & mask3
    sys_ids = train_db.loc[mask, "sys_id"].unique()
    train_db["pop_swud_corrected"] = train_db["population"]
    train_db["swud_corr_factor"] = 1.0
    for i, sys_id in enumerate(sys_ids):
        print(i)
        print(i / len(sys_ids))
        mask_sys_id = train_db["sys_id"] == sys_id

        curr_df = train_db[mask & mask_sys_id]
        corr_factor = curr_df["swud_pop"].mean() / curr_df["population"].mean()
        if not (corr_factor > 0):
            corr_factor = 1.0

        train_db.loc[mask_sys_id, "pop_swud_corrected"] = (
            corr_factor * train_db.loc[mask_sys_id, "population"].values
        )
        train_db.loc[mask_sys_id, "swud_corr_factor"] = corr_factor

    train_db["pc_swud_corrected"] = (
        train_db["wu_rate"] / train_db["pop_swud_corrected"]
    )
    train_db["pc_gb_data"] = train_db["wu_rate"] / train_db["small_gb_pop"]

    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Fill gaps in LAT/LON data
# =======================================
if fill_lat_lon_gaps:
    sys_ids = wsa_shp["WSA_AGIDF"].unique()
    mask_lat = train_db["LAT"].isna()
    mask_long = train_db["LONG"].isna()
    mask_xy = (mask_lat) | (mask_long)

    sys_no_xy = train_db[mask_xy]["sys_id"].unique()
    counter = 0
    for i, sys_id in enumerate(sys_no_xy):
        if sys_id[0].isdigit():
            if len(sys_id) < 9:
                maskcc = train_db["sys_id"] == sys_id
                sys_id = "0" + sys_id
                train_db.loc[maskcc, "sys_id"] = sys_id

        if not (sys_id in sys_ids):
            print("No XY info")
            continue

        counter = counter + 1
        print((i * 1.0) / len(sys_no_xy))
        mask_id = wsa_shp["WSA_AGIDF"] == sys_id
        Lat = wsa_shp.loc[mask_id, "LAT"].values[0]
        Long = wsa_shp.loc[mask_id, "LONG"].values[0]
        train_db.loc[train_db["sys_id"] == sys_id, "LAT"] = Lat
        train_db.loc[train_db["sys_id"] == sys_id, "LONG"] = Long
    mask_lat = train_db["LAT"].isna()
    mask_long = train_db["LONG"].isna()
    mask_xy2 = (mask_lat) | (mask_long)
    fraction_fixed = (1 - 1.0 * np.sum(mask_xy2)) / (1.0 * np.sum(mask_xy))
    print("Fraction of LAT/LONG fixed is {}".format(fraction_fixed * 100.0))

    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Try to correct WGB with TGB
# =======================================

if correct_wGB_population:
    train_db["pop_swud_gb_correction"] = train_db["pop_swud_corrected"]
    mask_extreme_pc = (train_db["pc_swud_corrected"] > 700) | (
        (train_db["pc_swud_corrected"] < 20)
    )
    mask2 = (train_db["pc_gb_data"] < 700) & (train_db["pc_gb_data"] > 20)
    mask = mask_extreme_pc & mask2
    train_db.loc[mask, "pop_swud_gb_correction"] = train_db.loc[
        mask, "small_gb_pop"
    ]
    train_db["pc_swud_gb_corrected"] = (
        train_db["wu_rate"] / train_db["pop_swud_gb_correction"]
    )
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add AWUDS data
# =======================================
if add_AWUS_data:
    awuds_df = awuds_df[awuds_df["YEAR"] > 2000]
    awuds_df["STATECODE"] = awuds_df["STATECODE"].astype(str).str.zfill(2)
    awuds_df["COUNTYCODE"] = awuds_df["COUNTYCODE"].astype(str).str.zfill(3)
    awuds_df["county_id"] = (
        awuds_df["STATECODE"].astype(str) + awuds_df["COUNTYCODE"]
    )

    county_id_df = awuds_df["county_id"]
    county_id_shp = wsa_county["GEOID"]
    sys_ids = train_db["sys_id"].unique()
    train_db["county_id"] = ""
    train_db["awuds_totw_cnt"] = 0
    train_db["awuds_dom_cnt"] = 0
    train_db["awuds_dom_cnt"] = 0
    train_db["awuds_pop_cnt"] = 0
    for i, sys_id in enumerate(sys_ids):
        print((1.0 * i) / len(sys_ids))
        mask = wsa_county["WSA_AGIDF"] == sys_id
        try:
            curr_county = wsa_county.loc[mask, "GEOID"].values[0]
        except:
            print("No data")
            continue

        mask2 = awuds_df["county_id"] == curr_county
        tot_withdrawal = awuds_df.loc[mask2, "PS-WTotl"].mean() * 1e6
        domestic_del = awuds_df.loc[mask2, "PS-DelDO"].mean() * 1e6
        ps_pop = awuds_df.loc[mask2, "PS-TOPop"].mean() * 1e3

        mask3 = train_db["sys_id"] == sys_id
        train_db.loc[mask3, "awuds_totw_cnt"] = tot_withdrawal
        train_db.loc[mask3, "awuds_dom_cnt"] = domestic_del
        train_db.loc[mask3, "awuds_pop_cnt"] = ps_pop
        train_db.loc[mask3, "county_id"] = curr_county
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add Water Exchange data
# =======================================
if add_water_exchange_data:

    sell_buy_df["WSA_AGIDF"] = sell_buy_df["WSA_AGIDF"].str.replace("'", "")
    sell_buy_df = sell_buy_df.drop_duplicates(subset=["WSA_AGIDF"])
    sell_buy_df = sell_buy_df[["WSA_AGIDF", "Ecode"]]
    train_db = train_db.merge(
        sell_buy_df, left_on="sys_id", right_on="WSA_AGIDF", how="left"
    )
    categories = train_db["Ecode"].unique()
    for i, categ in enumerate(categories):
        mask = train_db["Ecode"] == categ
        train_db.loc[mask, "Ecode_num"] = i
    del train_db["WSA_AGIDF"]
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add total number of houses
# =======================================
if compute_n_houses:
    house_age = train_db[
        [
            "h_age_newer_2005",
            "h_age_2000_2004",
            "h_age_1990_1999",
            "h_age_1980_1989",
            "h_age_1970_1979",
            "h_age_1960_1969",
            "h_age_1950_1959",
            "h_age_1940_1949",
            "h_age_older_1939",
        ]
    ]
    train_db["n_houses"] = house_age.sum(axis=1)
    train_db["pop_house_ratio"] = train_db["population"] / train_db["n_houses"]
    train_db["family_size"] = train_db["population"] / train_db["households2"]
    train_db.to_csv(annual_training_file, index=False)


# =======================================
# fraction of income and age
# =======================================
if use_fractions_of_income_age:
    income_info = (
        "income_lt_10k	income_10K_15k	income_15k_20k	income_20k_25k	income_25k_30k	income_30k_35k	income_35k_40k"
        "	income_40k_45k	income_45k_50k	income_50k_60k	income_60k_75k	income_75k_100k	income_100k_125k	income_125k_150k"
        "	income_150k_200k	income_gt_200k"
    )
    income_feat = income_info.split()
    sum_income = train_db[income_feat].sum(axis=1)
    for feat in income_feat:
        train_db[feat] = train_db[feat] / sum_income

    hs_age_feats = [
        "h_age_newer_2005",
        "h_age_2000_2004",
        "h_age_1990_1999",
        "h_age_1980_1989",
        "h_age_1970_1979",
        "h_age_1960_1969",
        "h_age_1950_1959",
        "h_age_1940_1949",
        "h_age_older_1939",
    ]
    for feat in hs_age_feats:
        train_db[feat] = train_db[feat] / train_db["n_houses"]
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add economic county data
# =======================================
if add_county_data:
    fid = open(
        r"C:\work\water_use\mldataset\ml\training\misc_features\parameters_eco.txt",
        "r",
    )
    content = fid.readlines()
    eco_info = {}
    for line in content:
        if line.strip()[0] == "#":
            continue
        field_key = line.strip().split(":")
        eco_info[field_key[0]] = field_key[1]
    fid.close()

    fid = open(
        r"C:\work\water_use\mldataset\ml\training\misc_features\parameters.txt",
        "w",
    )
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
        if field in ["fips"]:
            continue
        for ff in fields:
            ff2 = ff.lower()
            if field in ff2:
                break
            else:
                continue
        train_db[eco_info[ff]] = train_db[ff2]
        del train_db[ff2]

    sub_fields = ["income_", "n_jobs_", "indus_", "rur_urb_", "unemployment_"]
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
            del train_db[ff]
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add state fips from county data
# =======================================
if add_state_fips:
    train_db["state_id"] = (train_db["county_id"] / 1000.0).astype(int)
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add economic county data 2 - number of establishments
# =======================================
if Add_Manufacturing_establishments:
    manu_folder = r"C:\work\water_use\mldataset\ml\training\misc_features\ml_county_to_sa\county_data\US_County"
    fn2012 = os.path.join(
        manu_folder, r"US_MFG_SIC31-33_County_2012_20200911.csv"
    )
    fn2017 = os.path.join(
        manu_folder, r"US_MFG_SIC31-33_County_2017_20200911.csv"
    )
    train_db["county_ids"] = train_db["county_id"].astype(str).str.zfill(5)

    df = pd.read_csv(fn2012)
    df["countysss"] = df["id"].str[-5:]
    df = df[df["Meaning of 2012 NAICS code"] == "Manufacturing"]
    df = df[["countysss", "Number of establishments"]]
    df["Num_establishments_2012"] = df["Number of establishments"]
    del df["Number of establishments"]
    train_db = train_db.merge(
        df, left_on="county_ids", right_on="countysss", how="left"
    )
    del train_db["countysss"]

    df = pd.read_csv(fn2017)
    df["countysss"] = df["id"].str[-5:]
    df = df[df["Meaning of NAICS code"] == "Manufacturing"]
    df = df[["countysss", "Number of establishments"]]
    df["Num_establishments_2017"] = df[["Number of establishments"]]
    del df["Number of establishments"]
    train_db = train_db.merge(
        df, left_on="county_ids", right_on="countysss", how="left"
    )

    del train_db["county_ids"]
    del train_db["countysss"]
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
            if col in ["WSA_AGIDF"]:
                continue
            new_col = col + "_{}".format(year)
            df_[new_col] = df_[col]
            del df_[col]
        train_db = train_db.merge(
            df_, left_on="sys_id", right_on="WSA_AGIDF", how="left"
        )
        del train_db["WSA_AGIDF"]
    # Theobald
    df_ = pd.read_excel(
        r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\ICITE_Theobald_zonal_stats_033121.xlsx",
        sheet_name="finalSUMbyICITE",
    )
    train_db = train_db.merge(
        df_, left_on="sys_id", right_on="WSA_AGIDF", how="left"
    )
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Add area
# =======================================
if add_area:
    area_shp = wsa_shp[["WSA_SQKM", "WSA_AGIDF"]]
    train_db = train_db.merge(
        area_shp, left_on="sys_id", right_on="WSA_AGIDF", how="left"
    )
    del train_db["WSA_AGIDF"]
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# Enhance_population usning Landa use
# =======================================
if scale_pop_using_land_use:
    # usgs wall to wall
    lu_usgs = pd.read_excel(
        r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\NWALT_ZONAL_STATS_Subset_BG_2010_data_042321.xlsx",
        sheet_name=r"Pivot_table_LU2012",
    )
    lu_theobald = pd.read_excel(
        r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\ICITE_Theobalt_zonal_stats_BLOCKGROUP_043021.xlsx",
        sheet_name="finalSUMbyICITE",
    )

    # bg_select_wsa: is the set of BG's that intersect with WSA
    bg_select_wsa = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\misc_features\bg_intersect_wsa\bg_select_wsa.csv"
    )
    # bg_x_wsa: is the geometrical instersection of SWA and BG maps
    bg_x_wsa = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\misc_features\bg_intersect_wsa\blc_grp_x_swa.csv"
    )

    # C:\work\water_use\mldataset\gis\subset_BG_2010.shp
    bg_wsa_map = bg_x_wsa[["US_blck_12", "WSA_AGIDF"]]
    pop_tot_bg_map = bg_select_wsa.merge(
        bg_wsa_map, left_on="GISJOIN", right_on="US_blck_12", how="left"
    )
    pop_tot_bg = pop_tot_bg_map[["GISJOIN", "WSA_AGIDF", "H7V001"]]
    pop_tot_bg = pop_tot_bg.groupby("WSA_AGIDF").sum()

    # usgs land use corrections
    lu_usgs["Domestic_Total"] = (
        lu_usgs["Sum of Developed"] + lu_usgs["Sum of Semi-Developed"]
    )
    lu_usgs = lu_usgs.merge(
        bg_wsa_map, left_on="Row Labels", right_on="US_blck_12", how="left"
    )
    lu_domestic_total = lu_usgs.groupby(by="WSA_AGIDF").sum()
    lu_swa = train_db[["WSA_AGIDF", "Domestic_Total_2012"]]
    lu_swa = lu_swa.set_index("WSA_AGIDF")
    lu_swa["LU_BG"] = lu_domestic_total["Domestic_Total"]
    lu_swa["bg_usgs_correction_factor"] = (
        lu_swa["Domestic_Total_2012"] / lu_swa["LU_BG"]
    )

    # theobald land use corrections
    lu_theobald = lu_theobald.merge(
        bg_wsa_map, left_on="GISJOIN", right_on="US_blck_12", how="left"
    )
    lu_domestic_total_theo = lu_theobald.groupby(by="WSA_AGIDF").sum()
    lu_swa_theo = train_db[["WSA_AGIDF", "Domestic"]]
    lu_swa_theo = lu_swa_theo.set_index("WSA_AGIDF")
    lu_swa_theo["LU_BG"] = (
        lu_domestic_total_theo["Domestic"]
        + lu_domestic_total_theo["Urban_Misc"]
    )
    lu_swa_theo["bg_theo_correction_factor"] = (
        lu_swa_theo["Domestic"] / lu_swa_theo["LU_BG"]
    )

    train_db = train_db.merge(
        pop_tot_bg, left_on="sys_id", right_on="WSA_AGIDF", how="left"
    )
    train_db["bg_pop_2010"] = train_db["H7V001"]
    del train_db["H7V001"]
    del train_db["WSA_AGIDF"]
    del lu_usgs
    del lu_theobald
    del bg_select_wsa
    del bg_x_wsa
    del bg_wsa_map

    lu_swa.reset_index(inplace=True)
    lu_swa = lu_swa[["WSA_AGIDF", "bg_usgs_correction_factor"]]
    lu_swa_theo.reset_index(inplace=True)
    lu_swa_theo = lu_swa_theo[["WSA_AGIDF", "bg_theo_correction_factor"]]
    lu_swa_theo.drop_duplicates(subset=["WSA_AGIDF"], inplace=True)
    lu_swa.drop_duplicates(subset=["WSA_AGIDF"], inplace=True)

    train_db = train_db.merge(
        lu_swa[["WSA_AGIDF", "bg_usgs_correction_factor"]],
        left_on="sys_id",
        right_on="WSA_AGIDF",
        how="left",
    )

    train_db = train_db.merge(
        lu_swa_theo[["WSA_AGIDF", "bg_theo_correction_factor"]],
        left_on="sys_id",
        right_on="WSA_AGIDF",
        how="left",
    )

    cols = train_db.columns
    for col in cols:
        if "WSA_AGIDF" in col:
            print(col)
            del train_db[col]

    train_db.to_csv(annual_training_file, index=False)

# =======================================
# More Land Use analysis
# =======================================
if More_land_use:

    lu_theobald_BG = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\theo_bg_all_lu.csv"
    )
    lu_theobald_swa = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\misc_features\land_use\Theo_swa_all_lu.csv"
    )
    bg_x_wsa = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\misc_features\bg_intersect_wsa\blc_grp_x_swa.csv"
    )
    bg_wsa_map = bg_x_wsa[["US_blck_12", "WSA_AGIDF"]]
    lu_theobald_BG = lu_theobald_BG.merge(
        bg_wsa_map, left_on="GISJOIN", right_on="US_blck_12", how="left"
    )
    urban_id = list(range(211, 216)) + list(range(221, 224))
    density = [0.1, 1, 2.5, 10, 40, 1, 1, 1, 1]
    # density = [1, 1, 1, 1, 1]
    urban_col = ["VALUE_{}".format(id) for id in urban_id]
    total_bg_lu = lu_theobald_BG.groupby(by=["WSA_AGIDF"]).sum()
    # total_bg_lu = total_bg_lu[urban_col].sum(axis=1)
    total_bg_lu["dua"] = 0
    for ii, uid in enumerate(urban_col):
        total_bg_lu["dua"] = total_bg_lu["dua"] + total_bg_lu[uid] * (
            1.0 / density[ii]
        )
        del total_bg_lu[uid]

    lu_theobald_swa = lu_theobald_swa.groupby(by=["WSA_AGIDF"]).sum()
    lu_theobald_swa["dua"] = 0
    for ii, uid in enumerate(urban_col):
        lu_theobald_swa["dua"] = lu_theobald_swa["dua"] + lu_theobald_swa[
            uid
        ] * (1.0 / density[ii])
        del lu_theobald_swa[uid]

    lu_theobald_swa["dua_bg"] = total_bg_lu["dua"]
    lu_theobald_swa["ratio_lu"] = (
        lu_theobald_swa["dua"] / lu_theobald_swa["dua_bg"]
    )
    lu_theobald_swa = lu_theobald_swa[["ratio_lu"]]
    lu_theobald_swa.reset_index(inplace=True)
    train_db = train_db.merge(
        lu_theobald_swa, left_on="sys_id", right_on="WSA_AGIDF", how="left"
    )
    del train_db["WSA_AGIDF"]
    train_db["pop_urb"] = train_db["bg_pop_2010"] * train_db["ratio_lu"]
    db2010 = train_db[train_db.Year == 2010].copy()
    db2010["ratio_2010"] = db2010["pop_urb"] / db2010["population"]
    db2010 = db2010[["sys_id", "ratio_2010"]]

    train_db = train_db.merge(
        db2010, left_on="sys_id", right_on="sys_id", how="left"
    )
    train_db["pop_enhanced"] = train_db["population"] * train_db["ratio_2010"]
    train_db.to_csv(annual_training_file, index=False)


# =======================================
# enhance LU-pop using swud
# =======================================
if correct_enhance_pop_using_swud:
    swud_df_ = swud_df[swud_df["POP_SRV"] > 0]
    swud_df_ = swud_df_[swud_df_["TOT_WD_MGD"] > 0]
    swud_df_ = swud_df_[["WSA_AGIDF", "YEAR", "POP_SRV"]]
    swud_df_["swud_year"] = swud_df_["YEAR"]
    del swud_df_["YEAR"]

    swud_df_ = swud_df_.drop_duplicates(
        subset=["WSA_AGIDF", "swud_year"], keep="first"
    ).reset_index(drop=True)

    train_db = train_db.merge(
        swud_df_,
        left_on=["sys_id", "Year"],
        right_on=["WSA_AGIDF", "swud_year"],
        how="left",
    )
    train_db["swud_pop_ratio"] = train_db["POP_SRV"] / train_db["pop_enhanced"]

    rratio = train_db.groupby("sys_id").mean()["swud_pop_ratio"]
    del train_db["swud_pop_ratio"]
    del train_db["swud_year"]
    del train_db["POP_SRV"]
    del train_db["WSA_AGIDF"]

    rratio = rratio.reset_index()
    train_db = train_db.merge(
        rratio, left_on="sys_id", right_on="sys_id", how="left"
    )
    train_db.loc[train_db["swud_pop_ratio"].isna(), "swud_pop_ratio"] = 1.0
    train_db.loc[train_db["swud_pop_ratio"] == 0, "swud_pop_ratio"] = 1.0
    train_db["LUpop_Swudpop"] = (
        train_db["swud_pop_ratio"] * train_db["pop_enhanced"]
    )
    train_db.to_csv(annual_training_file, index=False)
    x = 1

# =======================================
# add Koppen Climate classification
# =======================================
if add_climate_classes:
    # there is error in some wsa where 0 is missing fron train_db name
    db_ = set(train_db["sys_id"].unique())
    cl_ = set(koppen_climate_df["WSA_AGIDF"].unique())
    sys_id_with_issues = db_.difference(cl_)
    for ss in sys_id_with_issues:
        ss_ = "0" + ss
        if ss_ in cl_:
            train_db.loc[train_db["sys_id"] == ss, "sys_id"] = ss_

    idx = (
        koppen_climate_df.groupby(["WSA_AGIDF"])["area_calc"].transform(max)
        == koppen_climate_df["area_calc"]
    )
    koppen_climate_df = koppen_climate_df[idx][["WSA_AGIDF", "gridcode"]]
    train_db = train_db.merge(
        koppen_climate_df, left_on="sys_id", right_on="WSA_AGIDF", how="left"
    )
    train_db["KG_climate_zone"] = train_db["gridcode"]
    del train_db["WSA_AGIDF"]
    del train_db["gridcode"]
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# add Zillow
# =======================================
if add_zillow_data:
    zillow_folder = (
        r"C:\work\water_use\mldataset\ml\training\misc_features\zillow"
    )
    zillow_files = ["zillow_wsa_2018", "zillow_wsa_2019", "zillow_wsa_2021"]

    all_zillow = []
    for file in zillow_files:
        fn = os.path.join(zillow_folder, file + ".csv")
        zdf = pd.read_csv(fn)
        zdf["zill_nhouse"] = zdf["NoOfHouses_sum"]
        cols = [
            "WSA_AGIDF",
            "zill_nhouse",
            "LotSizeSquareFeet_sum",
            "YearBuilt_mean",
            "BuildingAreaSqFt_sum",
            "TaxAmount_mean",
            "NoOfStories_mean",
        ]
        zdf = zdf[cols]

        for c in cols:
            if "WSA_AGIDF" in c:
                continue

            zdf.loc[zdf[c] < 1, c] = np.nan
        all_zillow.append(zdf.copy())

    all_zillow = pd.concat(all_zillow)
    all_zillow = all_zillow.groupby(by=["WSA_AGIDF"]).median()
    all_zillow.reset_index(inplace=True)
    train_db = train_db.merge(
        all_zillow, how="left", right_on="WSA_AGIDF", left_on="sys_id"
    )
    del train_db["WSA_AGIDF"]
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# add Building Footprint
# =======================================
if add_building_footprint:
    fn_blg_footprint = r"C:\work\water_use\mldataset\ml\training\misc_features\buildingfootprints\us_building_footprint.csv"
    blg_ft_df = pd.read_csv(fn_blg_footprint)
    blg_ft_df = blg_ft_df[["WSA_AGIDF", "count", "gt_2median"]].copy()
    blg_ft_df["bdg_ftp_count"] = blg_ft_df["count"]
    del blg_ft_df["count"]
    blg_ft_df["bdg_ftp_gt2median"] = blg_ft_df["gt_2median"]
    del blg_ft_df["gt_2median"]
    train_db = train_db.merge(
        blg_ft_df, how="left", right_on="WSA_AGIDF", left_on="sys_id"
    )
    del train_db["WSA_AGIDF"]
    train_db.to_csv(annual_training_file, index=False)

# =======================================
# add local pop
# =======================================
if add_local_pop:
    wsa_local_pop["loc_year"] = wsa_local_pop["POP_METH"].str[-4:]
    wsa_local_pop["loc_year"] = wsa_local_pop["loc_year"].astype(int)
    wsa_local_pop.loc[wsa_local_pop["loc_year"] == 19, "loc_year"] = 2019
    wsa_local_pop = wsa_local_pop[["WSA_AGIDF", "POP_SRV", "loc_year"]]
    train_db = train_db.merge(
        wsa_local_pop, how="left", left_on="sys_id", right_on="WSA_AGIDF"
    )
    del train_db["WSA_AGIDF"]


# =======================================
# add and interpolate places population
# =======================================
# I tried to download data from excel sheet. This is not used any more
if add_places_pop:
    del places_pop_df["Unnamed: 0"]
    places_pop_df["year2"] = places_pop_df["year"]
    del places_pop_df["year"]
    places_pop_df["plc_population"] = places_pop_df["popPlcwsa"]
    del places_pop_df["popPlcwsa"]
    global counter11
    counter11 = 0
    cols = [
        "POPESTIMATE2010",
        "POPESTIMATE2011",
        "POPESTIMATE2012",
        "POPESTIMATE2013",
        "POPESTIMATE2014",
        "POPESTIMATE2015",
        "POPESTIMATE2016",
        "POPESTIMATE2017",
        "POPESTIMATE2018",
        "POPESTIMATE2019",
        "POPESTIMATE2020",
    ]

    def compute_place_census(_df):

        global counter11
        counter11 = counter11 + 1
        print(100.0 * counter11 / 19123.0)
        cols = [
            "POPESTIMATE2010",
            "POPESTIMATE2011",
            "POPESTIMATE2012",
            "POPESTIMATE2013",
            "POPESTIMATE2014",
            "POPESTIMATE2015",
            "POPESTIMATE2016",
            "POPESTIMATE2017",
            "POPESTIMATE2018",
            "POPESTIMATE2019",
            "POPESTIMATE2020",
        ]

        df_out = pd.DataFrame(columns=["year", "pop"])
        df_out["year"] = np.arange(2000, 2021)
        df_out.set_index("year", inplace=True)
        df_out["pop"] = 0

        tract_pop = _df[["Year", "population"]]
        tract_pop = tract_pop[tract_pop["population"] > 0]
        tract_pop.set_index("Year", inplace=True)

        tract_pop = tract_pop[~tract_pop.index.duplicated(keep="first")]
        df_out["tract_pop"] = tract_pop["population"]

        df_out["year"] = df_out.index

        # constraint pop growth
        max_pop_growth = 5 / 100
        df_out["diff"] = df_out["tract_pop"].diff().values[1:].tolist() + [0]
        mask = abs(df_out["diff"]) / df_out["tract_pop"] < max_pop_growth
        ref_df = df_out[mask]

        ref_df = ref_df[ref_df.index < 2020]
        ref_df = ref_df[
            (ref_df["diff"] < ref_df["diff"].quantile(0.80))
            & (ref_df["diff"] > ref_df["diff"].quantile(0.2))
        ]
        if len(ref_df) == 0:
            ref_df = df_out[mask]

        pop_ref = ref_df["tract_pop"].mean()
        mean_increase = ref_df["diff"].mean()
        year_ref = np.mean(ref_df.index)
        new_pop = (pop_ref - (df_out["year"] - year_ref) * mean_increase)[
            ~mask
        ]
        new_pop.name = "pop"
        new_pop = pd.DataFrame(new_pop)
        mask2 = df_out["year"].isin(new_pop.index)
        df_out["ppp"] = new_pop["pop"]
        df_out.loc[mask2, "tract_pop"] = df_out[mask2]["ppp"].values

        # add annual place data
        yrs = np.arange(2010, 2021)
        pop = _df[cols].values[0, :]

        d = pd.DataFrame()
        d["year"] = yrs
        d["pop"] = pop
        d.set_index("year", inplace=True)
        d = d[d["pop"] > 0]

        if len(d) > 0:  # use transeint data
            df_out["pop"] = d["pop"]
            mask = df_out["pop"] > 0
            yr = int(df_out[df_out["pop"] > 0].index[0])
            tract_pyr = df_out.loc[yr]["tract_pop"]
            df_out["ratio"] = df_out["tract_pop"] / tract_pyr
            mask3 = df_out["pop"].isna()
            df_out.loc[mask3, "pop"] = (
                df_out[mask3]["ratio"].values * df_out.loc[yr]["pop"]
            )
            df_out["pop"] = df_out["pop"]
            df_out = df_out[["pop"]]
        else:
            if _df["plc_population"].values[0] > 0:
                tract_p2020 = df_out.loc[2020]["tract_pop"]
                df_out["ratio"] = df_out["tract_pop"] / tract_p2020
                df_out["pop"] = (
                    _df["plc_population"].values[0] * df_out["ratio"]
                )

                df_out["pop"] = df_out["pop"]
                df_out = df_out[["pop"]]

            else:
                if _df["POP_SRV"].values[0] > 0:
                    yr = int(_df["year2"].values[0])
                    tract_pyr = df_out.loc[yr]["tract_pop"]
                    df_out["ratio"] = df_out["tract_pop"] / tract_pyr
                    df_out["pop"] = _df["POP_SRV"].values[0] * df_out["ratio"]
                    df_out["pop"] = df_out["pop"]
                    df_out = df_out[["pop"]]
                else:
                    df_out["pop"] = df_out["tract_pop"]
                    df_out["pop"] = df_out["pop"]
                    df_out = df_out[["pop"]]

        df_out["sys_id"] = _df["sys_id"].values[0]
        # df_out.reset_index(inplace=True)
        # if len(train_db[train_db['population']>0]) != len(train_db):
        #     cc = 1
        return df_out

    train_db = train_db.merge(
        places_pop_df, left_on="sys_id", right_on="WSA_AGIDF", how="left"
    )
    del train_db["WSA_AGIDF"]
    if "pop" in train_db.columns:
        del train_db["pop"]

    Parallel = False

    if Parallel:
        start = time.time()
        from dask import dataframe as dd

        sd = dd.from_pandas(train_db, npartitions=8)
        xx = sd.groupby(["sys_id"], group_keys=False).apply(
            compute_place_census
        )
        yy = xx.compute(scheduler="processes")
        endd = time.time()
    else:
        start = time.time()
        yy = train_db.groupby(["sys_id"], group_keys=False).apply(
            compute_place_census
        )
        endd = time.time()
    print(" Running Time is {}".format(endd - start))
    yy.reset_index(inplace=True)

    train_db = train_db.merge(
        yy, left_on=["sys_id", "Year"], right_on=["sys_id", "year"], how="left"
    )
    for col in cols:
        del train_db[col]
    train_db.to_csv(annual_training_file, index=False)

# =========================================
# Get "places" census data  from collector
# =========================================
if add_census_place_data:
    if 0:
        census_places_folder = (
            r"C:\work\water_use\data\WSA_places\WSA_places\WSA_places"
        )
        folders = os.listdir(census_places_folder)
        all_dfs = []
        for folder in folders:
            print("----->" + folder)
            ws = os.path.join(census_places_folder, folder)
            files = os.listdir(ws)
            for file in files:
                print(file)
                sys_id = file.split("_")[2]
                df_ = pd.read_csv(os.path.join(ws, file))
                df_["sys_id"] = sys_id.upper()
                all_dfs.append(df_.copy())
        xx = 1
        all_dfs = pd.concat(all_dfs)
        place_features = all_dfs.columns
        for c_ in all_dfs.columns:
            col = "plc_" + c_
            all_dfs[col] = all_dfs[c_]
            del all_dfs[c_]
        train_db = train_db.merge(
            all_dfs,
            left_on=["sys_id", "Year"],
            right_on=["plc_sys_id", "plc_year"],
            how="left",
        )
    else:
        train_db_o = pd.read_csv(annual_training_file + ".2")
        train_db_o = train_db_o[~train_db_o["plc_sys_id"].isna()]
        plc_columns = [c for c in train_db_o.columns if "plc_" in c]

        train_db = train_db.merge(
            train_db_o[plc_columns],
            how="left",
            right_on=["plc_year", "plc_sys_id"],
            left_on=["Year", "sys_id"],
        )

    counter11 = 0

    def compute_place_census(_df):

        global counter11
        counter11 = counter11 + 1
        print(100.0 * counter11 / 19123.0)

        _dfc = _df.groupby(by="Year").mean()
        _dfc.reset_index(inplace=True)
        df_out = pd.DataFrame(columns=["year", "pop"])
        df_out["year"] = np.arange(2000, 2021)
        df_out.set_index("year", inplace=True)
        df_out["pop"] = 0

        tract_pop = _dfc[["Year", "population"]]
        tract_pop = tract_pop[tract_pop["population"] > 0]
        tract_pop.set_index("Year", inplace=True)

        tract_pop = tract_pop[~tract_pop.index.duplicated(keep="first")]
        df_out["tract_pop"] = tract_pop["population"]

        df_out["year"] = df_out.index

        # constraint pop growth
        max_pop_growth = 5 / 100
        df_out["diff"] = df_out["tract_pop"].diff().values[1:].tolist() + [0]
        mask = abs(df_out["diff"]) / df_out["tract_pop"] < max_pop_growth
        ref_df = df_out[mask]

        ref_df = ref_df[ref_df.index < 2020]
        ref_df = ref_df[
            (ref_df["diff"] < ref_df["diff"].quantile(0.80))
            & (ref_df["diff"] > ref_df["diff"].quantile(0.2))
        ]
        if len(ref_df) == 0:
            ref_df = df_out[mask]

        pop_ref = ref_df["tract_pop"].mean()
        mean_increase = ref_df["diff"].mean()
        year_ref = np.mean(ref_df.index)
        new_pop = (pop_ref - (df_out["year"] - year_ref) * mean_increase)[
            ~mask
        ]
        new_pop.name = "pop"
        new_pop = pd.DataFrame(new_pop)
        mask2 = df_out["year"].isin(new_pop.index)
        df_out["ppp"] = new_pop["pop"]
        df_out.loc[mask2, "tract_pop"] = df_out[mask2]["ppp"].values

        df_out["cdc_pop_0"] = tract_pop["population"]
        df_out.drop(labels=["diff", "ppp"], axis=1, inplace=True)

        # add annual place data
        cols = [col for col in _df.columns if "plc_" in col]
        df_plc = _dfc[["Year", "plc_year", "plc_population"]]
        df_plc.sort_values(by="Year", inplace=True)
        df_plc.set_index("Year", inplace=True)
        if np.any(df_plc["plc_population"] > 0):
            # place data is available
            df_out["plc_pop_interpolated"] = df_plc[
                "plc_population"
            ].interpolate()
        else:
            # no place data is available
            df_out["plc_pop_interpolated"] = np.nan

        # local usgs data
        if np.any(_dfc["TPOPSRV"] > 0):  # local usgs data
            if _dfc["loc_year"].values[0] > 0:
                yr = _dfc["loc_year"].mean()
            else:
                yr = 2010
            pop_srv = _dfc["TPOPSRV"].mean()
            ref_p = df_out.loc[df_out["year"] == int(yr), "tract_pop"].values[
                0
            ]

            df_out["tpop_srv"] = (pop_srv / ref_p) * df_out["tract_pop"]
        df_out["zillow_nh"] = _dfc["zill_nhouse"].values[0]
        df_out["bdg_ftp_count"] = _dfc["bdg_ftp_count"].values[0]
        wu_df = _dfc[["Year", "wu_rate"]].copy()
        wu_df.set_index(["Year"], inplace=True)
        wu_df[wu_df.index.duplicated()]
        df_out["wu_rate"] = wu_df["wu_rate"]

        df_out["sys_id"] = _df["sys_id"].values[0]
        del df_out["year"]
        df_out.reset_index(inplace=True)
        return df_out

    start = time.time()

    yrr16 = swud_df16["POP_METH"].str[-4:]
    yrr16[yrr16.isna()] = "-1"

    yrr16[yrr16.str.isalpha()] = "-1"
    yrr16 = yrr16.astype(int)
    yrr16[yrr16 < 0] = 2010
    swud_df16["year16"] = yrr16

    pop_info = train_db.groupby(["sys_id"], group_keys=False).apply(
        compute_place_census
    )
    endd = time.time()
    vv = 1
xx = 1
