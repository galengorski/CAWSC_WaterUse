import os, sys
import pandas as pd


house_age_df = pd.read_csv(
    r"C:\work\water_use\mldataset\ml\training\misc_features\ts_census\hs_age\nhgis0009_ds176_20105_blck_grp.csv",
    encoding="cp1252",
)
fmaily_hoshod = pd.read_csv(
    r"C:\work\water_use\mldataset\ml\training\misc_features\ts_census\misc_ts\nhgis0010_ts_geog2010_tract.csv",
    encoding="cp1252",
)
pop_bdg_timeseries = pd.read_csv(
    r"C:\work\water_use\ml_experiments\annual_v_0_0\pop_timeseries.csv"
)
nhouses_bdg_timeseries = pd.read_csv(
    r"C:\work\water_use\ml_experiments\annual_v_0_0\nhouses_timeseries.csv"
)

pop_bdg_timeseries["bg2wsa_factor"] = (
    pop_bdg_timeseries["partial_bdg_count"]
    / pop_bdg_timeseries["total_bdg_count"]
)

# =================
# house Age
# =================
codes = """
      JSDE001:     Total
        JSDE002:     Built 2005 or later
        JSDE003:     Built 2000 to 2004
        JSDE004:     Built 1990 to 1999
        JSDE005:     Built 1980 to 1989
        JSDE006:     Built 1970 to 1979
        JSDE007:     Built 1960 to 1969
        JSDE008:     Built 1950 to 1959
        JSDE009:     Built 1940 to 1949
        JSDE010:     Built 1939 or earlier
"""
codes = codes.strip().split("\n")
code_dict = {}
for cod in codes:
    key, value = cod.split(":")
    key = key.strip()
    value = value.strip()
    code_dict[key] = "hs_" + value.replace(" ", "")

bg_wsa_mapping = pop_bdg_timeseries[["GISJOIN", "WSA_AGIDF", "bg2wsa_factor"]]
house_age_df = house_age_df[["GISJOIN"] + list(code_dict.keys())]
house_age_df = bg_wsa_mapping.merge(house_age_df, how="left", on="GISJOIN")
hs_cols = list(code_dict.keys())
for col in hs_cols:
    house_age_df[code_dict[col]] = (
        house_age_df[col] * house_age_df["bg2wsa_factor"]
    )
    house_age_df[code_dict[col]] = house_age_df[code_dict[col]].astype(int)
    del house_age_df[col]
house_age_df = house_age_df.groupby("WSA_AGIDF").sum()


# =================
# household, nfamilies
# =================
codes = """
CM4AA1990:   1990: Households
CM4AA2000:   2000: Households
CM4AA2010:   2010: Households
CM5AA1990:   1990: Families
CM5AA2000:   2000: Families
CM5AA2010:   2010: Families
"""
codes = codes.strip().split("\n")
code_dict = {}
household_df = []
family_df = []
for cod in codes:
    key, year, value = cod.split(":")
    key = key.strip()
    value = value.strip()
    year = year.strip()
    code_dict[key] = value.replace(" ", "")
    df_ = fmaily_hoshod[["GISJOIN", key]].copy()
    if "Households" in cod:
        if len(household_df) == 0:
            df_.rename(columns={key: int(year)}, inplace=True)
            household_df = df_.copy()
        else:
            household_df[int(year)] = df_[key].copy()
    else:
        if len(family_df) == 0:
            df_.rename(columns={key: int(year)}, inplace=True)
            family_df = df_.copy()
        else:
            family_df[int(year)] = df_[key].copy()

household_df = household_df.melt(id_vars=["GISJOIN"])
household_df.rename(
    columns={"variable": "year", "value": "household3"}, inplace=True
)
family_df = family_df.melt(id_vars=["GISJOIN"])
family_df.rename(
    columns={"variable": "year", "value": "family3"}, inplace=True
)
fam_hous = family_df.merge(household_df, how="left", on=["GISJOIN", "year"])
fam_hous = bg_wsa_mapping.merge(fam_hous, how="left", on="GISJOIN")


x = 1
