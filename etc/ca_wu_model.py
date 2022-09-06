import os, sys
import pandas as pd
import numpy as np

import dask.dataframe as ddf


db_root = r"C:\work\water_use\mldataset"

annual_training_file = os.path.join(
    db_root, r"ml\training\train_datasets\annual\wu_annual_training.csv"
)
df_ca_audit = pd.read_csv(
    r"C:\work\water_use\data\CA_water\CA_water_audit_data.csv"
)
df_ca_sys = pd.read_csv(
    r"C:\work\water_use\data\CA_water\drinking-water-watch-public-water-system-facilities.csv"
)
df_ca_wu = pd.read_csv(
    r"C:\work\water_use\data\CA_water\produced-water-public2013-2016.csv"
)
train_db = pd.read_csv(annual_training_file)

# corrections
# 1) correct years
df_ca_audit.loc[df_ca_audit["year"] < 100, "year"] = (
    df_ca_audit.loc[df_ca_audit["year"] < 100, "year"] + 2000
)

# 2) convert units to gallons
flows_of_interest = [
    "water supplied own source",
    "water supplied imported",
    "water supplied exported",
    "total water supplied",
    "water losses",
    "real losses",
    "length of mains in miles",
    "number active and inactive connections",
    "service connection density conn/mile",
]
mask_acre_ft = df_ca_audit["units"].isin(["Acre-feet"])
mask_Mgallons = df_ca_audit["units"].isin(["Million gallons (US)"])
for foi in flows_of_interest:
    val = df_ca_audit.loc[mask_acre_ft, foi].values
    val = val * 325851  # acr-ft to gallons
    df_ca_audit.loc[mask_acre_ft, foi] = val

    val = df_ca_audit.loc[mask_Mgallons, foi].values
    val = val * 1e6  # acr-ft to gallons
    df_ca_audit.loc[mask_Mgallons, foi] = val

df_ca_audit["units"] = "gallons"

# fix system name
df_ca_audit["sys_id"] = df_ca_audit["pwsid"].str.extract("(\d+)")
df_ca_audit = df_ca_audit[df_ca_audit["sys_id"].str.len() == 7]
df_ca_audit["sys_id"] = "CA" + df_ca_audit["sys_id"].astype(str)
df_ca_audit.reset_index(inplace=True)

# population
df_ca_sys = df_ca_sys[
    df_ca_sys["Federal Water System Type"].isin(["Community"])
]
df_ca_wu["gpcd"] = df_ca_wu[
    " CALCULATED GPCD (Total Potable Produced in gallons per capita day) "
].str.replace(",", "")


def str2dig(x):
    x_ = str(x)
    x_ = x_.strip()
    x_ = x_.replace(",", "")
    try:
        return float(x_)
    except:
        return None


df_ca_wu["gpcd"] = df_ca_wu["gpcd"].apply(lambda x: str2dig(x))

df_ca_wu = df_ca_wu[
    df_ca_wu["Water.System.Classification"] == "Community Water System"
]
df_ca_wu["wu_rate"] = df_ca_wu[
    " TOTAL POTABLE WATER IN GALLONS (Total Does not Include Sold, Non-potable and Recycled amounts) "
].apply(lambda x: str2dig(x))
df_ca_wu = df_ca_wu[df_ca_wu["wu_rate"] > 0]
df_ca_wu["wu_import"] = df_ca_wu[
    " FINSIHIED WATER PURCHASED OR RECEIVED FROM ANOTHER PUBLIC WATER SYSTEM "
].apply(lambda x: str2dig(x))
df_ca_wu["wu_export"] = df_ca_wu[
    " WATER SOLD TO ANOTHER PUBLIC WATER SYSTEM "
].apply(lambda x: str2dig(x))
df_ca_wu["wu_gw"] = df_ca_wu[" WATER PRODUCED FROM GROUNDWATER "].apply(
    lambda x: str2dig(x)
)
df_ca_wu["wu_sw"] = df_ca_wu[" WATER PRODUCED FROM SURFACE WATER "].apply(
    lambda x: str2dig(x)
)
x = 1
