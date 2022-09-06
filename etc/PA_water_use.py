import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from dask.distributed import Client
import dask
import dask.dataframe as ddf

# client = Client(n_workers=20, threads_per_worker=1, processes=False, memory_limit='2.5GB')

# from pandas.tseries.tools import to_datetime
# my_ddf['time'].map_partitions(to_datetime, columns='time').compute()

wu_df = ddf.read_csv(r"C:\work\water_use\all_PA_raw_data.csv")


# wu_df = wu_df[['PUBLIC_WATER_SUPPLY', 'REPORT_DATE', 'QUANTITY']]
wu_df = wu_df[wu_df["QUANTITY"] >= 0]
q99 = wu_df["QUANTITY"].quantile(0.99)
wu_df = wu_df[wu_df["QUANTITY"] <= q99]

wu_df["REPORT_DATE"] = ddf.to_datetime(wu_df["REPORT_DATE"])

wu_df["month"] = wu_df["REPORT_DATE"].dt.month
wu_df["year"] = wu_df["REPORT_DATE"].dt.year


def operation(df):
    df["new"] = df[0]
    return df[["new"]]


wu_df.to_csv("boston.csv")
