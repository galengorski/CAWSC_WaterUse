import os
import matplotlib.pyplot as plt
import numpy as np
import time

t1 = time.time()
#os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask

if __name__ == "__main__":
    import modin.pandas as pd
    wu_df = pd.read_csv(r"C:\work\water_use\dataset\dailywu\pa\all_PA_raw_data.csv")

    def remove_outlier_sys(wu_df):
        wu_df['REPORT_DATE'] = pd.to_datetime(wu_df['REPORT_DATE'])
        wu_df['month'] = wu_df['REPORT_DATE'].dt.month
        wu_df['year'] = wu_df['REPORT_DATE'].dt.year
        wu_df['day'] = wu_df['REPORT_DATE'].dt.day
        wu_df.to_csv('PA_wu_cleaned.csv')
        # monthly_mean =  wu_df.groupby(by=['month']).mean()
        sys_ids = wu_df['PUBLIC_WATER_SUPPLY'].unique()
        montly = {}
        for i, sid in enumerate(sys_ids):
            print(i)
            if pd.isna(sid):
                continue
            curr_df = wu_df[wu_df['PUBLIC_WATER_SUPPLY'] == sid]
            curr_df = curr_df[['month', 'QUANTITY']]
            try:
                curr_df = curr_df.groupby(by=['month']).mean()
            except:
                continue
            curr_df['QUANTITY'] = curr_df['QUANTITY'] / curr_df['QUANTITY'].sum()
            curr_vals = curr_df['QUANTITY'].values.copy()
            if len(curr_vals) == 12:
                montly[sid] = curr_vals
        montly = pd.DataFrame(montly)
        montly.to_csv("C:\work\water_use\dataset\dailywu\pa\pa_monthly_fractions.csv")

    def remove_extreme_values():
        df_fr = pd.read_csv(r"C:\work\water_use\dataset\dailywu\pa\pa_monthly_fractions.csv")
        df_fr = df_fr.dropna(axis=1)
        service_areas = df_fr.columns

        for col in df_fr.columns:
            if 0 in (df_fr[col].unique()):
                df_fr.drop(col, inplace=True, axis=1)
        # del(df_fr['Unnamed: 0'])
        q95 = df_fr.quantile(q=0.95, axis=1)
        q5 = df_fr.quantile(q=0.05, axis=1)

        for col in df_fr.columns:
            if np.any(df_fr[col] > q95) or np.any(df_fr[col] < q5):
                df_fr.drop(col, inplace=True, axis=1)
        df_fr.to_csv(r"C:\work\water_use\dataset\dailywu\pa\wu_clean_outliers.csv")


    wu_df = wu_df[['REPORT_DATE', 'PUBLIC_WATER_SUPPLY', 'QUANTITY']]
    wu_df = wu_df[wu_df['QUANTITY'] > 0]
    q99 = wu_df['QUANTITY'].quantile(0.99)
    wu_df = wu_df[wu_df['QUANTITY'] <= q99]


    remove_outlier_sys(wu_df)
    remove_extreme_values()

    mon_fracs = pd.read_csv(r"C:\work\water_use\dataset\dailywu\pa\wu_clean_outliers.csv")
    service_area_list = []
    for service_area in mon_fracs.columns:
        try:
            service_area_list.append(float(service_area))
        except:
            pass

    wu_df = wu_df[wu_df['PUBLIC_WATER_SUPPLY'].isin(service_area_list)]
    wu_df = wu_df[wu_df['QUANTITY'] > 0]
    wu_df.to_csv(r"C:\work\water_use\dataset\dailywu\pa\PA_wu_cleaned.csv")
    print("Done....")
    xx = 1
