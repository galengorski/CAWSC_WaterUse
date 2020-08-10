import os
import matplotlib.pyplot as plt

#os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray
os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
if __name__ == "__main__":
    import modin.pandas as pd
    wu_df = pd.read_csv(r"C:\work\water_use\all_PA_raw_data.csv")
    wu_df = wu_df[wu_df['QUANTITY'] >= 0]
    q99 = wu_df['QUANTITY'].quantile(0.99)
    wu_df = wu_df[wu_df['QUANTITY'] <= q99]

    wu_df['REPORT_DATE'] = pd.to_datetime(wu_df['REPORT_DATE'])

    wu_df['month'] = wu_df['REPORT_DATE'].dt.month
    wu_df['year'] = wu_df['REPORT_DATE'].dt.year

    sys_ids = wu_df['PUBLIC_WATER_SUPPLY'].unique()
    for i, sid in enumerate(sys_ids):
        print(i)
        if pd.isna(sid):
            continue
        curr_df =  wu_df[wu_df['PUBLIC_WATER_SUPPLY']==sid]
        curr_df = curr_df[['month', 'QUANTITY' ]]
        curr_df = curr_df.groupby(by=['month']).mean()

        curr_df['QUANTITY'].plot()

plt.show()

