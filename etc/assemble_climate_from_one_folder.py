import os
import pandas as pd
import geopandas

wsa_shp = geopandas.read_file(r"C:\work\water_use\mldataset\gis\wsa\WSA_v1.shp")
sys_ids = wsa_shp['WSA_AGIDF'].values
database_root = r"C:\work\water_use\climate_wsa\wsa_2022_climate_data\output"

all_annual = []
all_monthly = []
all_daily = []
systems_not_found = []

# extract climate
climate_folder = database_root
for sys_id in sys_ids:
    # climate
    system_found = 0
    for var in ["etr", "pr", "tmmn", "tmmx"]:
        clim_fn = "{}_{}.csv".format(var, sys_id)
        clim_fn = os.path.join(climate_folder, clim_fn)
        if os.path.isfile(clim_fn):
            print(sys_id)
            system_found = 1
            cc_df = pd.read_csv(clim_fn, parse_dates=['day'])
            field_names = list(cc_df.columns)
            field_names.remove('day')
            cc_df['sys_id'] = sys_id
            cc_df['year'] = cc_df['day'].dt.year


            cc_df2 = cc_df.copy()
            cc_df2['month'] = cc_df2['day'].dt.month

            month_df = cc_df2.groupby(by=['year', 'month']).mean()
            month_df.rename(columns={sys_id: var}, inplace = True)

            annual_df =  cc_df2.groupby(by=['year']).mean()
            annual_df.rename(columns={sys_id: var}, inplace = True)
            del(annual_df['month'])

            cc_df_worm = cc_df2[cc_df2['month'].isin([4, 5, 6, 7, 8, 9])]
            cc_df_coo = cc_df2[cc_df2['month'].isin([10, 11, 12, 1, 2, 3])]

            annual_df[var + '_warm'] = cc_df_worm.groupby(by=['year']).mean()[sys_id]
            annual_df[var + '_cool'] = cc_df_coo.groupby(by=['year']).mean()[sys_id]

            # annual_df.reset_index(inplace=True)
            # month_df.reset_index(inplace=True)

            if var in ["pr"]:
                annual_df['pr_cumdev'] = (annual_df['pr'] - annual_df['pr'].mean()).cumsum()
                month_df['pr_cumdev'] = (month_df['pr'] - month_df['pr'].mean()).cumsum()

            if var in ['etr']:
                ann_df = annual_df.copy()
                mon_df = month_df.copy()
            else:
                for icol in annual_df.columns:
                    ann_df[icol] = annual_df[icol]

                for icol in month_df.columns:
                    mon_df[icol] = month_df[icol]
        else:
            print(sys_id, ".. Not Found")
            #systems_not_found.append(sys_id)
            continue

    if system_found == 1:
        ann_df.reset_index(inplace=True)
        mon_df.reset_index(inplace=True)
        ann_df['sys_id'] = sys_id
        mon_df['sys_id'] = sys_id
        all_annual.append(ann_df.copy())
        all_monthly.append(mon_df.copy())

all_annual = pd.concat(all_annual)
all_monthly = pd.concat(all_monthly)
all_annual.to_csv(r"C:\work\water_use\mldataset\ml\training\features\annual_climate.csv",  index = False)
all_monthly.to_csv(r"C:\work\water_use\mldataset\ml\training\features\monthly_climate.csv",  index = False)




