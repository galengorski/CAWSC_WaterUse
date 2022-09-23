import os
import pandas as pd

output_name = "wu_2000_2020_v_0.csv"
ws = r"C:\work\water_use\mldataset\ml\predictions"
mon_frac_file = r"C:\work\water_use\CAWSC_WaterUse\ml_experiments\annual_v_0_0\models\monthly\m7_29_2022\predictions\prediction_denoised_model_with_selected_features.csv"
ann_per_capita_file = r"C:\work\water_use\CAWSC_WaterUse\ml_experiments\annual_v_0_0\models\annual\m_9_20_2022\predictions\prediction_all_denoised_smc_xgb.csv"
huc12_file = r"C:\work\water_use\mldataset\ml\predictions\huc12_disaggregation\huc12_disaggregation.csv"

monthly_df = pd.read_csv(mon_frac_file)
annual_df = pd.read_csv(ann_per_capita_file)
huc12_df = pd.read_csv(huc12_file)
huc12_df.rename(columns = {"wsa_agidf":'sys_id'}, inplace=True)

pred_df = monthly_df.merge(annual_df, how = 'left', on = ['sys_id', 'Year'])
pred_df = pred_df[['sys_id', 'Year', 'Month', 'pop', 'est_per_capita', 'est_month_frac']]
all_df = []
for yr in range(2000,2021):
    df_ = huc12_df.copy()
    df_['Year'] = yr
    days_in_year = 365
    if yr%4 == 0:
        days_in_year = 366
    df_['ndays'] = days_in_year

    for mon in range(1,13):
        df_['Month'] = mon
        all_df.append(df_.copy())
all_df = pd.concat(all_df)
all_df = all_df.merge(pred_df, how='left', on = ['sys_id', 'Year', 'Month'])

all_df['final_wu'] = all_df['pop'] * all_df['est_per_capita']*all_df['wu_frac'] * all_df['est_month_frac']
all_df = all_df[['huc12', 'Year', 'Month', 'final_wu']]
all_df['final_wu'] = all_df['final_wu']/1e6
all_df = all_df.groupby(by = ['huc12', 'Year', 'Month']).sum()
all_df.reset_index(inplace = True)
all_df = all_df.pivot(index=['Year', 'Month'], columns='huc12')
all_df.columns = all_df.columns.droplevel(0)

all_df.to_csv(os.path.join(ws, output_name), index = False)

cc = 1