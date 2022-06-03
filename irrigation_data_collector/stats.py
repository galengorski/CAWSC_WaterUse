import numpy as np
import os
import pandas as pd


var_list = [
    'etr_mm',  # Reference ET (mm)
    'pet_mm',  # Potential ET (mm)
    'pr_mm',  # Precipitation (mm)
    'sph_kgkg',  # Specific Humidity (kg kg-1)
    'srad_wm2',  # Shortwave Radiation (W m-2)
    'tmmn_k',  # Minimum Temperature (K)
    'tmmx_k',  # Maximum Temperature (K)
    'vs_ms',  # Wind Velocity (m s-1)
]

monthly = True
annual = True

if monthly:
    # monthly
    shapelist =['pajaro', 'mississippi', 'colorado', 'idaho']
    year_list = ['2015', '2020']
    for shape_ind, shape in enumerate(shapelist):
        for var in var_list:
            for year in year_list:
                in_file = os.path.join('..', 'processed', shape, year, shape + '_' + var + '_' + year + '.csv')
                load_file = pd.read_csv(in_file)
                out_path = os.path.join('..', 'stats', shape, year)
                mean_df = pd.DataFrame(columns=load_file.columns)
                median_df = pd.DataFrame(columns=load_file.columns)
                max_df = pd.DataFrame(columns=load_file.columns)
                min_df = pd.DataFrame(columns=load_file.columns)
                months = np.arange(1, 13)
                mean_df.rename(columns={'Date': 'Month'}, inplace=True)
                median_df.rename(columns={'Date': 'Month'}, inplace=True)
                max_df.rename(columns={'Date': 'Month'}, inplace=True)
                min_df.rename(columns={'Date': 'Month'}, inplace=True)
                mean_df['Month'] = months
                median_df['Month'] = months
                max_df['Month'] = months
                min_df['Month'] = months
                mean_df['Variable'] = [var + '_' + shape[0] + '_mean_' + str(year) + '_' + str(month) for month in months]
                median_df['Variable'] = [var + '_' + shape[0] + '_median_' + str(year) + '_' + str(month) for month in months]
                max_df['Variable'] = [var + '_' + shape[0] + '_max_' + str(year) + '_' + str(month) for month in months]
                min_df['Variable'] = [var + '_' + shape[0] + '_min_' + str(year) + '_' + str(month) for month in months]
                month_arr = np.array([int(date.split('-')[1]) for date in load_file['Date']])
                for month in months:
                    subset = month_arr == month
                    # pandas dataframes
                    # [boolean, positional]
                    # needlessly complex
                    mean_df.iloc[int(month-1), 2:] = np.nanmean(load_file.loc[subset].iloc[:, 2:], axis=0)
                    median_df.iloc[int(month - 1), 2:] = np.nanmedian(load_file.loc[subset].iloc[:, 2:], axis=0)
                    max_df.iloc[int(month - 1), 2:] = np.nanmax(load_file.loc[subset].iloc[:, 2:], axis=0)
                    min_df.iloc[int(month - 1), 2:] = np.nanmin(load_file.loc[subset].iloc[:, 2:], axis=0)
                mean_df.to_csv(os.path.join(out_path, shape + '_' + var + '_' + year + '_mean.csv'), index=False)
                median_df.to_csv(os.path.join(out_path, shape + '_' + var + '_' + year + '_median.csv'), index=False)
                max_df.to_csv(os.path.join(out_path, shape + '_' + var + '_' + year + '_max.csv'), index=False)
                min_df.to_csv(os.path.join(out_path, shape + '_' + var + '_' + year + '_min.csv'), index=False)

if annual:
    # annual
    shapelist =['pajaro', 'mississippi', 'colorado', 'idaho']
    year_list = ['2015', '2020']
    for shape in shapelist:
        for var in var_list:
            for year in year_list:
                in_file = os.path.join('..', 'processed', shape, year, shape + '_' + var + '_' + year + '.csv')
                load_file = pd.read_csv(in_file)
                out_path = os.path.join('..', 'stats', shape, year)
                out_df = pd.DataFrame(columns=load_file.columns)
                out_df.rename(columns={'Date': 'Year'}, inplace=True)
                out_df['Year'] = np.repeat(year, 4)
                new_col = [var + '_' + shape[0] + '_' + str(year) + '_' + str(month) for month in months]
                stat_list = ['mean', 'median', 'max', 'min']
                out_df['Variable'] = [var + '_' + shape[0] + '_' + str(year) + '_' + stat for stat in stat_list]
                out_df.iloc[0, 2:] = np.nanmean(load_file.iloc[:, 2:], axis=0)
                out_df.iloc[1, 2:] = np.nanmedian(load_file.iloc[:, 2:], axis=0)
                out_df.iloc[2, 2:] = np.nanmax(load_file.iloc[:, 2:], axis=0)
                out_df.iloc[3, 2:] = np.nanmin(load_file.iloc[:, 2:], axis=0)
                out_df.to_csv(os.path.join(out_path, shape + '_' + var + '_' + year + '_stats.csv'), index=False)




