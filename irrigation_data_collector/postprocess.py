import numpy as np
import os
import pandas as pd


def vp_deficit_from_q_and_T(q, T_K):
    R_da = 284.04  # dry air gas constant (J kg-1 K-1)
    row_da = 1.292  # dry air gas density (kg m-3)
    T_C = T_K - 273.15  # Celcius Temperature
    vp_saturation = 611 * np.exp(17.27 * T_C / (T_C + 237.3))
    vp = q * R_da * row_da * T_K / 0.622
    vp_deficit = vp_saturation - vp
    return vp_deficit


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

shapelist =['pajaro', 'idaho', 'mississippi', 'colorado']
year_list = ['2015', '2020']
for shape in shapelist:
    for var in var_list:
        for year in year_list:
            out_path = os.path.join('..', 'output', shape, year)
            out_df = pd.DataFrame()
            added_dates = False
            for filename in os.listdir(out_path):
                if filename.endswith('.csv'):
                    var_name = os.path.splitext(filename)[0].split('_')[0]
                    if var_name in var:
                        load_file = pd.read_csv(os.path.join(out_path, filename))
                        if not added_dates:
                            out_df['Date'] = load_file['day']
                            new_col = [
                                var + '_' + shape[0] + '_' + date.split('-')[0] + '_' + date.split('-')[1] + '_' +
                                date.split('-')[2] for date in load_file['day']]
                            out_df.insert(0, 'Variable', new_col)
                            added_dates = True
                        if (shape == 'idaho') or (shape == 'mississippi'):
                            id_name = os.path.splitext(filename)[0].split('_')[1] + '_' + \
                                      os.path.splitext(filename)[0].split('_')[2]
                        else:
                            id_name = os.path.splitext(filename)[0].split('_')[1]
                        out_df[id_name] = load_file[id_name]
            out_df.to_csv(os.path.join('..', 'processed', shape, year, shape + '_' + var + '_' + year + '.csv'), index=False)

# now do vapor pressure deficit
shapelist =['pajaro', 'idaho', 'mississippi', 'colorado']
year_list = ['2015', '2020']
for shape in shapelist:
    for year in year_list:
        vp_totals = [0, 0]
        out_path = os.path.join('..', 'output', shape, year)
        tmmx_k = pd.read_csv(os.path.join('..', 'processed', shape, year, shape + '_' + 'tmmx_k' + '_' + year + '.csv'))
        sph_kgkg = pd.read_csv(os.path.join('..', 'processed', shape, year, shape + '_' + 'sph_kgkg' + '_' + year + '.csv'))
        vpmax_df = pd.DataFrame()
        vpmax_df['Date'] = tmmx_k['Date']
        new_col = ['vp_def_pa_' + shape[0] + '_' + date.split('-')[0] + '_' + date.split('-')[1] + '_' +
                   date.split('-')[2] for date in vpmax_df['Date']]
        vpmax_df.insert(0, 'Variable', new_col)
        for col in tmmx_k.columns:
            if (col != 'Date') and (col != 'Variable'):
                vpmax_df[col] = vp_deficit_from_q_and_T(sph_kgkg[col], tmmx_k[col])
        vpmax_df.to_csv(os.path.join('..', 'processed', shape, year, shape + '_vpdef_pa_' + year + '.csv'), index=False)
        total_max_negs = np.round(100 * np.sum(vpmax_df.iloc[:, 2:].values < 0) / np.sum(vpmax_df.iloc[:, 2:].values != -9999.9999), 3)
        print('\nFor ' + shape + ' ' + year + '...')
        print(str(total_max_negs) + ' percent of (maximum) vapor pressure deficits are negative')




