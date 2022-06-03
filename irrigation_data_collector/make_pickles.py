import os
import pandas as pd
import pickle


def save_pickles(dict_, filename_):
    with open(os.path.join(filename_+'.pickle'), 'wb') as handle:
        pickle.dump(dict_, handle)
    return


shapelist =['colorado', 'idaho', 'mississippi','pajaro']
year_list = ['2015', '2020']
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
res_list = ['Daily', 'Monthly', 'Annual']
stat_list = ['Mean', 'Median', 'Max', 'Min']
shape_dict = {}

for shape in shapelist:
    year_dict = {}
    for year in year_list:
        var_dict = {}
        for var in var_list:
            res_dict = {}
            for res in res_list:
                if res == 'Daily':
                    in_file = os.path.join('..', 'processed', shape, year, shape + '_' + var + '_' + year + '.csv')
                    load_file = pd.read_csv(in_file)
                    res_dict[res] = load_file
                elif res == 'Monthly':
                    stat_dict = {}
                    for stat in stat_list:
                        in_file = os.path.join('..', 'stats', shape, year, shape + '_' + var + '_' + year + '_' + stat.lower() + '.csv')
                        load_file = pd.read_csv(in_file)
                        stat_dict[stat] = load_file
                    res_dict[res] = stat_dict
                elif res == 'Annual':
                    in_file = os.path.join('..', 'stats', shape, year, shape + '_' + var + '_' + year + '_stats.csv')
                    load_file = pd.read_csv(in_file)
                    stat_dict = {}
                    for stat_ind, stat in enumerate(stat_list):
                        stat_dict[stat] = load_file.iloc[stat_ind, 1:]
                    res_dict[res] = stat_dict
            var_dict[var] = res_dict
        year_dict[year] = var_dict
    shape_name = shape.capitalize()
    shape_dict[shape_name] = year_dict


save_pickles(shape_dict, 'nested_data')

for key1 in shape_dict.keys():
    for key2 in shape_dict[key1].keys():
        for key3 in shape_dict[key1][key2].keys():
            for key4 in shape_dict[key1][key2][key3].keys():
                print(key1)
                print('     ' + key2)
                print('           ' + key3)
                print('                ' + key4)













