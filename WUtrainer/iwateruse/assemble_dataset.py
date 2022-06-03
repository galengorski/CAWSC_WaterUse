import os, sys
import dask.dataframe as pdd
import pandas as pd
import numpy as np

def frac_to_date(frac_date):
    yyear = int(frac_date)
    days = frac_date - yyear
    if yyear%4 == 0:
        days = round(366 * days)
    else:
        days = round(365 * days)
    ddate  = pd.to_datetime('{}-1-1'.format(yyear)) + pd.to_timedelta(days , unit='D')
    return ddate


def assemble_training_dataset(wu_file, date_field = 'date', wu_field = 'wu', sys_id_field = 'x',
                              output_file = 'master_file.csv', census_climate_folder = '.',
                              func_to_process_sys_name = None, to_galon = 1.0):

    wu_df = pd.read_csv(wu_file, parse_dates=[date_field])
    folder = census_climate_folder
    sys_ids = wu_df[sys_id_field].unique()

    wu_df['population'] = np.NAN
    wu_df['median_income'] = np.NAN
    wu_df['pop_density'] = np.NAN

    wu_df['pr']  = np.NAN
    wu_df['pet']  = np.NAN
    wu_df['tmmn']  = np.NAN
    wu_df['tmmx']  = np.NAN

    wu_df[wu_field] =  wu_df[wu_field] * to_galon

    wu_df = wu_df.set_index(keys = [date_field])
    index = 0
    for sys_id in sys_ids:
        index = index + 1
        # get current wu
        sa_mask = wu_df[sys_id_field]==sys_id
        curr_wu = wu_df[sa_mask]
        print(sys_id)

        if not(func_to_process_sys_name is None):
            sys_id = func_to_process_sys_name(sys_id)

        #census
        fn = "cs_{}.csv".format(sys_id)
        fn_cs = os.path.join(folder, fn)
        if os.path.isfile(fn_cs):
            cs_df = pd.read_csv(fn_cs)[['dyear', 'population', 'median_income', 'pop_density']]
            cs_df['date'] = cs_df['dyear'].apply(frac_to_date)
            cs_df['sys_id'] = sys_id.lower()
            cs_df = cs_df.set_index(keys=['date'])
            for cs_var in ['population', 'median_income', 'pop_density']:
                wu_df.loc[sa_mask, cs_var] = cs_df[cs_var]

        #climate
        for var in ["pet", "pr", "tmmn", "tmmx"]:
            clim_fn = "{}_{}.csv".format(var, sys_id)
            clim_fn = os.path.join(folder, clim_fn)
            if os.path.isfile(clim_fn):
                cc_df = pd.read_csv(clim_fn, parse_dates=['day'])
                field_names = list(cc_df.columns)
                field_names.remove('day')
                cc_df['sys_id'] = sys_id
                cc_df = cc_df.set_index(keys=['day'])
                wu_df.loc[sa_mask, var] = cc_df[field_names[0]]

    wu_df.reset_index(inplace=True)
    columns = {date_field: "Date", wu_field: "wu_rate", sys_id_field : "sys_id"}
    for field in columns.keys():
        wu_df[columns[field]] = wu_df[field]


    subcolumns = ["sys_id", "Date", 'wu_rate', 'population', 'median_income', 'pop_density', 'pr',
       'pet', 'tmmn', 'tmmx']
    wu_df = wu_df[subcolumns]
    wu_df.to_csv(output_file)


if __name__ == '__main__':

    # Example....
    wu_file = r"C:\work\water_use\dataset\dailywu\wi\wi_raw_data.csv"
    output_file = r"C:\work\water_use\dataset\dailywu\wi\wi_master.csv"
    assemble_training_dataset(wu_file, date_field='PUMP_DATE', wu_field='PUMP_Kgal', sys_id_field='PWS_ID',
                              output_file= output_file, census_climate_folder= os.path.dirname(wu_file),
                              to_galon = 1000)