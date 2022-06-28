# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas
import configparser
from tqdm.auto import tqdm


# import geopandas
# from ipyleaflet import Map, GeoData, basemaps, LayersControl
# import contextily as cx
# %matplotlib widget
# %matplotlib inline
# %matplotlib ipympl

# %% [markdown]
# # Estimation of Population Size for Water Use Predictions

# %% [markdown]
# Background
# -------------
#
# Population is a critical feature used in estimating water use. The population size, along with other other socioeconomic factors, were downloaded using the census data collector (ref). The census data collector intersects with the spatial boundaries of a service area with spatial boundaries of census block group, census tract, and census places. Two issues arise from intersecting the service area boundary with census boundaries: (1) when a service area, usually not aligned with census boundaries, is smaller than the block group, the extracted service area population value is weighted based on the ratio of intersecting area to total block group area, which can produce inaccurate values for very small service areas, and (2) uncertainty in service area boundaries produces uncertain population estimates. In the following sections, some approaches were used to mitigate the impact of these two issues.
#
#

# %% [markdown]
# Methods
# ---
# Two methods are presented in this document to mitigate uncertainty in served population size.
# ###  Mitigating the effect of partial intersection between service area and census block group. 
# For this task, we used Microsoft US Building Footprint $^{(1)}$ to enhance the estimation of served population. In fig. 1, for example, the service area is very small comparing with the Block group. Computing the population based on weight of the area assumes that the population is uniformly distributed which might not be the case. The US Building Footprint downloaded from https://github.com/Microsoft/USBuildingFootprints, and was converted to a point shapefile. The building point shapefile was intersected with the census Block group shapefile and service area shapefile. For each block group,  the numbers of buildings inside and outside the service area were extracted, and the population size was computed such that it's proportional to the number of buildings.
#
#
# ![image-5.png](attachment:image-5.png)
#
# <p><center> <strong>Figure (1):</strong> Illustration of Population Enhancing Using Microsoft US Building Footprint data</center></p>

# %% [markdown]
# ###  Produce Population Estimates. 

# %%
# uitl functions

def inf_to_nan(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def extrat_year(txt):
    new_string = ''
    for s in txt:
        if s.isdigit():
            new_string = new_string + s
        else:
            new_string = new_string + " "

    years = new_string.strip().split()
    years = [int(y) for y in years]
    years = np.array(years)
    years = years[years > 1000]
    year = np.mean(years)
    if pd.isna(year):
        return 2013
    else:
        return int(year)


# %% [markdown]
# ## Files with information about population
#
# * swud_df15: is annual (and some monthly) water use data from SWUD15. Columns with population info are:
#     - POP_SRV: population served
#     - POP_METH: method of population estimation. Method is not clear
#
# * swud_df16: is annual (and some monthly) water use data from SWUD16. Columns with population info are:
#     - POP_SRV: population served
#     - POP_METH: method of population estimation. Method is not clear
#     
# * (??) wsa_v1: is the population in the service area shapefile. It also has the total and poly population
#
# * (??) wsa_local_pop: this population information that is provided by local agencies.
#
# * pop_timeseries.csv:  time series of population and count of buildings for each block-group service area intersection.
#
# * nhouses_timeseries.csv: time series of number of houses and count of buildings for each block-group service area intersection.
#
# * Zillow dataset
#
#

# %%
# Files Declarations
db_root = r"C:\work\water_use\mldataset"
swud15_file = r"ml\training\targets\monthly_annually\swud_v15.csv"
swud16_file = r"ml\training\targets\monthly_annually\SWUDS v16 WDs by FROM-sites and by WSA_AGG_ID_04282021.csv"
wsa_v1_file = r"C:\work\water_use\mldataset\gis\wsa\WSA_v1.shp"

local_pop_info = r"C:\work\water_use\mldataset\ml\training\misc_features\V1_polys_with_water_service_08072021_for_Ayman_popsrv_year.csv"
pop_bdg_timeseries = r"pop_timeseries.csv"
nhouses_bdg_timeseries = r"nhouses_timeseries.csv"
conveyance_file = r"C:\work\water_use\mldataset\ml\training\misc_features\selling_buying\water_exchange_data.csv"
main_train_file = r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\wu_annual_training3.csv"

# %%

# %%
swud_df15 = pd.read_csv(os.path.join(db_root, swud15_file), encoding='cp1252')
swud_df16 = pd.read_csv(os.path.join(db_root, swud16_file), encoding='cp1252')
wsa_v1 = geopandas.read_file(os.path.join(wsa_v1_file))
wsa_local_pop = pd.read_csv(local_pop_info, encoding='cp1252')

pop_wsa_bg_info = pd.read_csv(pop_bdg_timeseries)
nhs_wsa_bg_info = pd.read_csv(nhouses_bdg_timeseries)
xchg_df = pd.read_csv(conveyance_file, encoding='cp1252')

# read files
df_main = pd.read_csv(main_train_file)
df_train = df_main[df_main['wu_rate'] > 0]

# %%
sys_id = set(swud_df15['WSA_AGIDF']).union(set(swud_df16['WSA_AGIDF']))
sys_id = sys_id.union(set(wsa_v1['WSA_AGIDF']))
sys_id = sys_id.union(set(df_main['sys_id']))
sys_id = sys_id.union(set(pop_wsa_bg_info['WSA_AGIDF']))
sys_id = sys_id.union(set(nhs_wsa_bg_info['WSA_AGIDF']))

master_pop_df = pd.DataFrame()
master_pop_df['WSA_AGIDF'] = list(sys_id)

# add a label for data source
master_pop_df['isin_wsa'] = master_pop_df['WSA_AGIDF'].isin(df_main['sys_id'])
master_pop_df['isin_swud15'] = master_pop_df['WSA_AGIDF'].isin(swud_df15['WSA_AGIDF'])
master_pop_df['isin_swud16'] = master_pop_df['WSA_AGIDF'].isin(swud_df16['WSA_AGIDF'])
master_pop_df['isin_bg'] = master_pop_df['WSA_AGIDF'].isin(nhs_wsa_bg_info['WSA_AGIDF'])
master_pop_df['isin_conv'] = master_pop_df['WSA_AGIDF'].isin(xchg_df['WSA_AGIDF'])

# Let us only keep sys in training data
temp_mean = df_main[['sys_id', 'Year', 'population']].copy()
master_pop_df = master_pop_df[master_pop_df['isin_wsa']].copy()
temp_mean = temp_mean.merge(master_pop_df, how='left', left_on='sys_id', right_on='WSA_AGIDF')
master_pop_df = temp_mean
del (temp_mean)
del (master_pop_df['sys_id'])

# add swud15
sw15 = swud_df15[['WSA_AGIDF', 'POP_SRV']]
sw15 = sw15.groupby(by=['WSA_AGIDF']).mean().reset_index()
master_pop_df = master_pop_df.merge(sw15, how='left', right_on='WSA_AGIDF', left_on='WSA_AGIDF')
master_pop_df['pop_swud15'] = master_pop_df['POP_SRV']
del (master_pop_df['POP_SRV'])
del (sw15)

# add swud16
sw16 = swud_df16[['WSA_AGIDF', 'POP_SRV']]
sw16 = sw16.groupby(by=['WSA_AGIDF']).mean().reset_index()
master_pop_df = master_pop_df.merge(sw16, how='left', right_on='WSA_AGIDF', left_on='WSA_AGIDF')
master_pop_df['pop_swud16'] = master_pop_df['POP_SRV']
del (master_pop_df['POP_SRV'])
del (sw16)

# add wsa_v1
wsav1 = wsa_v1[['WSA_AGIDF', 'WSA_SQKM', 'TPOLYPOP', 'TPOPSRV']]
wsav1 = wsav1.groupby(by=['WSA_AGIDF']).mean().reset_index()
master_pop_df = master_pop_df.merge(wsav1, how='left', right_on='WSA_AGIDF', left_on='WSA_AGIDF')

# compute pop and nhs correction factors
pop_wsa_bg_info = pop_wsa_bg_info.merge(df_main[['sys_id', 'Year', 'population']], how = 'left',
                                        left_on=['WSA_AGIDF', 'year'],
                                        right_on=['sys_id', 'Year'])
del(pop_wsa_bg_info['sys_id'])
del(pop_wsa_bg_info['Year'])

pop_wsa_bg_info['pop_bdg_correc_fac'] = (pop_wsa_bg_info['partial_bdg_count'] / pop_wsa_bg_info['total_bdg_count'])
pop_wsa_bg_info['pop_c'] = pop_wsa_bg_info['pop_bdg_correc_fac'] * pop_wsa_bg_info['pop']
pop_wsa_bg_info['lpop_c'] = pop_wsa_bg_info['pop_bdg_correc_fac'] * pop_wsa_bg_info['lpop']
pop_wsa_bg_info['upop_c'] = pop_wsa_bg_info['pop_bdg_correc_fac'] * pop_wsa_bg_info['upop']
pop_wsa_bg_info = pop_wsa_bg_info.groupby(by=['WSA_AGIDF', 'year']).sum()
pop_wsa_bg_info.reset_index(inplace=True)

# master_pop_df = master_pop_df.merge(pop_wsa_bg_info, how='left', left_on=['WSA_AGIDF', 'Year'],
#                                     right_on=['WSA_AGIDF', 'year'])

def interplotate_ts_pop(df_):
    df_out = pd.DataFrame(np.arange(1990, 2021), columns=['year'])
    df_out = df_out.merge(df_, how='left', on='year')
    df_out.reset_index(inplace=True)
    df_out.sort_values(by='year', inplace=True)

    df_out['pop_c'].interpolate(inplace=True)
    df_out['lpop_c'].interpolate(inplace=True)
    df_out['upop_c'].interpolate(inplace=True)
    df_out.loc[df_out['population'] == 0, 'population'] = np.NaN
    df_out['population'].interpolate(inplace=True)
    df_out.rename(columns={'population':'population_c'}, inplace=True)

    df_out = df_out[df_out['year'] > 1999][['year', 'WSA_AGIDF', 'total_bdg_count',
                                            'pop_c', 'lpop_c', 'upop_c', 'population_c']]
    name = df_out['WSA_AGIDF'].copy()
    sys_id = name.dropna().unique()[0]
    df_out['WSA_AGIDF'] = sys_id
    df_out['total_bdg_count'] = df_out['total_bdg_count'].mean()
    df_out = df_out.astype({'total_bdg_count': int, 'pop_c': int, 'lpop_c': int, 'upop_c': int})
    return df_out

tqdm.pandas(desc = "interpolating population ...")
vv = pop_wsa_bg_info.groupby(['WSA_AGIDF'], group_keys=False).progress_apply(interplotate_ts_pop)
vv.rename(columns = {'year':'Year'}, inplace = True)
master_pop_df = master_pop_df.merge(vv, how = 'left', on = ['WSA_AGIDF', 'Year'])

# use enhanced pop
master_pop_df['pop'] =  master_pop_df['pop_c']
very_small_sys = master_pop_df[master_pop_df['pop'].isna()]['WSA_AGIDF'].unique()
mean_swud = master_pop_df[['WSA_AGIDF','TPOPSRV', 'population_c']].groupby('WSA_AGIDF').mean()
mean_swud['corr_factor'] = mean_swud['TPOPSRV']/mean_swud['population_c']
mean_swud.reset_index(inplace = True)
mean_swud = mean_swud[['WSA_AGIDF', 'corr_factor' ]]
master_pop_df = master_pop_df.merge(mean_swud, how = 'left', on = 'WSA_AGIDF')

# some systems are too small and no bdgs are reported!! For thos systems, we data from swuds first
w_sys_no_bdg = master_pop_df[master_pop_df['pop'].isna()]['WSA_AGIDF'].unique()
for wsys in w_sys_no_bdg:
    sys_mask = master_pop_df['WSA_AGIDF'].isin([wsys])
    popu_c = master_pop_df[sys_mask]['population_c'].mean()
    popu = master_pop_df[sys_mask]['population'].mean()
    pswud16 = master_pop_df[sys_mask]['pop_swud16'].mean()
    tpopsrv = master_pop_df[sys_mask]['TPOPSRV'].mean() # 15 and 16 population are the same

    if not(np.isnan(popu_c)):
        if not (np.isnan(tpopsrv)):
            pop_new = tpopsrv/popu_c * master_pop_df[sys_mask]['population_c']
            master_pop_df.loc[sys_mask, 'pop'] = pop_new.values
            continue

    if not(np.isnan(popu)):
        if not(np.isnan(tpopsrv)):
            pop_new = tpopsrv/popu * master_pop_df[sys_mask]['population']
            master_pop_df.loc[sys_mask, 'pop'] = pop_new.values
            print('Done')
            continue



master_pop_df.to_csv(r"master_population.csv")




