import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas
import seaborn as sns

annual_wu = pd.read_csv(r"annual_wu.csv")
monthly_wu = pd.read_csv("monthly_wu.csv")
clusters = pd.read_csv("all_clusters.csv")
wsa_shp = geopandas.read_file(r"C:\work\water_use\mldataset\gis\wsa\WSA_v1.shp")
cluster_shp = geopandas.read_file(r"C:\work\water_use\mldataset\gis\clusters\WSA_Clustering_info.shp")

# =======================
# Add cluster ID to annual wu
# =======================


clusters.rename(columns={'Unnamed: 0': 'cluster_id'}, inplace=True)
cluster_ids = clusters['cluster_id'].unique()
cols = [str(i) for i in range(236)]
sys_names = []
clus_ids = []
for irow, row in clusters.iterrows():
    syss = row[cols][~row[cols].isna()]
    syss = syss.values.tolist()
    sys_names = sys_names + syss
    clus_ids = clus_ids + len(syss) * [row['cluster_id']]

cluster_info = pd.DataFrame(clus_ids, columns = ['cluster_id'])
cluster_info['WSA_AGIDF'] = sys_names
annual_wu = annual_wu.merge(cluster_info, how = 'left', on = 'WSA_AGIDF')

### Neutral sys
annual_wu['swuds_pc'] = np.log10(1+annual_wu['swuds_pc'])
annual_wu['nonswuds_pc'] = np.log10(1+annual_wu['nonswuds_pc'])
neutral_sys = annual_wu[annual_wu['cluster_id'].isna()]
exg_sys = annual_wu[~annual_wu['cluster_id'].isna()]
exg_sys_merged = exg_sys.groupby(['cluster_id', 'YEAR']).sum()
exg_sys_merged.reset_index(inplace = True)

exg_sys_merged = exg_sys_merged[exg_sys_merged['pop']>0]
exg_sys_merged['swuds_pc'] = np.log10(1+exg_sys_merged['annual_wu_G_swuds']/(365*exg_sys_merged['pop']))
exg_sys_merged['nonswuds_pc'] = np.log10(1+exg_sys_merged['annual_wu_G_nonswuds']/(365*exg_sys_merged['pop']))
clusters = clusters[clusters['in_WSA'] == 1]


## Check how many nonwuds are in WSA
annual_nonswud = annual_wu[annual_wu['annual_wu_from_monthly_nonswuds_G']>0]
print("Number of unique systems in Nonswud data is {}".format(len(annual_nonswud['WSA_AGIDF'].unique())))

outWSA_nonswud = annual_nonswud[annual_nonswud['inWSA'] == 0]

print("Number of unique nonswud systems that are not in WSA is {}".format(len(outWSA_nonswud['WSA_AGIDF'].unique())))

clust_sys = np.unique(clusters[cols].values.astype(str))
clust_sys = clust_sys[clust_sys != 'nan']

outWSA_nonswud_connected = outWSA_nonswud[outWSA_nonswud['WSA_AGIDF'].isin(clust_sys)]['WSA_AGIDF'].unique()
cc = 1