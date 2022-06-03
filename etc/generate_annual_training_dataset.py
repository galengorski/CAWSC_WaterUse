import os, sys
import pandas as pd
import geopandas

from assemble_dataset import get_all_annual_db
from assemble_dataset import get_all_monthly_db

#wu_file_swud = r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\SWUDS_v14.csv"
#wu_file_nonswud = r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\NSWUD_annual_monthly_counties.csv"
wu_file_joint =  r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\Join_swud_nswud3.csv"
wsa_file = r"C:\work\water_use\mldataset\gis\wsa\WSA_v1\WSA_v1.shp"
database_root = r"C:\work\water_use\mldataset\ml\training\features"


#suwd_db = pd.read_csv(wu_file_swud)
#nonswud_db = pd.read_csv(wu_file_nonswud)
#wsa_shp = geopandas.read_file((wsa_file))
xx = 1

if True:
    get_all_annual_db(database_root, wu_file_joint, wsa_file, update_train_file=True, wu_field = 'TOT_WD_MGD',
                  sys_id_field = 'WSA_AGIDF')
if False:
    get_all_monthly_db(database_root, wu_file_joint, wsa_file, update_train_file=True, wu_field='TOT_WD_MGD',
                      sys_id_field='WSA_AGIDF')

