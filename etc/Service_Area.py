import os, sys
import pandas as pd
import numpy as np

"""
A generic class to deal with service area information.
"""

class ServiceArea(object):

    def __init__(self, serv_data_excel = None, shp_attribute = None):
        self.serv_data_excel = serv_data_excel
        self.shp_attribute = shp_attribute
        pass


    def get_sev_areas_with_wu_data(self):

        self.shp_df = pd.read_csv(self.shp_attribute, encoding = "ISO-8859-1")
        self.wu_df = pd.read_csv(self.serv_data_excel)

        #self.shp_df['SID_PWS'] = list(zip(self.shp_df['USGS_SID'], self.shp_df['PWS_ID']))
        #self.wu_df['SID_PWS'] = list(zip(self.wu_df['T_SID'], self.wu_df['T_PWSID']))

        found_list = []

        self.wu_df['GNIS_ID'] = 0
        fidw = open('SystemsNotinShapeFile.txt', 'w')
        for irow, row in self.wu_df.iterrows():

            print(100.0* irow/len(self.wu_df))

            pws_id = row['T_PWSID']
            sid = row['T_SID']
            if (pws_id in found_list) or (sid in found_list):
                continue

            if (pws_id in self.shp_df['WSA_AGG_ID'].values)  and not(pd.isna(pws_id)):


                mask = self.shp_df['WSA_AGG_ID'].values == pws_id
                gnis = self.shp_df.loc[mask, 'GNIS_ID'].values[0]
                if np.isnan(gnis):
                    fidw.write('{}---{}'.format(pws_id, sid))
                    fidw.write("\n")
                    continue

                self.wu_df.loc[self.wu_df['T_PWSID'].values==pws_id, 'GNIS_ID'] = int(gnis)
                found_list.append(pws_id)
                continue



            if (sid in self.shp_df['USGS_SID'].values) and not(pd.isna(sid)):
                gnis = self.shp_df.loc[self.shp_df['USGS_SID'].values == sid, 'GNIS_ID'].values[0]
                if np.isnan(gnis):
                    fidw.write('{}---{}'.format(pws_id, sid))
                    fidw.write("\n")
                    continue

                self.wu_df.loc[self.wu_df['T_SID'].values == sid, 'GNIS_ID'] = int(gnis)
                found_list.append(sid)
                continue

            fidw.write('{}---{}'.format(pws_id,sid ))
            fidw.write("\n")
        fidw.close()
        self.wu_df.to_csv(r"D:\Workspace\projects\machine_learning\data\dataset\WU_Data\water_use_data_with_shp.csv")



    def Data_Summary(self):
        wu_df = pd.read_csv(r"D:\Workspace\projects\machine_learning\data\dataset\WU_Data\water_use_data_with_shp.csv")
        shp =  pd.read_csv(self.shp_attribute, encoding = "ISO-8859-1")

        xx = 1
        pass




if __name__ == '__main__':

    srv_info_file = r"D:\Workspace\projects\machine_learning\data\dataset\WU_Data\water_use_data.csv"
    shp_attribute = r"D:\Workspace\projects\machine_learning\data\dataset\WU_Data\service_areas_shp.csv"
    wu_with_shp = r"D:\Workspace\projects\machine_learning\data\dataset\WU_Data\water_use_data_with_shp.csv"
    sa = ServiceArea(serv_data_excel=srv_info_file, shp_attribute = shp_attribute)

    if 1: # link shp with excel
        sa.get_sev_areas_with_wu_data()
    sa.Data_Summary()