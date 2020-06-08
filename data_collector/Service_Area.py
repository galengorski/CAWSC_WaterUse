import os, sys
import pandas as pd

"""
A generic class to deal with service area information.
"""

class ServiceArea(object):

    def __init__(self, serv_data_excel = None):
        self.serv_data_excel = serv_data_excel
        pass


    def get_sev_areas_with_wu_data(self):
        pass




if __name__ == '__main__':

    srv_info_file = r"D:\Workspace\projects\machine_learning\data\dataset\WU_Data\WBEE-PS Water Use Conveyance Data-v11 SWUDS records_2020_0528.xlsx"
    sa = ServiceArea(serv_data_excel=srv_info_file)