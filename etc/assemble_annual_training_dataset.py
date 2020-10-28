import os, sys
import pandas as pd
import geopandas as gpd
import numpy as np

"""
Assemble Annual training dataset
"""



def assemble_annual_wu(input):
    """
    This function uses swud wu use data (csv file), WSA shapefile, data colloctor to generate the
    initial training dataset
    :param input:
    :return:
    """
    # read input files
    wu_df = pd.read_csv(input.wu_file)
    wsa_shp = gpd.read_file(input.wsa_shp_file)

    # extract intersection of wsa shapefile and wu_df
    ws = set(wsa_shp['WSA_AGIDF']).intersection(set(wu_df['WSA_AGIDF']))
    wu_df = wu_df[wu_df['WSA_AGIDF'].isin(ws)]

    # get census data

    func_end = 1


if __name__ == "__main__":

    # just container of input data
    class Inputs():
        pass


    wu_inputs = Inputs()
    wu_inputs.wu_file = r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\SWUDS v13.csv"
    wu_inputs.wsa_shp_file = r"C:\work\water_use\mldataset\gis\wsa\WSA_v2_1_alb83_attrib.shp"

    assemble_annual_wu(wu_inputs)



