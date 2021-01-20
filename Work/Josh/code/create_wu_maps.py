import os
import utilities as utl
import spatial_outlier_detection as sod


def dummy(ddf):
    return ddf


miles_to_m = 1.60934 * 1000
radius = 50 * miles_to_m
ws = os.path.abspath(os.path.dirname(__file__))
wsa_locations = os.path.join(ws, "..", "data", "WSA_v2_1_alb83_attrib.txt")
wsa_data = os.path.join(ws, "..", "data", "Join_swud_nswud.csv")

df = utl.get_input_data(wsa_locations)
df.drop(columns=["tot_wd_mgd"], inplace=True)
df = utl.get_input_data(wsa_data, df)
df1 = sod.spatial_detect(df, radius, sod.mean_stdev)
print(df1.std_flg_1.unique())
print('break')