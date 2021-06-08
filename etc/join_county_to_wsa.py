import os
from CAWSC_WaterUse.etc import utilities as utl

# fields to keep/read in! Please see parameters.txt for a description
# on how to use this file
def add_county_data():
    annual_db_root = r"C:\work\water_use\mldataset\ml"
    PARAMETERS = utl.load_parameters(os.path.join(annual_db_root, r'training\misc_features\parameters.txt'))

    # load existing training data
    trn_data = os.path.join(annual_db_root, r"training\train_datasets\Annual\wu_annual_training.csv")
    trn_df = utl.load_training_data(trn_data)

    # load all csv files and shapefile data
    dfs = utl.load_all(PARAMETERS)
    wsa_df = utl.load_wsas(PARAMETERS)

    # join the county fips code by water service area
    trn_df = utl.join_data(trn_df, wsa_df,
                           left_on="sys_id",
                           right_on='wsa_agidf')

    # loop through the list of dataframes and join by county FIPS code
    for df in dfs:
        trn_df = utl.join_data(trn_df, df)

    # debugging code to validate all fields have been tagged
    for s in list(trn_df):
        print(s)

    # write dataframe to file
    w_out = os.path.dirname(trn_data)
    return trn_df
    #trn_df.to_csv(os.path.join(w_out,"wsa_annual_training_2_extended.csv"), index=False)
