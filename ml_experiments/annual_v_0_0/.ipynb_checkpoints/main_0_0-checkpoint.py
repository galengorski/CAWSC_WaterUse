import os, sys
sys.path.append(r"C:\work\water_use\CAWSC_WaterUse\WUtrainer")
import data_cleaning
import report
from featurize import MultiOneHotEncoder


import configparser

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import xgboost as xgb
import matplotlib.pyplot as plt
#import seaborn as sns
from xgboost import plot_importance
import tensorflow as tf
import warnings

#sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

class Model:
    pass

def initialize(model, config_file):

    model.config = configparser.ConfigParser()
    model.config.read(r"C:\work\water_use\ml_experiments\annual_v_0_0\config_file.ini")

    model.name = model.config.get("General", 'Model_Name')
    model.version = model.config.get("General", 'Version')
    model.type = model.config.get("General", 'Type')

    model.workspace = model.config.get("Files", "Workspace")
    model.train_file = model.config.get("Files", "Train_file")
    model.target = model.config.get("Target", "target_field")

    logfile = model.config.get("Files", "Log_File")
    if os.path.isabs(logfile):
        abs_logfile = logfile
    else:
        abs_logfile = os.path.join(model.workspace, logfile)
    model.log = report.Logger(abs_logfile, title = "Public Water Use")
    model.log.info("initialize()...")
    model.log.info("Model Name: {}".format(model.name))
    model.log.info("Type: {}".format(model.type))
    model.log.info("Version: {}".format(model.version))

def load(model):
    #model.log.break_line()
    model.log.info("load()...")

    feature_info_df = pd.read_excel(os.path.join(model.workspace, model.config.get('Features', 'features_info_file')))
    model.feature_info_df = feature_info_df
    model.features = model.feature_info_df[feature_info_df['Skip'] != 1]['Feature_name'].values.tolist()

    df_main = pd.read_csv(model.train_file )
    df_train = df_main[df_main['wu_rate'] > 0]
    model.df_main = df_main
    model.df_train = df_train

    model.log.info("Finish Loading the data ...")

    title = 'Full annual data, shape = {}'.format(model.df_train.shape)
    model.log.to_table(df_train, title = title)

    summary = model.df_train.describe()
    model.log.to_table(summary, title = "Raw Training Data Summary")

def preprocess(model):
    #model.log.break_line()
    model.log.info("preprocess() ...")

    model.log.info("drop unused columns ...")
    feature_mask = ~((model.feature_info_df['Skip'] == 1) | (model.feature_info_df['Not_Feature'] == 1))
    features_to_drop = model.feature_info_df[~feature_mask]['Feature_name'].values.tolist()
    model.df_train = model.df_train.drop(features_to_drop, axis = 1)
    msg = ""
    for ft in features_to_drop:
        msg = msg + ft + ", "
    model.log.info(" features dropped : {}".format(msg))

    # make sure categorical data are integer
    if 0: #xgboost doenot like this
        cat_features = model.feature_info_df[model.feature_info_df['Type'] == 'categorical']['Feature_name'].values
        for feat in cat_features:
            if feat in model.df_train.columns:
                model.df_train[feat] = model.df_train[feat].astype("Int64")


    model.log.to_table(model.df_train, title= "Training data -- shape = {}".format(model.df_train.shape))


def clean(model):
    model.log.info("clean()...")

    """ # number of establishment
    Remove D values from Num_establishments_2017
    """
    mask = model.df_train['Num_establishments_2017'].str.contains('D')
    model.df_train.loc[mask.astype(bool), 'Num_establishments_2017'] = np.nan
    model.df_train['Num_establishments_2017'] = model.df_train['Num_establishments_2017'].astype(float)



    """ # Summarize missing data"""
    model.log.info("Missing data")
    missing_df = data_cleaning.summarize_missing_data (df = model.df_train, target = model.target)
    model.log.to_table(missing_df, title="Missing data ", header=len(missing_df))



    """ Impute use XGBOOST"""
    model.log.info("Imputing using Xgboost")
    df_in = model.df_train.copy()
    cat_features = model.feature_info_df[model.feature_info_df['Type'] == 'categorical']['Feature_name'].values
    for impute_col in missing_df['index'].values:
        flg = False
        if impute_col in cat_features:
            flg = True

        model.log.info("Imputing {}".format(impute_col))
        imputed_feature = data_cleaning.fill_missing_data_using_xgboost(df_in, impute_col, is_categorical = flg)
        model.df_train[impute_col] = imputed_feature

    model.log.info("Missing data after imputing ....")
    missing_df = data_cleaning.summarize_missing_data(df=model.df_train, target=model.target)
    model.log.to_table(missing_df, title="Missing data ", header=len(missing_df))

    fn_clean_df = os.path.join(model.workspace, "clean_train_db.csv")
    model.log.info("Clean database is generated at {} ....".format(fn_clean_df))
    model.df_train.to_csv(fn_clean_df, index = False)

def featurize(model):
    """
    From on we use pipelines
    :param model:
    :return:
    """
    pass

def split(model):
    pass

def main():

    #
    model = Model()
    config_file = r"config_file.ini"
    initialize(model, config_file)
    load(model)
    preprocess(model)

    use_clean_file = True
    if use_clean_file:
        fn_clean_df = os.path.join(model.workspace, "clean_train_db.csv")
        model.log.info("Clean database is readed from {} ....".format(fn_clean_df))
        model.df_train.to_csv(fn_clean_df, index=False)

    else:
        clean(model)

    featurize(model)

    c = 1


if __name__ == "__main__":
    main()




