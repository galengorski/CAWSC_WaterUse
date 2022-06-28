import os, sys
from datetime import datetime
import configparser

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import xgboost as xgb
import matplotlib.pyplot as plt

from xgboost import plot_importance
import tensorflow as tf
import warnings

# sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

# local utils
from iwateruse import data_cleaning, report
from iwateruse.featurize import MultiOneHotEncoder
from iwateruse import impute_utils


class Model:
    pass


def initialize(model, config_file):
    model.config = configparser.ConfigParser()
    model.config.read(config_file)

    model.name = model.config.get("General", 'Model_Name')
    model.version = model.config.get("General", 'Version')
    model.type = model.config.get("General", 'Type')

    model.workspace = model.config.get("Files", "Workspace")
    model.train_file = model.config.get("Files", "Train_file")
    model.target = model.config.get("Target", "target_field")

    logfile = model.config.get("Log", "Log_File")
    append_date_to_log_file = model.config.get("Log", "append_date_to_file_name")
    append_date_to_log_file = append_date_to_log_file.strip().lower()
    if "true" in append_date_to_log_file:
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        date = date.replace(':', "_")
        date = date.replace('-', "_")
        logfile = os.path.splitext(logfile)[0] + date + ".log"
    if os.path.isabs(logfile):
        abs_logfile = logfile
    else:
        abs_logfile = os.path.join(model.workspace, logfile)

    model.log = report.Logger(abs_logfile, title="Public Water Use")
    model.log.info("initialize()...")
    model.log.info("Model Name: {}".format(model.name))
    model.log.info("Type: {}".format(model.type))
    model.log.info("Version: {}".format(model.version))


def load(model):
    # model.log.break_line()
    model.log.info("load()...")

    model.log.info(
        "loading: {}".format(os.path.join(model.workspace, model.config.get('Features', 'features_info_file'))))
    feature_info_df = pd.read_excel(os.path.join(model.workspace, model.config.get('Features', 'features_info_file')),
                                    sheet_name="annual")
    model.feature_info_df = feature_info_df
    model.features = model.feature_info_df[feature_info_df['Skip'] != 1]['Feature_name'].values.tolist()

    model.log.info("loading: {}".format(model.train_file))
    df_main = pd.read_csv(model.train_file)
    df_main['sample_id'] = list(df_main.index)
    df_train = df_main[df_main['wu_rate'] > 0]
    model.df_main = df_main
    model.df_train = df_train

    model.log.info("Finish Loading the data ...")
    title = 'Full annual data, shape = {}'.format(model.df_train.shape)
    model.log.to_table(df_train, title=title)

    summary = model.df_train.describe()
    model.log.to_table(summary, title="Raw Training Data Summary")


def preprocess(model):
    """
    (1) filtering (not part of the sklearn pipeline)
     - PC filtering
     - Pop filtering

    (2) categorical features transfromations

    (3) Contenuous features transfromations
        - StandardScaler()

    """

    pass


def prepare(model):
    """

    :param model:
    :return:
    """
    pass


def list_to_df(clist, ncols):
    all_columns = []
    curr_line = []
    for ifeat, feat in enumerate(clist):
        if np.mod(ifeat, ncols) == 0:
            all_columns.append(curr_line)
            curr_line = []
        curr_line.append(feat)

    if len(curr_line) <= ncols:
        clen = len(curr_line)
        all_columns.append(curr_line + [np.nan] * (ncols - clen))

    return pd.DataFrame(all_columns)


def clean(model):
    model.log.info("clean()...")
    model.log.info("Columns in feature-statues file are ...")
    dff = list_to_df(model.feature_info_df['Feature_name'].values, 5)
    model.log.to_table(dff, header=len(dff))

    features_not_in_data_base = set(model.feature_info_df['Feature_name']).difference(set(model.df_train.columns))
    dff = list_to_df(list(features_not_in_data_base), 5)
    model.log.to_table(dff, header=len(dff), title="Columns not in the database ...")

    features_not_in_feature_list = set(model.df_train.columns).difference(set(model.feature_info_df['Feature_name']))
    dff = list_to_df(list(features_not_in_feature_list), 5)
    model.log.to_table(dff, header=len(dff), title="Columns not in the features list ...")

    feature_mask = ~((model.feature_info_df['Skip'] == 1))# | (model.feature_info_df['Not_Feature'] == 1))
    not_features = model.feature_info_df[~(model.feature_info_df['Not_Feature'] == 1)]['Feature_name'].values.tolist()
    features_to_drop = model.feature_info_df[~feature_mask]['Feature_name'].values.tolist()
    features_to_drop = (set(features_to_drop).union(features_not_in_feature_list)).difference(features_not_in_data_base)
    features_to_drop = list(features_to_drop)

    model.df_train = model.df_train.drop(features_to_drop, axis=1)
    msg = ""
    for ft in features_to_drop:
        msg = msg + ft + ", "
    model.log.info(" features dropped : {}".format(msg))
    model.log.to_table(model.df_train, title="Training data -- shape = {}".format(model.df_train.shape))

    """ # Summarize missing data"""
    model.log.info("Missing data")
    missing_df = data_cleaning.summarize_missing_data(df=model.df_train, target=model.target)
    model.log.to_table(missing_df, title="Missing data ", header=len(missing_df))

    """ Impute use XGBOOST"""
    model.log.info("Imputing ...")
    model.impute_method = model.config.get('Features', 'Impute_method')
    model.impute_method = model.impute_method.lower()
    if model.impute_method in ["none"]:
        model.log.info("No imputing is made ...")


    elif model.impute_method in ['iterative']:
        # need some work - drop strings, insure
        X = impute_utils.iterative_imputer(df=model.df_train.copy())

    elif model.impute_method in ['xgboost']:
        model.log.info("Imputing using Xgboost")
        df_in = model.df_train.copy()
        cat_features = model.feature_info_df[model.feature_info_df['Type'] == 'categorical']['Feature_name'].values
        for impute_col in missing_df['index'].values:
            if impute_col in not_features:
                continue

            flg = False
            if impute_col in cat_features:
                flg = True

            model.log.info("Imputing {}".format(impute_col))
            imputed_feature = data_cleaning.fill_missing_data_using_xgboost(df_in, impute_col, is_categorical=flg)
            model.df_train[impute_col] = imputed_feature

    model.log.info("Missing data after imputing ....")
    missing_df = data_cleaning.summarize_missing_data(df=model.df_train, target=model.target)
    model.log.to_table(missing_df, title="Missing data ", header=len(missing_df))

    fn = model.config.get('Files', 'cleaned_training_file')
    fn_clean_df = os.path.join(model.workspace, fn)
    model.log.info("Clean database is generated at {} ....".format(fn_clean_df))
    model.df_train.to_csv(fn_clean_df, index=False)


def featurize(model):
    """
    From now on we use pipelines
    :param model:
    :return:
    """
    pass


def split(model):
    pass


def validate(model):
    pass


def make_pipeline(model):
    pass


def main(config_file):
    #
    model = Model()
    initialize(model, config_file)
    load(model)

    # preprocess(model)

    model.clean_file_name = os.path.abspath(model.config.get("Files", "cleaned_training_file"))

    clean_the_data = model.config.get("Filter Data", "clean_data")
    clean_the_data = clean_the_data.lower()
    if clean_the_data in ['true']:
        clean_the_data = True
    else:
        clean_the_data = False

    if clean_the_data:
        clean(model)
    else:
        model.log.info("No data cleaning is made. A cleaned file is used ... ")
        fn_clean_df = pd.read_csv(model.clean_file_name)
        model.log.info("Clean database is read from {} ....".format(model.clean_file_name))
        model.df_train = fn_clean_df

    featurize(model)

    c = 1


if __name__ == "__main__":
    main(config_file=r"config_file2.ini")
