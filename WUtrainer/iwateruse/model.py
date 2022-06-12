import os, sys
import pandas as pd
from . import report

class Model:
    def __init__(self, name='exp1', df_train=None, feature_status_file = None,  log_file=None):
        self.name = name
        if log_file is None:
            self.log_file = 'train_log.log'
        else:
            self.log_file = log_file


        self.log = report.Logger(self.log_file, title="Annual Water Use")
        self.log.info("initialize ...")
        self.log.info("Model Name: {}".format(name))

        if not (df_train is None):
            self.df_train_bk = df_train.copy()
            self.df_train = df_train.copy()
            self.log.to_table(df_train, title="Raw Training Dataset", header=10)
            summary = self.df_train.describe()
            self.log.to_table(summary, title="Raw Training Data Summary")

        self.raw_target = ''
        self.target = ''
        self._features = []
        #self.features = []

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self._categorical_features = []
        self.feature_status_file = feature_status_file
        self._features_to_skip = []

        if feature_status_file is None:
            self.feature_status = None
        else:
            self.get_feature_status(fn = feature_status_file)

        self.func_types = ['target_func', 'outliers_func',
                           'add_features_func', 'pre_train_func', 'split_func']

        self.steps = []


    def apply_func(self, func=None, type=None, **kwargs):

        if not (func is None):
            func(self, **kwargs)

    def get_feature_status(self, fn):
        self.feature_status_file = fn
        self.feature_status = pd.read_excel(fn, sheet_name='annual')
        not_features = self.feature_status[self.feature_status['Not_Feature']==1]['Feature_name'].values.tolist()
        skip_features = self.feature_status[self.feature_status['Skip']==1]['Feature_name'].values.tolist()
        self._features_to_skip = self._features_to_skip + not_features + skip_features

        self._categorical_features = self.feature_status[self.feature_status['Type'].isin(['categorical'])]['Feature_name'].values.tolist()


    @property
    def features(self):
        features = []
        not_features = self._features_to_skip + [self.target] + [self.raw_target]
        for col in self.df_train.columns:
            if col in not_features:
                continue
            features.append(col)
        self._features = features
        return self._features

    @property
    def categorical_features(self):
        return self._categorical_features

    def add_training_df(self, df_train = None):
        if not (df_train is None):
            self.df_train_bk = df_train.copy()
            self.df_train = df_train.copy()
            self.log.to_table(df_train, title="Raw Training Dataset", header=10)
            summary = self.df_train.describe()
            self.log.to_table(summary, title="Raw Training Data Summary")
        else:
            raise ValueError("Training dataset is empty...")

    def add_feature_to_skip_list(self, features):

        self._features_to_skip = self._features_to_skip + features
        msg = "The folowing features are added to skip list: "
        for feat in features:
            msg = msg + feat + ","
        self.log.info(msg)





