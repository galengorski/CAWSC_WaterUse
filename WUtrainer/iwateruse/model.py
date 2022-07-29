import os, sys,shutil
import pandas as pd
from . import report
import json


class Model:
    def __init__(self, name='exp1', df_train=None, feature_status_file=None, log_file=None, model_type = 'annual',
                 model_ws = "model_1", clean = False):


        self.name = name
        self.model_type = model_type.strip().lower()
        self.model_ws= model_ws

        if os.path.isdir(self.model_ws):
            if clean:
                print("...>  remove {}".format(self.model_ws))
                shutil.rmtree(model_ws)
                os.mkdir(self.model_ws)
                os.mkdir(os.path.join(self.model_ws, 'figs'))
        else:
            os.mkdir(self.model_ws)
            os.mkdir(os.path.join(self.model_ws, 'figs'))


        if log_file is None:
            self.log_file = os.path.join(self.model_ws, 'train_log.log')
        else:
            self.log_file = os.path.join(self.model_ws, log_file)

        if model_type.lower()=='annual':
            title = "Annual Water Use"
        else:
            title = "Monthly Water Use"
        self.log = report.Logger(self.log_file, title=title)
        self.log.info("initialize ...")
        self.log.info("Model Name: {}".format(name))

        if not (df_train is None):
            #self.df_train_bk = df_train.copy()
            self.df_train = df_train.copy()
            self.log.to_table(df_train, title="Raw Training Dataset", header=10)
            summary = self.df_train.describe()
            self.log.to_table(summary, title="Raw Training Data Summary")

        self.raw_target = ''
        self.target = ''
        self._features = []
        self._features_selected = []
        self.dropped_features = []
        # self.features = []

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
            self.get_feature_status(fn=feature_status_file)

        self.func_types = ['target_func', 'outliers_func',
                           'add_features_func', 'pre_train_func', 'split_func']

        self.steps = []

    def apply_func(self, func=None, type=None, **kwargs):

        if not (func is None):
            func(self, **kwargs)

    def get_feature_status(self, fn):
        self.feature_status_file = fn
        self.feature_status = pd.read_excel(fn, sheet_name='annual')

        if self.model_type == 'annual':
            not_features = self.feature_status[self.feature_status['Not_Feature'] == 1]['Feature_name'].values.tolist()
            skip_features = self.feature_status[self.feature_status['Skip'] == 1]['Feature_name'].values.tolist()
            self._features_to_skip = self._features_to_skip + not_features + skip_features

            cat_feat = self.feature_status[self.feature_status['Type'].isin(['categorical'])][
                'Feature_name'].values.tolist()
            self._categorical_features = []
            for feat in cat_feat:
                if feat in self._features_to_skip:
                    continue
                self._categorical_features.append(feat)
        else:
            not_features = self.feature_status[self.feature_status['Not_Feature'] == 1]['Feature_name'].values.tolist()
            skip_features = self.feature_status[self.feature_status['Skip'] == 1]['Feature_name'].values.tolist()
            skip_features = skip_features+ self.feature_status[self.feature_status['monthly Skip'] == 1]['Feature_name'].values.tolist()
            self._features_to_skip = list(set(self._features_to_skip + not_features + skip_features))

            cat_feat = self.feature_status[self.feature_status['Type'].isin(['categorical'])][
                'Feature_name'].values.tolist()
            self._categorical_features = []
            for feat in cat_feat:
                if feat in self._features_to_skip:
                    continue
                self._categorical_features.append(feat)


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

    def add_training_df(self, df_train=None):

        if not (df_train is None):
            print("Warning: You are overwriting an existing database...")
            self.log.info("Warning: You are overwriting an existing database")


        #self.df_train_bk = df_train.copy()
        self.df_train = df_train.copy()
        self.log.to_table(df_train, title="Raw Training Dataset", header=10)
        summary = self.df_train.describe()
        self.log.to_table(summary, title="Raw Training Data Summary")


    def add_feature_to_skip_list(self, features):
        self.dropped_features = self.dropped_features + features
        self._features_to_skip = list(set(self._features_to_skip + features))
        msg = "The folowing features are added to skip list: "
        for feat in features:
            msg = msg + feat + ","
        self.log.info(msg)

    def dict_to_file(self, data, fn):
        with open(fn, 'w') as ff:
            ff.write(json.dumps(data))

    def load_features_selected(self, method = 'xgb_cover'):
        feat_selec_file = 'confirmed_selected_features.json'
        f = open(feat_selec_file)
        feature_selection_info = json.load(f)
        f.close()
        self.features_selected = feature_selection_info[method]
        return self.features_selected

