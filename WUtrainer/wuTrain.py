import os, sys
import numpy as np
import pandas as pd
from report import Logger
import featurize

# ML imports
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

class Learner(object):
    def __init__(self, data, features, target, logfile = 'wu_log'):

        if os.path.isfile(logfile):
            os.remove(logfile)
        self.logfile = logfile
        self.loger = Logger(filename=logfile)
        self.loger.add_info_msg(msg = "Initialize Machine Learning ...")
        self.X = data[features]  # m * n
        self.y = data[target]  # m * 1

        self.loger.df_info(self.X)

        self.feature_transforms = []
        self.target_transfrorms = []

        # defualt estimator
        self.estimator = ('xgboost', XGBRegressor())
        self.parameters = {}  # todo: add defualt model params



    def remove_outliers(self):
        #
        # filtering -->
        pass

    def augment_data(self):
        #  X where new_m > m
        #  X  increase n
        pass

    def _pre_training(self):

        # add model
        steps = [self.estimator]

        # setup transforms
        all_parameters = []
        for func_name in self.parameters:
            func_params = self.parameters[func_name]
            for parname in func_params.keys():
                pname = "{}_{}".format(func_name, parname)
                all_parameters[pname] = func_params[parname]

        if len(self.feature_transforms) > 0:
            print("--- Setup features transforms")
            for transfrom in self.feature_transforms:
                steps.insert(0, transfrom)

        pipe = Pipeline(steps=steps)
        pipe.set_params(**all_parameters)

    def train(self):
        """

        :return:
        """
        pass

    def tune(self):
        pass

    def evaluate(self):
        pass

    def generate_model(self):

        # 1) assemble raw dataset in pandas dataframe (generate_dataset)

        # 2) apply the QA/QC (remove_outliers) [X,y]

        # 3) augment the dataset by adding features that helps extrapolate [min and max bounds]

        # 4) split dataset training/validation. Maybe kfold partitioning

        # training

        # 5) Hyper-parameter tuning
        x = 1
        pass

    def post_training(self):
        # features importance, model performance, others
        pass

    def report(self):
        """
        Generate a pdf (or HTML) report that document (with plots) all decision made to produce the model.
        Report model training progress and performance
        Plot data for visual inspection
        Plot validation results.
        .
        .

        :return:
        """
        pass

    def load_model(self):
        pass

    def predict(self):
        pass

    def add_feature_transform(self, transform, parameters={}):
        """

        :param transform: a tuple (name, func)
            name: arbitrary name
            func: functions that use to transform the data
        :param parameters:
        :return:
        """
        self.feature_transforms.append(transform)
        self.parameters[transform[0]] = parameters

    def add_target_transform(self, func, inverse_func):
        self.target_transfrorms.append((func, inverse_func))

    def get_feat_transfrom_list(self):
        transforms = []
        for item in self.wu_pipeline:
            transforms.append(item[0])
        return transforms

    def feature_engineering(self, feat_chain):
        """
        Any feature changes or feature additions that is not implemented in sklearn is implemented herein
        :param feat_chain: is a list of functions. Each function take X (df)
        :return: argument and return changed X
        """
        print(" Feature Engineering ......")
        if len(feat_chain) > 0:
            for func in feat_chain:
                self.X = func(self.X)
                print("  ..... Function {} is implemented".format(func.__name__))



    def add_model(self, model=('xgbrg', XGBRegressor()), parameters={}):
        """

        :param model: tuple (model name, model object)
        :param parameters: dictionary {par_name: par_value}
        :return: None
        """

        self.estimator = model
        self.estimator_parameters = parameters


if __name__ == "__main__":
    datafile = r"C:\work\water_use\dataset\dailywu\wi\wi_master.csv"
    data = pd.read_csv(datafile)

    # 1) assemble raw dataset in pandas dataframe (generate_dataset)

    # 2) apply the QA/QC (remove_outliers) [X,y]

    # 3) augment the dataset by adding features that helps extrapolate [min and max bounds]

    # 4) split dataset training/validation. Maybe kfold partitioning

    # training

    # 5) Hyper-parameter tuning
    features = ['population', 'median_income', 'pop_density', 'pr', 'pet', 'tmmn', 'tmmx']
    name = 'keras'
    model = ''
    model_param = {}

    # initialize learner
    wi_case = Learner(data, features=features, target='wu_rate')

    # add more features
    wi_case.feature_engineering(feat_chain=[])

    # choose ML estimator, defualt is xgboost
    wi_case.add_model(model=('keras', XGBRegressor()),
                      parameters={})

    ## --------------------Transform Example ----------------------------
    from sklearn.base import BaseEstimator, TransformerMixin


    class TranFunc(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X_ = X.copy()
            X_.X2 = X_.X2 + 1
            return X_


    ## ---------------------------------------------------------------------

    wi_case.add_feature_transform(transform=('shift', TranFunc()),
                                  parameters={'scale': 0})

    ## -------------- Target transform --------------------------------

    from sklearn.compose import TransformedTargetRegressor

    tt = TransformedTargetRegressor(regressor=XGBRegressor(),
                                    func=np.log, inverse_func=np.exp)

    ## -----------------------------------------------------------------
    wi_case.add_target_transform(func=np.log, inverse_func=np.exp)

    md = wi_case.generate_model()
    Y_hat = Learner.predict(X_hat)

################3

# date, pop, climate1, wu_daily, montly, annual
