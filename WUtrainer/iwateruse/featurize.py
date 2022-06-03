import os, sys
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

try:
    import tsfel
except:
    pass


class MultiOneHotEncoder(TransformerMixin):
    #

    def __init__(self, catfeatures = None):
        """ catfeaturesis a list"""
        self.catfeatures = catfeatures

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.catfeatures is None:
            self.catfeatures = X.columns
        X_ = X.copy()
        all_cat_feats = []
        for feature in self.catfeatures:
            ft_df = pd.get_dummies(X_[feature], prefix=feature)
            all_cat_feats.append(ft_df.copy())
            X_.drop(feature, axis = 1, inplace = True)
        X_ = [X_] + all_cat_feats
        X_= pd.concat(X_, axis=1)
        return X_




def get_feature_type(feature_info_df, features):
    """

    :param feature_info_df: data frame with feature info
    :param features: a list of features to get type of
    :return: dictionary  {feature_name: number}
     types = [bool, number, category]
    """
    feat_types = {}
    for feat in features:
        mask = feature_info_df['Feature_name'].isin([feat])
        typ = feature_info_df.loc[mask, "Type"].values[0]
        if "number" in typ:
            feat_types[feat] = 'number'
        elif "bool" in typ:
            feat_types[feat] = 'bool'
        elif "categorical" in typ:
            feat_types[feat] = 'categorical'
        else:
            raise Warning("unknown type")
            feat_types[feat] = "unknown"

    return feat_types


def add_dayOfweek(X):
    X_ = X.copy()
    X_['DoW'] = X_['Date'].dt.dayofweek
    return X_

def add_holidays(X):
    X_ = X.copy()
    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
    start_date = X_['Date'].min()
    end_date = X_['Date'].max()
    cal = calendar()
    holidays = cal.holidays(start = start_date, end = end_date)
    X_['Holiday'] = X_['Date'].isin(holidays)
    X_['Holiday'] = X_['Holiday'].astype('int')
    return X_

def add_per_capita_wu(X):
    X_ = X.copy()
    X_['per_capita_wu'] = X_['wu_rate']/X_['population']
    return X_

def drop_na_features(X, y):
    xx = 1;
    pass

def drop_na_targets(X, y):
    pass


def hot_encode(X, y, cat_features):
    all_cat_feats = []
    for feature in cat_features:
        ft_df = pd.get_dummies(X[feature], prefix = feature)
        all_cat_feats.append(ft_df.copy())
        del(X[feature])
    all_cat_feats = pd.concat(all_cat_feats)

def Log10_target_transform(pipe):
    """
    Take a feature pipe line and target transform func and inv_func, and return a model
    :param y:
    :param pipe:
    :return:
    """
    def target_transform(y):
        y_ = np.log10(y.copy())
        return y_

    def inverse_target_transform(y):
        y_ = np.power(10.0, y.copy())
        return y_

    from sklearn.compose import TransformedTargetRegressor
    model = TransformedTargetRegressor(regressor=pipe,
                                       func= target_transform,
                                       inverse_func= inverse_target_transform,
                                       check_inverse=False
                                       )
    return model



def add_moving_average_to_dataset(dataset, pc_stat):
    df_ = pc_stat[['sys_id', 'pop_tmean', 'pop_median',
                   'swud_tmean', 'swud_median', 'tpop_tmean', 'tpop_median']]
    dataset = dataset.merge(df_, right_on=['sys_id'], left_on=['sys_id'], how='left')
    del (df_)
    return dataset
