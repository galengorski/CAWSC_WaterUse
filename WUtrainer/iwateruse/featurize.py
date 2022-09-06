import os, sys
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
import category_encoders as ce

try:
    import tsfel
except:
    pass


def summary_encode2(model, cols, quantiles=None):

    sum_ce = ce.quantile_encoder.SummaryEncoder(
        cols=cols, quantiles=quantiles, handle_missing="return_nan"
    )

    df_ = model.df_train.copy()
    target = model.target
    cat_df = df_[["sample_id"] + cols]
    df_["cleaned_target"] = df_[target].copy()
    df_.loc[df_["cleaned_target"] < 25, "cleaned_target"] = np.NAN
    df_.loc[df_["cleaned_target"] > 500, "cleaned_target"] = np.NAN
    df_trans = sum_ce.fit_transform(df_, df_["cleaned_target"])
    del df_trans["cleaned_target"]
    # generate_summary of target encoding
    temp = df_trans.merge(cat_df, how="left", on="sample_id")
    feat_encod_map = {}
    for col in cols:
        nfeat = []
        for q in quantiles:
            nm = col + "_" + str(int(100 * q))
            nfeat.append(nm)
        feat_encod_map[col] = nfeat

    encoding_summary = {}
    for group in cols:
        scol = feat_encod_map[group] + [group]
        ddf_ = temp[scol].groupby(by=group).mean()
        encoding_summary[group] = ddf_.to_dict()
    return temp, encoding_summary


def summary_encode(
    model, cols, quantiles=None, max_target=500, min_target=25, min_pop=1000
):
    """

    :param model:
    :param cols:
    :param quantiles:
    :return:
    """
    df_ = model.df_train.copy()
    target = model.target
    df_["target2"] = df_[target].copy()
    default_value = df_["target2"].mean()
    df_.loc[df_["target2"] > max_target, "target2"] = np.NAN
    df_.loc[df_["target2"] < min_target, "target2"] = np.NAN
    df_.loc[df_["pop"] < min_pop, "target2"] = np.NAN
    for feat in cols:
        for q in quantiles:
            nm = feat + "_" + "smc_" + str(int(q * 100))
            df2 = df_[[feat, "target2"]].groupby(feat).quantile(q)
            df2.reset_index(inplace=True)
            df2.loc[df2["target2"].isna(), "target2"] = default_value
            df2.rename(columns={"target2": nm}, inplace=True)
            df_ = df_.merge(df2, how="left", on=feat)
    del df_["target2"]
    return df_


def summary_encode_pred(model, cols, df_pred):
    sum_ce = ce.quantile_encoder.SummaryEncoder(
        cols=cols, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    target = model.target
    df_trans = sum_ce.fit_transform(df_pred, df_pred[target])
    return df_trans


class MultiOneHotEncoder(TransformerMixin):
    #

    def __init__(self, catfeatures=None):
        """catfeaturesis a list"""
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
            X_.drop(feature, axis=1, inplace=True)
        X_ = [X_] + all_cat_feats
        X_ = pd.concat(X_, axis=1)
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
        mask = feature_info_df["Feature_name"].isin([feat])
        typ = feature_info_df.loc[mask, "Type"].values[0]
        if "number" in typ:
            feat_types[feat] = "number"
        elif "bool" in typ:
            feat_types[feat] = "bool"
        elif "categorical" in typ:
            feat_types[feat] = "categorical"
        else:
            raise Warning("unknown type")
            feat_types[feat] = "unknown"

    return feat_types


def add_dayOfweek(X):
    X_ = X.copy()
    X_["DoW"] = X_["Date"].dt.dayofweek
    return X_


def add_holidays(X):
    X_ = X.copy()
    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

    start_date = X_["Date"].min()
    end_date = X_["Date"].max()
    cal = calendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    X_["Holiday"] = X_["Date"].isin(holidays)
    X_["Holiday"] = X_["Holiday"].astype("int")
    return X_


def add_per_capita_wu(X):
    X_ = X.copy()
    X_["per_capita_wu"] = X_["wu_rate"] / X_["population"]
    return X_


def drop_na_features(X, y):
    xx = 1
    pass


def drop_na_targets(X, y):
    pass


def hot_encode(X, y, cat_features):
    all_cat_feats = []
    for feature in cat_features:
        ft_df = pd.get_dummies(X[feature], prefix=feature)
        all_cat_feats.append(ft_df.copy())
        del X[feature]
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

    model = TransformedTargetRegressor(
        regressor=pipe,
        func=target_transform,
        inverse_func=inverse_target_transform,
        check_inverse=False,
    )
    return model


def add_moving_average_to_dataset(dataset, pc_stat):
    df_ = pc_stat[
        [
            "sys_id",
            "pop_tmean",
            "pop_median",
            "swud_tmean",
            "swud_median",
            "tpop_tmean",
            "tpop_median",
        ]
    ]
    dataset = dataset.merge(
        df_, right_on=["sys_id"], left_on=["sys_id"], how="left"
    )
    del df_
    return dataset
