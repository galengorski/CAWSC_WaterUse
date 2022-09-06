import sys, os
import numpy as np
import joblib
import xgboost as xgb
from lightgbm import LGBMRegressor


def xgb_estimator(params={}):
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "colsample_bytree": 0.8,
            "learning_rate": 0.20,
            "max_depth": 7,
            "alpha": 100,
            "n_estimators": 500,
            "rate_drop": 0.9,
            "skip_drop": 0.5,
            "subsample": 0.8,
            "reg_lambda": 10,
            "min_child_weight": 5,
            "gamma": 10,
            "max_delta_step": 0,
            "seed": 123,
        }
    results = {}
    gb = xgb.XGBRegressor(**params, evals_result=results)
    return gb


def lightGB_estimator(params):
    params = {
        "n_jobs": 1,
        "max_depth": 8,
        "min_data_in_leaf": 20,
        "subsample": 0.5,
        "n_estimators": 500,
        "learning_rate": 0.2,
        "colsample_bytree": 0.5,
        "boosting_type": "gbdt",
    }
    alpha = 0.5
    gb = LGBMRegressor(objective="regression", **params)
    return gb


def meta_model():
    pass

def get_model_from_file(model_file_name):
    trained_model = joblib.load(model_file_name)
    try:
        pfeatures = trained_model.get_booster().feature_names
    except:
        pfeatures = trained_model.steps[-1][1].get_booster().feature_names

    return trained_model, pfeatures
