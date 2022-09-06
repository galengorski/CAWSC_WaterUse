import os
import copy

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import category_encoders as ce


def make_pipeline(model):
    categorical_features = copy.deepcopy(model.categorical_features)

    categorical_features2 = []
    for feat in categorical_features:
        if feat in model.df_train.columns:
            categorical_features2.append(feat)

    for feat in categorical_features2:
        model.df_train[feat] = model.df_train[feat].astype("category")

    main_pipeline = []

    # Transformation of categorical features
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    summary_encoding = ce.quantile_encoder.SummaryEncoder(
        cols=categorical_features, quantiles=(0.05, 0.25, 0.75, 0.5, 0.95)
    )
    preprocessor = ColumnTransformer(
        remainder="passthrough",
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            # ('Target encoding', summary_encoding, categorical_features)
        ],
    )
    main_pipeline.append(("preprocessor", preprocessor))
    return main_pipeline


class Spipe(object):
    def __init__(self, pipeline=None):
        """

        :param pipeline:
        must have two steps: preprocess and estimator
        """
        self.pipeline = pipeline

        if len(pipeline.steps) != 2:
            raise ValueError(
                "Only two steps are allowed: preprocessor ans estimator"
            )

        try:
            self.preprocessor = pipeline["preprocessor"]
        except:
            raise ValueError(
                " The pipeline must have a first step called 'preprocessor'"
            )

        try:
            self.estimator = pipeline["estimator"]
        except:
            raise ValueError(
                " The pipeline must have a first step called 'estimator'"
            )

        self.df_in = None
        self.df_out = None

    def fit(self, X, y):

        X_ = self.preprocess(X, y)
        est_obj = self.estimator.fit(X_, y)
        self.trained_model = est_obj
        return self.trained_model

    def predict(self, X):
        col1 = X.columns[0]
        y_dummy = X[[col1]].copy() * np.NAN
        y_dummy.rename(columns={col1: "0"}, inplace=True)
        X_ = self.preprocess(X, y=y_dummy)
        yhat = self.trained_model.predict(X_)
        return yhat

    def preprocess(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        X_ = self.preprocessor.fit_transform(X, y)
        features_out = self.preprocessor.get_feature_names_out()
        X_ = pd.DataFrame.sparse.from_spmatrix(X_, columns=features_out)

        # clean columns
        cols = X_.columns
        for col in cols:
            if "remainder" in col:
                ncol = col.replace("remainder__", "")
                X_.rename(columns={col: ncol}, inplace=True)
        self.X_transf = X_
        return X_
