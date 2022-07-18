import os
import copy

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
    preprocessor = ColumnTransformer(
        remainder='passthrough',
        transformers=[
            #("cat", categorical_transformer, categorical_features),
            ('Target encoding', ce.quantile_encoder.SummaryEncoder(cols=categorical_features, quantiles=(0.05, 0.25, 0.75, 0.5, 0.95)), categorical_features)
        ])
    main_pipeline.append(('preprocess', preprocessor))

    return main_pipeline
