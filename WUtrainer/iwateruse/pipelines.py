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
    categorical_features = model.categorical_features
    categorical_features2 = copy.deepcopy(categorical_features)

    for feat in categorical_features2:
        if not (feat in model.df_train.columns):
            categorical_features.remove(feat)
        elif (feat in model.columns_to_drop):
            categorical_features.remove(feat)
        else:
            model.df_train[feat] = model.df_train[feat].astype("category")


    main_pipeline = []

    # Transformation of categorical features
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ('Target encoding', ce.OneHotEncoder(cols=categorical_features), categorical_features)
        ])
    main_pipeline.append(('preprocess', preprocessor))

    # # drop r2 to 0.774
    # if 0:
    #     from feature_engine.encoding import CountFrequencyEncoder
    #     encoder = CountFrequencyEncoder(encoding_method='frequency',
    #                                     variables=categorical_features)
    #     main_pipeline.append(('enc', encoder))
    #
    # # drop r2 to 0.663 for ordered and 0.78 for arbitrary
    # if 0:
    #     from feature_engine.encoding import OrdinalEncoder
    #     encoder = OrdinalEncoder(encoding_method='ordered',
    #                              variables=categorical_features)
    #     main_pipeline.append(('enc', encoder))

    return main_pipeline
