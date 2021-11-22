import os, sys
import pandas as pd
import numpy as np
import configparser
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import FunctionTransformer

def change_nan_value(df, features, value):
    """
    Just to test using custom transnformer
    :param df:
    :param features:
    :param value:
    :return:
    """
    def fun():
        pass

    pass

def summarize_missing_data (df = None, target= '', plot = False):
    features = df.columns
    features = features.to_list()
    features.remove(target)
    missing_data = df[features].isna().sum().reset_index(name="n")
    if plot:
        plt.Figure()
        missing_data.plot.bar(x='index', y='n', rot=45)
    missing_df = pd.DataFrame(missing_data)
    missing_df = missing_df.sort_values(by=['n'], ascending=False, inplace=False)
    missing_df = missing_df[missing_df['n'] > 0]
    missing_df['% missing'] = 100 * missing_df['n'] / len(df)
    return missing_df

def fill_missing_annual_climate_data(df, sys_id_field, climate_variables, x_field, y_field):
    """
    Fill using closest service area
    """

    for var in climate_variables:
        isnan_mask = df[var].isna()
        sys_ids = df[isnan_mask][sys_id_field].unique()
        for sys_id in sys_ids:
            sys_id_mask = df[sys_id_field] == sys_id
            mean_val = df[sys_id_mask][var].mean()
            if pd.isna(mean_val):
                sys_x = df.loc[sys_id_mask, x_field].mean()
                sys_y =  df.loc[sys_id_mask, y_field].mean()
                delx = df[x_field] - sys_x
                dely = df[y_field] - sys_y
                dist = np.power(((delx**2.0) + dely**2.0 ), 0.5)

            else:
                df.loc[sys_id_mask, var] = mean_val

def fillna_mean(df, col):
    df[col].fillna(df[col].mean(), inplace=True)

def fillna_media(df, col):
    df[col].fillna(df[col].mean(), inplace=True)

def fillna_most_commen(df, col):
    most_common_category = df[col].value_counts().index[0]
    df[col].fillna(most_common_category, inplace=True)


def drop_sample():
    pass

def drop_column(list):
    pass


def fill_missing_data_using_xgboost(df_in, impute_col, target = None, is_categorical = False):

    df = df_in.copy()

    if not(target is None):
       df.drop(target, axis = 1, inplace = True)


    # index of nans
    col_nan_ix = df[df[impute_col].isnull()].index

    #split data
    col_test = df[df.index.isin(col_nan_ix)]
    col_train = df.drop(col_nan_ix, axis=0)

    if is_categorical:
        model = XGBClassifier()
        model.fit(col_train.drop(impute_col, axis=1), col_train[impute_col])
    else:
        model = XGBRegressor()
        model.fit(col_train.drop(impute_col, axis=1), col_train[impute_col])

    # predict
    pred_col = model.predict(col_test.drop(impute_col, axis=1))


    imputed_feature = df[impute_col].copy()
    imputed_feature[col_nan_ix] = pred_col

    return imputed_feature

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(r"C:\work\water_use\ml_experiments\annual_v_0_0\config_file.ini")

    workspace = config.get('Files', 'Workspace')
    annual_db = pd.read_csv(config.get('Files', 'Train_file'))
    target_col = config.get('Target', 'target_field')
    feature_info = pd.read_excel( os.path.join(workspace, config.get('Features', 'features_info_file')) )

    train_df = annual_db[annual_db[target_col]>0]

    features_to_drop = feature_info.loc[feature_info['Not_Feature']==1, 'Feature_name'].values.tolist()
    features_to_drop = features_to_drop + feature_info.loc[feature_info['Skip'] == 1, 'Feature_name'].values.tolist()
    features_to_drop.remove(target_col)

    summarize_missing_data( train_df, target = 'wu_rate')
    pass



