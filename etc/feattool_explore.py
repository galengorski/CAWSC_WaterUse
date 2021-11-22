import os, sys
import featuretools as ft
import configparser
import warnings
import pandas as pd
import numpy as np


# Load training data
warnings.filterwarnings('ignore')

config = configparser.ConfigParser()
config.read(r"C:\work\water_use\ml_experiments\annual_v_0_0\config_file.ini")

workspace = config.get("Files", "Workspace")
train_file = config.get("Files", "Train_file")
target_name = config.get("Target", "target_field")

feature_info_df = pd.read_excel(os.path.join(workspace, config.get('Features', 'features_info_file')))

def get_feature_type(feature_info_df, feature_list):

    feat_types = {}
    for feat in feature_list:
        mask = feature_info_df['Feature_name'].isin([feat])
        typ = feature_info_df.loc[mask, "Type"].values[0]
        if "number" in typ:
            feat_types[feat] = ft.variable_types.Numeric
        elif "bool" in typ:
            feat_types[feat] = ft.variable_types.Boolean
        elif "categorical" in typ:
            feat_types[feat] = ft.variable_types.Categorical
        else:
            raise Warning("unknown type")
            feat_types[feat] = "unknown"

    return feat_types


feature_mask = ~((feature_info_df['Skip'] == 1) | (feature_info_df['Not_Feature'] == 1))
features = feature_info_df[feature_mask]['Feature_name'].values.tolist()

df_main = pd.read_csv(train_file)
df_train = df_main[df_main['wu_rate'] > 0]

#
df = df_train[features]
wsys_id = df_train['sys_id']
#df['sys_id'] = wsys_id
df.fillna(0, inplace = True)
target = df[target_name]

es = ft.EntitySet(id = 'annual_wu')
variable_types = get_feature_type(feature_info_df, df.columns)
del(variable_types[target_name])
es = es.entity_from_dataframe(entity_id = 'df', dataframe = df.drop([target_name], axis=1),
                              variable_types =variable_types, index = 'id',  make_index=True)

features, feature_names = ft.dfs(entityset = es,
                                 target_entity = 'df',
                                 trans_primitives = ["county_id"],
                                 max_depth = 2)
xx = 1