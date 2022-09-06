import os, sys
import pandas as pd
from pycaret.regression import *

sys.path.append(r"C:\work\water_use\CAWSC_WaterUse\WUtrainer")
from featurize import MultiOneHotEncoder
from pycaret.utils import check_metric
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

target = "wu_rate"

dataset = pd.read_csv(
    r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv"
)
pop_info = pd.read_csv(r"pop_info.csv")

Qdataset = pd.read_csv(
    r"C:\work\water_use\ml_experiments\annual_v_0_0\clean_train_db.csv"
)
Qpop_info = pd.read_csv(r"pop_info.csv")
Qpc_50_swud = pd.read_csv(
    r"C:\work\water_use\mldataset\ml\training\misc_features\spatial_features\pc_50_pop_swud15.csv"
)
Qpc_50_plc = pd.read_csv(
    r"C:\work\water_use\mldataset\ml\training\misc_features\spatial_features\pc_50_plc_pop_interpolated.csv"
)

Qdf_ = Qpc_50_swud[["sys_id", "pc_median"]]
Qdataset = dataset.merge(
    Qdf_, right_on=["sys_id"], left_on=["sys_id"], how="left"
)
del Qdf_

# pop_info
pop_info["pop"] = pop_info["pop_swud16"].copy()
mask = pop_info["pop"].isna() | pop_info["pop"] == 0
pop_info.loc[mask, "pop"] = pop_info[mask]["plc_pop_interpolated"]
mask = pop_info["pop"].isna() | pop_info["pop"] == 0
pop_info.loc[mask, "pop"] = pop_info[mask]["TPOPSRV"]
mask = pop_info["pop"].isna() | pop_info["pop"] == 0
pop_info.loc[mask, "pop"] = pop_info[mask]["tract_pop"]

Qpop_info["pop"] = Qpop_info["pop_swud16"].copy()
Qmask = Qpop_info["pop"].isna() | Qpop_info["pop"] == 0
Qpop_info.loc[Qmask, "pop"] = Qpop_info[Qmask]["plc_pop_interpolated"]
Qmask = Qpop_info["pop"].isna() | Qpop_info["pop"] == 0
Qpop_info.loc[Qmask, "pop"] = Qpop_info[Qmask]["TPOPSRV"]
Qmask = Qpop_info["pop"].isna() | Qpop_info["pop"] == 0
Qpop_info.loc[Qmask, "pop"] = Qpop_info[Qmask]["tract_pop"]


# dataset = dataset[dataset['Ecode_num']==0]

pop_df = pop_info[["sys_id", "pop", "Year"]]
dataset = dataset.merge(
    pop_df, right_on=["sys_id", "Year"], left_on=["sys_id", "Year"], how="left"
)

Qpop_df = Qpop_info[["sys_id", "pop", "Year"]]
Qdataset = Qdataset.merge(
    Qpop_df,
    right_on=["sys_id", "Year"],
    left_on=["sys_id", "Year"],
    how="left",
)


# pop_df = pop_info[['sys_id', 'pop', 'Year']]
# 'sys_id' in dataset.columns
# dataset = dataset.merge(pop_df, right_on=['sys_id', 'Year'], left_on=['sys_id', 'Year'] , how = 'left')

# categorical transformation
categorical_features = ["HUC2", "state_id", "KG_climate_zone"]  # , 'county_id'
ohc1 = MultiOneHotEncoder(catfeatures=categorical_features)
dataset = ohc1.transform(dataset)
Qdataset = ohc1.transform(Qdataset)

columns_to_drop = ["population", "sys_id", "pc"]


# del(dataset['pop'])
dataset = dataset[dataset["pop"] > 100]
dataset["pc"] = np.log10(dataset["wu_rate"] / dataset["pop"])
# del(dataset['wu_rate'])
if True:
    dataset["pop"] = np.log10(dataset["pop"])
    dataset["wu_rate"] = np.log10(dataset["wu_rate"])

dataset = dataset[dataset["pc"] < dataset["pc"].quantile(0.98)]
dataset = dataset[dataset["pc"] > dataset["pc"].quantile(0.05)]
dataset = dataset.drop(columns_to_drop, axis=1)

# columns to drop
Qdf = Qdataset.copy()
Qdf["pc"] = Qdf["wu_rate"] / Qdf["pop"]
Qdf = Qdf[Qdf["pop"] > 100]
Qmask = (Qdf["pc"] > 20) & (Qdf["pc"] < 500)
Qdf = Qdf[Qmask]

Qdf = Qdf.drop(columns_to_drop, axis=1)
features = list(Qdf.columns)
features.remove("wu_rate")
QX = Qdf[features]
Qy = Qdf["wu_rate"]

QX_train, QX_test, Qy_train, Qy_test = train_test_split(
    QX, Qy, test_size=0.2, random_state=123
)
QX["pop"] = np.log10(QX["pop"])


data = dataset.sample(frac=0.8, random_state=123)  # 786
data_unseen = dataset.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print("Data for Modeling: " + str(data.shape))
print("Unseen Data For Predictions: " + str(data_unseen.shape))

exp_reg101 = setup(
    data=data,
    target="wu_rate",
    train_size=0.7,
    # categorical_features =categorical_features,
    fold_shuffle=True,
    data_split_shuffle=True,
    session_id=123,
    # normalize = True, transformation = True,
    # combine_rare_levels = True, rare_level_threshold = 0.05,
    # remove_outliers = True, outliers_threshold = 0.05,
    # pca = True, pca_method = 'kernel', pca_components = 50,
    # create_clusters = True, cluster_iter = 20
    # polynomial_features = True, polynomial_degree = 2,
    # feature_interaction = True, feature_ratio = True,
    # use_gpu = True
)
xgb1 = create_model("xgboost", fold=3)
# plot_model(xgb1)
# plot_model(xgb1, plot = 'error')

plt.figure()
unseen_predictions = predict_model(xgb1, data=data_unseen)
unseen_predictions
r2 = check_metric(
    unseen_predictions.wu_rate / unseen_predictions["pop"],
    unseen_predictions.Label / unseen_predictions["pop"],
    "R2",
)
plt.scatter(
    10 ** (unseen_predictions.wu_rate / unseen_predictions["pop"]),
    10 ** (unseen_predictions.Label / unseen_predictions["pop"]),
)
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.plot([20, 300], [20, 300], "r")
plt.title("Unseen Data -- $R^2$ = {}".format(r2))

xx = 1
