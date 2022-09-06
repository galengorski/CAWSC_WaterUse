import pandas as pd
from numpy import sqrt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import tensorflow as tf

tf.random.set_seed(11)
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from matplotlib import pyplot as plt

# load dataset
df_feat = pd.read_csv(
    r"C:\work\water_use\ml_experiments\annual_v_0_0\wu_annual_training.csv"
)

cols_to_del = [
    "is_swud",
    "Num_establishments_2017",
    "NoData",
    "Ecode",
    "sys_id",
    "fips",
]
cols_to_del2 = [
    "population",
    "swud_pop",
    "small_gb_pop",
    "pc_swud",
    "pc_tract_data",
    "pop_swud_corrected",
    "swud_corr_factor",
    "pc_swud_corrected",
    "pc_gb_data",
    "pop_swud_gb_correction",
    "pc_swud_gb_corrected",
    "bg_pop_2010",
    "bg_usgs_correction_factor",
    "bg_theo_correction_factor",
    "ratio_lu",
    "pop_urb",
    "ratio_2010",
    "swud_pop_ratio",
]
cat_feat = ["Ecode_num", "HUC2", "county_id"]

# for now delete all
for f in cols_to_del:
    del df_feat[f]
for f in cols_to_del2:
    del df_feat[f]
for f in cat_feat:
    del df_feat[f]


df_feat = df_feat[df_feat["wu_rate"] > 0]


##### outliears
df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
df_feat.dropna(axis=1, how="all")
df_feat = df_feat.dropna(axis=0)
if 0:

    from sklearn.ensemble import IsolationForest
    import numpy as np

    X = df_feat.values
    clf = IsolationForest(n_estimators=10, warm_start=True)
    clf.fit(X)  # fit 10 trees
    clf.set_params(n_estimators=20)  # add 10 more trees
    mask1 = clf.fit_predict(X)  # fit the added trees
################33
if 0:
    from sklearn.neighbors import LocalOutlierFactor

    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    # use fit_predict to compute the predicted labels of the training samples
    # (when LOF is used for outlier detection, the estimator has no predict,
    # decision_function and score_samples methods).
    X = df_feat.values
    mask2 = clf.fit_predict(X)
    ##################333
    XX = df_feat[
        [
            "wu_rate",
            "LUpop_Swudpop",
            "median_income",
            "median_h_year",
            "tot_h_age",
            "Commercial",
            "Conservation",
            "Domestic",
            "Industrial",
            "Institutional",
            "Recreation_Misc",
            "Urban_Misc",
            "Production",
        ]
    ]
    XX = XX.values

    from sklearn.svm import OneClassSVM

    clf = OneClassSVM(gamma="auto")
    flg = clf.fit_predict(XX)

    # mask3 = clf.score_samples(X)


df_feat = df_feat[df_feat["LUpop_Swudpop"] > 5000]


pc = df_feat["wu_rate"] / df_feat["LUpop_Swudpop"]
# df_feat['LUpop_Swudpop'] = np.log10(df_feat['LUpop_Swudpop'])
df_feat = df_feat[pc < pc.quantile(0.90)]
df_feat = df_feat[pc > pc.quantile(0.1)]
df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
df_feat.dropna(axis=1, how="all")
df_feat = df_feat.dropna(axis=0)


pc = df_feat["wu_rate"] / df_feat["LUpop_Swudpop"]


# split into input and output columns
y = df_feat["wu_rate"].values
# y = pc.values
# y = np.log10(df_feat['wu_rate'].values)
del df_feat["wu_rate"]
X = df_feat.values
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=123, test_size=0.33
)
scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
act = "relu"
# act = tf.keras.layers.LeakyReLU(alpha=0.01)

model.add(
    Dense(
        50,
        activation=act,
        kernel_initializer="he_normal",
        input_shape=(n_features,),
    )
)
model.add(Dense(25, activation=act, kernel_initializer="he_normal"))
# model.add(Dropout(rate=0.2))
model.add(Dense(12, activation=act, kernel_initializer="he_normal"))
model.add(Dense(6, activation=act, kernel_initializer="he_normal"))
model.add(Dense(1, activation=act))
# compile the model
loss = tf.keras.losses.MeanSquaredLogarithmicError()
# loss = 'mse'
model.compile(optimizer="adam", loss=loss)
# fit the model
history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    verbose=2,
)


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["loss", "val"], loc="upper left")


# evaluate the model
error = model.evaluate(X_test, y_test, verbose=2)
print("MSE: %.3f, RMSE: %.3f" % (error, sqrt(error)))
# make a prediction

yhat = model.predict([X_test])
plt.figure()
plt.scatter(y_test, yhat)
from sklearn.metrics import r2_score

accuracy = r2_score(y_test, yhat)
c = [min(y_test), max(y_test)]
plt.plot(c, c, "r")
plt.title(str(accuracy))
plt.show()
# print('Predicted: %.3f' % yhat)
Pxx = 1
