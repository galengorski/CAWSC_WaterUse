import os, sys
import pandas as pd
import numpy as np
import dask
import dask.dataframe as pdd
from dask.distributed import Client
import xgboost
import dask_ml
from dask_ml.model_selection import train_test_split
import dask_xgboost
import matplotlib.pyplot as plt

# ===============================================
#
# ===============================================
if __name__ == "__main__":
    client = Client(n_workers=1, threads_per_worker=1)

    wu = pdd.read_csv(
        r"C:\work\water_use\dataset\dailywu\pa\pa_master_training.csv"
    )
    wu = wu.dropna()

    y = wu["QUANTITY"]
    X = wu[
        [
            "population",
            "pop_density",
            "median_income",
            "pr",
            "pet",
            "tmmn",
            "tmmx",
        ]
    ]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": ["error", "logloss"],
    }

    bst = dask_xgboost.train(
        client, params, X_train, y_train, num_boost_round=50
    )
    y_hat = dask_xgboost.predict(client, bst, X_test).persist()
    y_test = y_test.compute()
    y_hat = y_hat.compute()
    plt.scatter(y_test, y_hat)
    plt.plot([[min(y_test), max(y_test)], [min(y_test), max(y_test)]])

    import xgboost as xgb

    dtest = xgb.DMatrix(X_test.head())
    bst.predict(dtest)

    from sklearn.metrics import r2_score

    accuracy = r2_score(y_test, y_hat)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    plt.title(accuracy)

    xx = 1
