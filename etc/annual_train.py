import os, sys
import pandas as pd
import numpy as np
import dask
import dask.dataframe as pdd
from dask.distributed import Client
import xgboost as xgb
import dask_ml
from dask_ml.model_selection import train_test_split
import dask_xgboost
import matplotlib.pyplot as plt
# ===============================================
#
# ===============================================
if __name__ == '__main__':
    client = Client(n_workers=1, threads_per_worker=1)

    db_root = r"C:\work\water_use\mldataset\ml\training\features"
    huc2_folders = os.listdir(db_root)
    wu = []
    for huc2_folder in huc2_folders:
        fn = os.path.join(db_root, os.path.join(huc2_folder, "assemble"))
        fn = os.path.join(fn, "train_db_{}.csv".format(huc2_folder))
        wu_ = pdd.read_csv(fn)
        wu.append(wu_)

    wu = pdd.concat(wu)
    wu = wu.dropna()
    wu['wu_per_capita'] = wu['wu_rate']/(365*wu['population'])
    wu = wu[ wu['wu_per_capita']<1000]
    wu = wu[wu['wu_per_capita'] >30]
    delay = False
    if delay:
        wu = wu.compute()
    #wu = wu[wu['wu_per_capita'] > 20]
    y = wu['wu_rate']
    X = wu[['population', 'households2', 'median_income',
       'pop_density', 'etr', 'pr',
       'tmmn', 'tmmx']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 123)
    # *****************************************************
    if delay:
        #data_dmatrix = xgb.DMatrix(data=X, label=y)  # 'objective': 'binary:logistic',
        gb = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.01,
                              max_depth=5, alpha=0.1, n_estimators=1000, rate_drop=0.9, skip_drop=0.5)
        gb.fit(X_train, y_train)

        ypredict = gb.predict(X_test)
        roh = np.corrcoef(y_test.squeeze(), ypredict.squeeze())[0, 1]
        roh = int(roh * 1000) / 1000.0
        plt.scatter(y_test, ypredict)
        # plt.plot([0, 6000], [0, 6000], 'r')
        plt.title(str(roh))
        plt.show()
# ************************************************************************************************
    else:
        params = {'objective': 'reg:squarederror', 'eval_metric':["error", "rmse"]} #, 'booster':'gbtree'
        params = {'objective':'reg:squarederror', 'colsample_bytree':0.3, 'learning_rate':0.01,
                                  'max_depth':7, 'alpha':0.1, 'n_estimators':1000, 'rate_drop':0.9, 'skip_drop':0.5}
        #params['booster'] = 'dart' # become very slow
        params['verbosity'] = 2 # print information
        #params['min_split_loss'] = 10e10
        #params['min_child_weight '] = 0
        params['subsample'] = 0.5

        bst = dask_xgboost.train(client, params, X_train, y_train,  num_boost_round=2000)
        y_hat = dask_xgboost.predict(client, bst, X_test).persist()
        y_test = y_test.compute()
        y_hat = y_hat.compute()
        plt.scatter(y_test, y_hat)
        plt.plot([[min(y_test),  max(y_test)],[min(y_test), max(y_test)]])

        import xgboost as xgb

        dtest = xgb.DMatrix(X_test.head())
        bst.predict(dtest)

        from sklearn.metrics import r2_score

        accuracy = r2_score(y_test, y_hat)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        plt.title(accuracy)
        plt.show()

    xx = 1
