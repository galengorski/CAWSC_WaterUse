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
from xgboost import plot_importance
# ===============================================
#
# ===============================================
if __name__ == '__main__':
    client = Client(n_workers=1, threads_per_worker=1)
    db_root = r"C:\work\water_use\mldataset\ml\training\features"
    huc2_folders = os.listdir(db_root)
    wu = []

    wu = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\train_datasets\Monthly\wu_monthly_training.csv")
    df_feat = wu[wu['wu_rate'] > 0]
    SumMonthly = df_feat.groupby(by=['YEAR', 'sys_id']).sum()
    SumMonthly.reset_index(inplace=True)
    SumMonthly = SumMonthly[['YEAR', 'sys_id', 'wu_rate']]
    SumMonthly['wu_rate_annual'] = SumMonthly['wu_rate']
    del (SumMonthly['wu_rate'])
    df_feat = df_feat.merge(SumMonthly, left_on=['sys_id', 'YEAR'], right_on=['sys_id', 'YEAR'], how='left')
    df_feat['month_frac'] = df_feat['wu_rate'] / df_feat['wu_rate_annual']

    wu_annual = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\wu_annual_training.csv")
    wu_annual = wu_annual.groupby(by=['sys_id', 'Year']).mean()
    wu_annual = wu_annual[['pop_swud_corrected', 'median_income',  'h_age_newer_2005',
                           'h_age_2000_2004', 'h_age_1990_1999', 'h_age_1980_1989',
                           'h_age_1970_1979', 'h_age_1960_1969', 'h_age_1950_1959',
                           'h_age_1940_1949', 'h_age_older_1939', 'pop_density']]
    wu_annual.reset_index(inplace=True)
    df_feat = df_feat.merge(wu_annual, left_on=['sys_id', 'YEAR'], right_on=['sys_id', 'Year'], how='left')
    ### ****************


    df_feat['Aridity'] = df_feat['etr'] / (1+df_feat['pr'])
    df_feat['Aridity2'] = df_feat['tmmx'] / (1 + df_feat['pr'])


    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.dropna(axis=0)

    df_feat['wu_per_capita'] = df_feat['wu_rate']/(df_feat['pop_swud_corrected'])
    #wu = wu[wu['population3']>3000]
    df_feat = df_feat[ df_feat['wu_per_capita']<1000]
    df_feat = df_feat[df_feat['wu_per_capita'] > 10]

    df_feat = df_feat[df_feat['HUC2'].isin([ 2, 5])]

    #df_feat['wu_rate'] = np.log10(df_feat['wu_rate'])
    #df_feat['wu_per_capita'] = np.log10(df_feat['wu_per_capita'])
    #df_feat['pop_swud_corrected'] = np.log10(df_feat['pop_swud_corrected'])
    #del(df_feat['wu_per_capita'])
    #wu = wu[wu['wu_per_capita']> wu['wu_per_capita'].quantile(0.05)]
    #wu = wu[wu['wu_per_capita']< wu['wu_per_capita'].quantile(0.95)]
    use_normal_xgb = True
    #if use_normal_xgb:
    #    wu = wu.compute()
    #wu = wu[wu['wu_per_capita'] > 20]☺
    #y = wu['wu_rate']
    df_feat = df_feat[df_feat['month_frac'] < 0.14]
    df_feat = df_feat[df_feat['month_frac'] > 0.05]
    #y = np.log10(df_feat['wu_rate'])
    y = np.log10(df_feat['month_frac'])
    X = df_feat.copy()
    one_hot = pd.get_dummies(X['HUC2'])
    X = X.drop('HUC2', axis=1)
    # Join the encoded df
    X = X.join(one_hot)
    del(X['wu_per_capita'])
    del (X['wu_rate'])
    del(X['month_frac'])
    del(X['sys_id'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 123)
    # *****************************************************
    if 0:
        from sklearn.ensemble import RandomForestRegressor

        regr = RandomForestRegressor(max_depth=4, random_state=0, n_estimators=5000,  n_jobs=-1)
        regr.fit(X_train, y_train)
        feat_labels = X.columns
        importance_df = []
        for feature in zip(feat_labels, regr.feature_importances_):
            importance_df.append([feature[0], feature[1]])
        importance_df = pd.DataFrame(importance_df)
        X_test2 = X_test
        ypredict = regr.predict(X_test2)
        roh = np.corrcoef(y_test.squeeze(), ypredict.squeeze())[0, 1]

    if use_normal_xgb:
        if 0:
            importance_df = importance_df.sort_values(by=[1], ascending=False)
            top_features = importance_df.iloc[:20][0].values.tolist()
            X = df_feat.copy()#[top_features]
            y = df_feat['wu_rate']
            try:
                del (X['wu_rate'])
                del (X['wu_per_capita'])
            except:
                pass
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        if 0:
            from keras.models import Sequential
            from keras.layers import Dense
            from keras.optimizers import Adam

            model = Sequential()
            model.add(Dense(50, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
            model.add(Dense(50, activation='relu'))
            model.add(Dense(50,  activation="relu"))
            model.add(Dense(50, activation="relu"))
            model.add(Dense(1, activation="sigmoid"))

            opt = Adam(lr=1e-3, decay=1e-3 / 200)
            model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
            print("[INFO] training model...")
            model.fit(x=X_train, y=y_train,
                      validation_data=(X_test, y_test),
                      epochs=10, batch_size=50)
            ypredict = model.predict(X_test)
            plt.scatter(y_test, ypredict)

        #data_dmatrix = xgb.DMatrix(data=X, label=y)  # 'objective': 'binary:logistic',
        from sklearn.metrics import r2_score
        gb = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.8, learning_rate=0.01,
                              max_depth=8, alpha=0, n_estimators=500, n_jobs=-1,
                               subsample=0.8,  seed = 123, reg_lambda=0.0) #tree_method = 'hist',reg_lambda=10,
        eval_set = [(X_train, y_train), (X_test, y_test)]
        gb.fit(X_train, y_train, eval_metric=["rmse","rmse"], eval_set=eval_set, verbose=True)

        ypredict = gb.predict(X_test)

        accuracy = r2_score(10**y_test, 10**ypredict)
        results = gb.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['rmse'], label='train')
        ax.plot(x_axis, results['validation_1']['rmse'], label='test')
        roh = np.corrcoef(y_test.squeeze(), ypredict.squeeze())[0, 1]
        roh = int(roh * 1000) / 1000.0
        plt.figure()
        plt.scatter(10**y_test, 10**ypredict)
        # plt.plot([0, 6000], [0, 6000], 'r')
        c = [min(10**y_test), max(10**y_test)]
        plt.plot(c, c,'r')
        plt.title("roh = {}, R2 = {} ".format(roh, accuracy))
        plt.figure()
        plot_importance(gb, xlabel='Gain', importance_type='gain', max_num_features = 20)


        gain_imp = gb.get_booster().get_score(importance_type='gain')
        gain_imp = pd.DataFrame(gain_imp.items()).sort_values(by=[1], ascending=True)
        gain_imp = gain_imp.iloc[-20:]
        fig, ax = plt.subplots()
        ax.barh(gain_imp[0], gain_imp[1], align='center')
        plt.tight_layout()


        plt.show()
# ************************************************************************************************
    else:
        params = {'objective': 'reg:squarederror', 'eval_metric':["error", "rmse"]} #, 'booster':'gbtree'
        params = {'objective':'reg:squarederror', 'colsample_bytree':0.3, 'learning_rate':0.1,
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
        c = [min(y_test), max(y_test)]
        plt.plot(c,c)

        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        plt.title(accuracy)
        plt.show()

    xx = 1
