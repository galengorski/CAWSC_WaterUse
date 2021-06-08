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
    read_file = True
    if read_file:
        wu = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\wu_annual_training.csv")
    else:
        for huc2_folder in huc2_folders:
            fn = os.path.join(db_root, os.path.join(huc2_folder, "assemble"))
            fn = os.path.join(fn, "train_db_{}.csv".format(huc2_folder))
            if os.path.isfile(fn):
                wu_ = pdd.read_csv(fn)
                wu.append(wu_)

        wu = pdd.concat(wu)
        wu_df = wu.compute()
        wu_df.to_csv(r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\wu_annual_training.csv")
        del(wu_df)
    ### ****************
    df_feat = wu[wu['wu_rate'] > 0]
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat[ (df_feat['pc_swud_corrected']>20) & (df_feat['pc_swud_corrected']<700) ]
    df_feat['ICITE'] =  (df_feat['awuds_totw_cnt'] - df_feat['awuds_dom_cnt'] )
    df_feat['pub_serv_ratio'] = df_feat['awuds_pop_cnt'] / df_feat['county_tot_pop_2010']
    df_feat['Aridity'] = df_feat['etr'] / (1+df_feat['pr'])

    feats = ['income_lt_10k',
             'income_10K_15k', 'income_15k_20k', 'income_20k_25k', 'income_25k_30k',
             'income_30k_35k', 'income_35k_40k', 'income_40k_45k', 'income_45k_50k',
             'income_50k_60k', 'income_60k_75k', 'income_75k_100k',
             'income_100k_125k', 'income_125k_150k', 'income_150k_200k',
             'income_gt_200k', 'median_income', 'h_age_newer_2005',
             'h_age_2000_2004', 'h_age_1990_1999', 'h_age_1980_1989',
             'h_age_1970_1979', 'h_age_1960_1969', 'h_age_1950_1959',
             'h_age_1940_1949', 'h_age_older_1939', 'pop_density', 'etr_warm',
             'etr_cool', 'etr', 'pr_warm', 'pr_cool', 'pr', 'tmmn_warm', 'tmmn_cool',
             'tmmn', 'tmmx_warm', 'tmmx_cool', 'tmmx', 'LAT', 'LONG',
             'Year', 'wu_rate', 'HUC2',  'pop_swud_corrected',
             'Ecode_num', 'pop_house_ratio', 'family_size', 'prc_hschool',
             'prc_diploma', 'prc_college', 'prc_high_edu', 'pov_2019', 'income_cnty',
             'n_jobs_cnty', 'indus_cnty', 'rur_urb_cnty', 'unemployment_cnty', 'pub_serv_ratio', 'ICITE',
             'Aridity']

    df_feat = df_feat[feats]
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.dropna(axis=0)

    df_feat['wu_per_capita'] = df_feat['wu_rate']/(df_feat['pop_swud_corrected'])
    #wu = wu[wu['population3']>3000]
    df_feat = df_feat[ df_feat['wu_per_capita']<700]
    df_feat = df_feat[df_feat['wu_per_capita'] > 20]
    df_feat =  df_feat[df_feat['pop_swud_corrected'] > 100]
    df_feat['pop_swud_corrected'] = np.log10(df_feat['pop_swud_corrected'])
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
    y = df_feat['wu_per_capita']
    X = df_feat.copy()
    one_hot = pd.get_dummies(X['HUC2'])
    X = X.drop('HUC2', axis=1)
    # Join the encoded df
    X = X.join(one_hot)
    del(X['wu_per_capita'])
    del (X['wu_rate'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 123)
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
        #importance_df = importance_df.sort_values(by=[1], ascending=False)
        #top_features = importance_df.iloc[:20][0].values.tolist()
        #X = df_feat.copy()#[top_features]
        y = np.log10(df_feat['wu_rate'])
        try:
            del (X['wu_rate'])
            del (X['wu_per_capita'])
        except:
            pass
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


        #data_dmatrix = xgb.DMatrix(data=X, label=y)  # 'objective': 'binary:logistic',
        gb = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.8, learning_rate=0.01,
                              max_depth=7, alpha=0.01, n_estimators=3000, rate_drop=0.9, skip_drop=0.5, subsample=0.8,
                              seed = 123, reg_lambda=0.0)
        ppcc = y_train/y_train

        #w2_train = 1/np.abs(ppcc-100*1.000012545)
        #nor = w2_train.min()
        w2_train =1+ np.log10( ppcc/ppcc.min())
        ppcc = y_test/y_test
        #w2_test = 1/np.abs(ppcc-100*1.000012545)
        w2_test = 1+np.log10(ppcc / ppcc.min())
        eval_set = [(X_train, y_train), (X_test, y_test)]
        gb.fit(X_train, y_train, eval_metric=["rmse", "rmse"], eval_set=eval_set, verbose=True, sample_weight=w2_train)
        from sklearn.metrics import r2_score
        ypredict = gb.predict(X_test)
        accuracy = r2_score(10**y_test, 10**ypredict, sample_weight = w2_test)

        results = gb.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['rmse'], label='train')
        ax.plot(x_axis, results['validation_1']['rmse'], label='test')
        plt.figure()
        roh = np.corrcoef(y_test.squeeze(), ypredict.squeeze())[0, 1]
        roh = int(roh * 1000) / 1000.0
        plt.scatter(y_test, ypredict, c = w2_test, cmap = 'jet', s = 3)
        # plt.plot([0, 6000], [0, 6000], 'r')
        c = [min(y_test), max(y_test)]
        plt.plot(c, c,'r')
        plt.title(str(accuracy))
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
        c = [min(y_test), max(y_test)]
        plt.plot(c,c)

        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        plt.title(accuracy)
        plt.show()

    xx = 1
