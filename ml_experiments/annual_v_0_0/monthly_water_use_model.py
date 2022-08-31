# ===============****===================
"""
Monthly Water Use Model
Date: 8/04/2022
"""
# ===============****===================
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json
import xgboost as xgb
from sklearn.pipeline import Pipeline
from iwateruse.model import Model
from iwateruse import targets, weights, pipelines, outliers_utils, estimators, featurize
from iwateruse import selection
from iwateruse import data_cleaning, report, splittors, pre_train_utils, make_dataset, figures
from iwateruse import denoise, model_diagnose
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings('ignore')
xgb.set_config(verbosity=0)

# =============================
# Flags
# ==============================
work_space = r"models\monthly\m7_29_2022"
clean_folder = False
log_file = 'train_log_monthly.log'

apply_onehot_encoding = False
apply_summary_encoding = True
train_initial_model = False
plot_diagnosis = False
run_boruta = False
use_boruta_results = False
run_permutation_selection = False
run_chi_selection = False
run_RFECV_selection = False
detect_outliers = False
train_denoised_model = False
make_prediction = True
interpret_model = False
quantile_regression = False

# files
files_info = {}
files_info['usa_conus_file'] = r"C:\work\water_use\mldataset\gis\nation_gis_files\conus_usa2.shp"
files_info['huc2_shp_file'] = r'C:\work\water_use\mldataset\gis\nation_gis_files\conus_huc2.shp'
files_info['state_shp_file'] = r'C:\work\water_use\mldataset\gis\nation_gis_files\conus_states.shp'
files_info['county_shp_file'] = r'C:\work\water_use\mldataset\gis\nation_gis_files\conus_county.shp'
files_info['outliers_file_used'] = r"monthly_outliers.csv"
files_info['AWUDS_file'] = r"C:\work\water_use\mldataset\ml\training\misc_features\awuds_all_years.csv"

# =============================
# Setup Training
# =============================
model = Model(name='Monthly_frac', log_file= log_file, feature_status_file=r"features_status.xlsx",
              model_type='monthly', model_ws=work_space, clean = clean_folder)
figures_folder = os.path.join(model.model_ws, "figs")
model.raw_target = 'monthly_wu'
model.target = 'monthly_fraction'
model.files_info = files_info
datafile = r"clean_train_db_monthly.csv"  # clean_train_db_backup_6_24_2022.csv
df_train = pd.read_csv(datafile)
model.add_training_df(df_train=df_train)
base_features = model.features

# =============================
# Use selected features
# =============================
if 1:
    selected_features = model.load_features_selected(method='xgb_total_cover',
                                                     feat_selec_file = 'confirmed_selected_features_monthly.json')
    dropped_feat = list(set(base_features).difference(selected_features))
    model.add_feature_to_skip_list(dropped_feat)

# space for target function

# =============================
# Feature Engineering
# =============================
# make cat features int and replace missing with -999
for catfeat in model.categorical_features:
    model.df_train.loc[model.df_train[catfeat].isna(), catfeat] = -999  # NaN
    model.df_train[catfeat] = model.df_train[catfeat].astype(int)

if apply_onehot_encoding:
    ohc = featurize.MultiOneHotEncoder(catfeatures=model.categorical_features)
    df_tr = ohc.transform(model.df_train)
    cat_df = model.df_train[model.categorical_features + ['sample_id']]
    df_tr = df_tr.merge(cat_df, how='left', on='sample_id')
    model.add_feature_to_skip_list(model.categorical_features)
    model.add_training_df(df_train=df_tr)
    del (df_tr)
    del (cat_df)

if apply_summary_encoding:
    cat_feat = ['KG_climate_zone', 'HUC2'] #'state_id', 'rur_urb_cnty'
    df_trans = featurize.summary_encode(model, cols=cat_feat,
                                        quantiles=[0.25, 0.5, 0.75],
                                        max_target=0.18, min_target=0.03,
                                        min_pop=0)
    model.add_feature_to_skip_list(model.categorical_features)
    model.add_training_df(df_train=df_trans)
    del (df_trans)

# =============================
# Prepare Prediction df
# =============================
model.df_pred = model.df_train.copy()

# =============================
# Outliers
# =============================
#opts = ['monthly_fraction>0.18', 'monthly_fraction<0.03', 'simil_stat<0.7']
opts = ['monthly_fraction>0.2', 'monthly_fraction<0.01', 'simil_stat<0.3']
model.apply_func(func=outliers_utils.drop_values, type='outliers_func', opts=opts)
model.apply_func(func=outliers_utils.drop_na_target, type='outliers_func')

# =============================
# Prepare the initial estimator
# =============================

params = {
    'objective': "reg:squarederror",
    'tree_method': 'hist',
    'colsample_bytree': 0.9,  #
    'learning_rate': 0.06,  #
    'max_depth': 10,  #
    'alpha': 0.0,  #
    'n_estimators': 600,  #
    'rate_drop': 0.1,  #
    'skip_drop': 0.5,  #
    'subsample': 0.85,  #
    'reg_lambda': 1,
    'min_child_weight': 1,
    'gamma': 0,  #
    'max_delta_step': 0,
    'seed': 456,
    'importance_type': 'total_gain'  # this is used in boruta
}


































gb = estimators.xgb_estimator(params)
features = model.features
target = model.target

final_dataset = model.df_train
model.log.to_table(final_dataset[features].describe(percentiles=[0, 0.05, 0.5, 0.95, 1]), "Features", header=-1)
model.log.to_table(final_dataset[[target]].describe(percentiles=[0, 0.05, 0.5, 0.95, 1]), "Target", header=-1)

X_train, X_test, y_train, y_test = splittors.stratified_split(model, test_size=0.3, id_column='HUC2', seed=123)

model.splits = {"X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test}

# =============================
# initial Model Training
# =============================

if train_initial_model:
    trained_gb = gb.fit(X_train[features], y_train)

    model_file_name = os.path.join(model.model_ws, r"1_initial_model.json")
    model.log.info("\n\n Initial Model saved at ")
    joblib.dump(trained_gb, model_file_name)

    #  diagnose
    if plot_diagnosis:
        model_diagnose.complete_model_diagnose(model, estimator=trained_gb, basename="initial", monthly=True)

# =============================
# Simplify model by dropping
# features with little importance
# =============================
drop_features = {}
feat_selec_file = 'confirmed_selected_features_monthly.json'
if run_boruta:
    rf = xgb.XGBRFRegressor(n_estimators=500, subsample=0.8, colsample_bynode=0.5, max_depth=7, )
    gb = trained_gb
    # impute by mean for boruta
    final_dataset_ =  model.df_train.copy()
    for feat in features:
        nanMask = final_dataset_[feat].isna()
        feat_mean = final_dataset_[feat].mean()
        final_dataset_.loc[nanMask, feat] = feat_mean

    importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    for imtype in importance_types:
        rf.set_params(importance_type=imtype)
        confirmed_features = selection.boruta(X=final_dataset_[features], y=final_dataset_[target], estimator=rf)
        keyname = "rf_{}".format(imtype)
        drop_features[keyname] = confirmed_features.values.tolist()

        gb.set_params(importance_type=imtype)
        confirmed_features = selection.boruta(X=final_dataset_[features], y=final_dataset_[target], estimator=gb)
        keyname = "xgb_{}".format(imtype)
        drop_features[keyname] = confirmed_features.values.tolist()

    seeds = [123, 5795, 2136, 5214]
    perm_confirm_features = []
    for sd in seeds:
        X_train, X_test, y_train, y_test = splittors.stratified_split(model, test_size=0.3, id_column='HUC2', seed=sd)
        estm_ = gb.fit(X_train[features], y_train)
        scoring = ['r2']  # , 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
        feature_importance = selection.permutation_selection(X_test[features], y_test, estimator=estm_, scoring=scoring,
                                                             n_repeats=5, features=features)

        best_r2 = feature_importance[feature_importance['metric'].isin(['r2'])]
        confirmed_features = best_r2[(best_r2['mean_reduction'] / best_r2['mean_reduction'].max()) > 0.015]['feature']
        perm_confirm_features = perm_confirm_features + confirmed_features.values.tolist()

    keyname = "permutation_r2"
    drop_features[keyname] = list(set(perm_confirm_features))
    with open(feat_selec_file, 'w') as convert_file:
        convert_file.write(json.dumps(drop_features))
# =============================
# Evaluate model after features
# selection
# =============================
if use_boruta_results:

    model.log.info(" *** Evaluate Selection using file : {}".format(feat_selec_file))
    f = open(feat_selec_file)
    feature_selection_info = json.load(f)
    f.close()

    selection_method = sorted(list(feature_selection_info.keys()))
    selection_eval_results = []
    for method in selection_method:
        print("Evaluating selection method {}".format(method))
        curr_features = feature_selection_info[method]
        if "permut" in method:
            model_type = 'xgb'
            imp_metric = 'r2'
        else:
            parts = method.split("_")
            model_type = parts[0]
            if len(parts) > 2:
                imp_metric = "_".join(parts[1:])
            else:
                imp_metric = parts[1]

        vv = gb.fit(X_train[curr_features], y_train)
        y_hat = vv.predict(X_test[curr_features])
        err = pd.DataFrame(y_test - y_hat)
        df_metric = model_diagnose.generate_metrics(y_true=y_test, y_pred=y_hat)
        df_metric['model'] = model_type
        df_metric['importance_metrix'] = imp_metric
        df_metric['nfeatures'] = len(curr_features)
        selection_eval_results.append(df_metric.copy())
    selection_eval_results = pd.concat(selection_eval_results)
    model.log.to_table(df=selection_eval_results, title="Evaluation of feature selection ",
                       header=len(selection_eval_results))
    selection_eval_results.to_csv(r"results\feature_selection_summary_monthly.csv")

# =============================
# Outliers Detection ...
# =============================
if detect_outliers:
    dataset = model.df_train.copy()

    f = open(feat_selec_file)
    feature_selection_info = json.load(f)
    f.close()
    features_to_use = model.features

    mis_mse =  (1.0/100.0)**2.0
    df_results = denoise.purify(dataset, target='monthly_fraction', features=features_to_use, col_id='sample_id',
                                max_iterations=400, estimator=gb, score='neg_root_mean_squared_error',
                                min_signal_ratio=0.17, min_mse=mis_mse)

# =============================
# Train denoised model
# =============================
if train_denoised_model:
    model.log.info("**** Train model with denoised data ...")
    model.log.info("**** Outliers file name : {}".format(files_info['outliers_file_used']))
    weight_threshold = 0.1
    model.log.info("**** Weight Threshold value: {}".format(weight_threshold))
    dfff = model.df_train.copy()

    w = pd.read_csv(r"monthly_weights_weights.csv")
    dfff = dfff.merge(w, how='left', on='sample_id')
    dfff1 = dfff[dfff['weight'] > weight_threshold]
    dfff1['weight'] = 1.0
    model.log.to_table(dfff1, title='Denoised Data set')

    X_train, X_test, y_train, y_test = train_test_split(dfff1, dfff1[target], test_size=0.3,
                                                        shuffle=True,
                                                        random_state=123, stratify=dfff1['HUC2'])

    model.splits = {"X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test}

    # --- use all features
    train_with_weights = False
    if train_with_weights:
        w_train = X_train['weight'].values
        w_test = X_test['weight'].values
        gb.fit(X_train[features], y_train, sample_weight=w_train)
    else:
        gb.fit(X_train[features], y_train)
    y_hat = gb.predict(X_test[features])
    df_metric = model_diagnose.generate_metrics(y_true=y_test, y_pred=y_hat)
    df_metric.to_csv(os.path.join(model.model_ws, 'metric_denoised_all_features.csv'), index=False)
    model.log.to_table(df_metric, title="Metrics of denoised model with selected features")

    # diagnose model
    model_diagnose.complete_model_diagnose(model, estimator=gb, basename="denoised",
                                           monthly=True)

    # save model
    model_file_name = os.path.join(model.model_ws, "denoised_model_with_selected_features.json")
    model.log.info("\n\n Initial Model saved at {}".format(model_file_name))
    joblib.dump(gb, model_file_name)
    v = 1





if make_prediction:
    saved_models = [r"1_initial_model.json",  r"denoised_model_with_selected_features.json"]
    basenames = ['denoised'] #'intial',
    for im, mm in enumerate(saved_models):
        model_file_name = os.path.join(model.model_ws,mm )
        model_predict = joblib.load(model_file_name)
        try:
            pfeatures = model_predict.get_booster().feature_names
        except:
            pfeatures = model_predict.steps[-1][1].get_booster().feature_names
        model.df_pred['est_month_frac'] = model_predict.predict(model.df_pred[pfeatures])
        model_diagnose.complete_monthly_model_eval(model, estimator=model_predict,
                                                   basename=basenames[im])
        vvv = 1

# =============================
# Interpret the model
# =============================
if interpret_model:
    import shap

    shap.partial_dependence_plot('BuildingAreaSqFt_sum', gb.predict, X_test, ice=
    False, model_expected_value=True, feature_expected_value=True)

    # X100 = shap.utils.sample(X_train, 100)
    # X500 = shap.utils.sample(X_train, 5000)
    # explainer = shap.Explainer(model.predict, X100)
    X100 = X_train[X_train['HUC2'] == 18]
    X100 = shap.utils.sample(X100, 2000)
    explainer = shap.Explainer(vv, X100)
    # explainer = shap.TreeExplainer(gb, X100)
    shap_values = explainer(X100)
    # shap.plots.waterfall(shap_values[150], max_display=14)
    # shap.summary_plot(shap_values, X100)
    shap.plots.beeswarm(shap_values, max_display=14)
    shap.plots.scatter(shap_values[:, "etr"], color=shap_values)
    # PDPs
    from sklearn.inspection import PartialDependenceDisplay

    common_params = {
        "subsample": 50,
        "n_jobs": -1,
        "grid_resolution": 20,
        "random_state": 0,
    }
    interp_feat = ['bdg_lt_2median']
    PartialDependenceDisplay.from_estimator(gb, X_test, interp_feat)

# =============================
# Quantile regression
# =============================
if quantile_regression:
    from lightgbm import LGBMRegressor

    lgb_params = {'boosting_type': 'gbdt',
                  'class_weight': None,
                  'colsample_bytree': 1.0,
                  'importance_type': 'split',
                  'learning_rate': 0.0789707832086975,
                  'max_depth': -1,
                  'min_child_samples': 1,
                  'min_child_weight': 0.001,
                  'min_split_gain': 0.09505847801910139,
                  'n_estimators': 300,
                  'n_jobs': -1,
                  'num_leaves': 256,
                  'objective': 'quantile',
                  'random_state': 1880,
                  'reg_alpha': 0.7590311640897429,
                  'reg_lambda': 1.437857111206781e-10,
                  'silent': 'warn',
                  'subsample': 1.0,
                  'subsample_for_bin': 200000,
                  'subsample_freq': 0,
                  'bagging_fraction': 1.0,
                  'bagging_freq': 7,
                  'feature_fraction': 0.6210738953447875}
    quantile_alphas = [0.25, 0.5, 0.75]

    lgb_quantile_alphas = {}
    for quantile_alpha in quantile_alphas:
        lgb = LGBMRegressor(alpha=quantile_alpha, **lgb_params)
        lgb.fit(X_train, y_train)
        lgb_quantile_alphas[quantile_alpha] = lgb

    plt.figure()
    for quantile_alpha, lgb in lgb_quantile_alphas.items():
        ypredict = lgb.predict(X_test)
        plt.scatter(y_test, ypredict, s=4, label="{}".format
        (quantile_alpha))
    plt.legend()
    lim = [min(y_test), max(y_test)]
    plt.plot(lim, lim, 'k')
    plt.xlabel("Actual Water Use")
    plt.ylabel("Estimated Water Use")
