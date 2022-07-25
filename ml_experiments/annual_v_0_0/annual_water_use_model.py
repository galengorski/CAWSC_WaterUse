# ===============****===================
"""
Annual Water Use Model
Date: 7/14/2022
"""
# ===============****===================
import os
import warnings
import pandas as pd
from iwateruse import data_cleaning, report, splittors, pre_train_utils, make_dataset, figures
from iwateruse import denoise, model_diagnose
import matplotlib.pyplot as plt
import xgboost as xgb
import json
import shap
import joblib
from sklearn.pipeline import Pipeline
from iwateruse.model import Model
from iwateruse import targets, weights, pipelines, outliers_utils, estimators, featurize, predictions
from iwateruse import selection
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
xgb.set_config(verbosity=0)

# =============================
# Flags
# =============================
train_initial_model = True
plot_diagnosis = False
run_boruta = True
use_boruta_results = False
run_permutation_selection = False
run_chi_selection = False
run_RFECV_selection = False
detect_outliers = False
use_denoised_data = True

interpret_model = False
quantile_regression = False

# files
usa_conus_file = r"C:\work\water_use\mldataset\gis\nation_gis_files\conus_usa2.shp"
huc2_shp_file = r'C:\work\water_use\mldataset\gis\nation_gis_files\conus_huc2.shp'
state_shp_file = r'C:\work\water_use\mldataset\gis\nation_gis_files\conus_states.shp'
county_shp_file = r'C:\work\water_use\mldataset\gis\nation_gis_files\conus_county.shp'
prediction_file = os.path.abspath(r"..\..\..\mldataset\ml\training\train_datasets\Annual\wu_annual_training3.csv")

# =============================
# Setup Training
# =============================
figures_folder = "figs"
model = Model(name='annual_pc', log_file='train_log.log', feature_status_file=r"features_status.xlsx")
model.raw_target = 'wu_rate'
model.target = 'per_capita'

datafile = r"clean_train_db.csv"
df_train = pd.read_csv(datafile)
del (df_train['dom_frac'])
del (df_train['cii_frac'])
model.add_training_df(df_train=df_train)

# make_dataset.make_ds_per_capita_basic(model, datafile=datafile)
# model.df_train['pop_density']  = model.df_train['pop']/model.df_train['WSA_SQKM']
# model.df_train.loc[model.df_train['WSA_SQKM']==0, 'pop_density'] = np.NaN
# add water use
seed1 = 123
seed2 = 456

model.apply_func(func=targets.compute_per_capita, type='target_func', args=None)

opts = ['pop<=100', 'per_capita>=500', 'per_capita<=25']
model.apply_func(func=outliers_utils.drop_values, type='outliers_func', opts=opts)
model.apply_func(func=outliers_utils.drop_na_target, type='outliers_func')

# =============================
# Feature Engineering
# =============================
# Target summary
df_trans, encoding_summary = featurize.summary_encode(model, cols=model.categorical_features)
fn = "summary_encoding.json"
model.dict_to_file(data = encoding_summary, fn=fn)
if False:
    model.add_feature_to_skip_list(model.categorical_features)
    model.add_training_df(df_train=df_trans)


# =============================
# Prepare the initial estimator
# =============================
params = {
    'objective': "reg:squarederror",
    'tree_method': 'hist',
    'colsample_bytree': 0.5972129389586888,
    'learning_rate': 0.04371772054983907,
    'max_depth': 11,
    'alpha': 100,
    'n_estimators': 300,
    'subsample': 1,
    'reg_lambda': 1.2998981941716606e-08,
    'min_child_weight': 4,
    'gamma': 10,
    'max_delta_step': 0,
    'seed': seed2,
    'importance_type': 'total_gain'
}

gb = estimators.xgb_estimator(params)

encode_cat_features = False
if encode_cat_features:
    # pipeline
    main_pipeline = pipelines.make_pipeline(model)
    main_pipeline.append(('estimator', gb))
    gb = Pipeline(main_pipeline)

features = model.features
target = model.target

final_dataset = model.df_train
model.log.to_table(final_dataset[features].describe(percentiles=[0, 0.05, 0.5, 0.95, 1]), "Features", header=-1)
model.log.to_table(final_dataset[[target]].describe(percentiles=[0, 0.05, 0.5, 0.95, 1]), "Target", header=-1)

X_train, X_test, y_train, y_test = splittors.stratified_split(model, test_size=0.3, id_column='HUC2', seed=123)
# X_train, X_test, y_train, y_test = splittors.split_by_id(model, args = {'frac' : 0.7, 'seed':547, 'id_column':'sys_id' })
# X_train, X_test, y_train, y_test = train_test_split(final_dataset[features],  final_dataset[target],
#                                                               test_size=0.2, random_state=123)#, shuffle = False

# w_train, w_test = weights.generate_weights_ones(model)
# =============================
# initial Model Training
# =============================

if train_initial_model:
    vv = gb.fit(X_train[features], y_train)
    y_hat = gb.predict(X_test[features])
    err = pd.DataFrame(y_test - y_hat)

    model_file_name = r".\models\annual\1_initial_model.json"
    model.log.info("\n\n Initial Model saved at ")
    joblib.dump(vv, model_file_name)
    # to load model ==> loaded_model = joblib.load(model_file_name)


    # =============================
    #  diagnose
    # =============================
    df_metric = model_diagnose.generate_metrics(y_true=y_test, y_pred=y_hat)
    if plot_diagnosis:
        heading = "Scatter Plot for Annual Water Use"
        xlabel = "Actual Per Capita Water Use - Gallons"
        ylabel = "Estimated Per Capita Water Use - Gallons"
        figfile = os.path.join(figures_folder, "annual_one_2_one_initial.pdf")
        figures.one_to_one(y_test, y_hat, heading=heading, xlabel=xlabel, ylabel=ylabel, figfile=figfile)
        model.log.info("\n\n\n ======= Initial model performance ==========")

        model.log.info("See initial model perfromance at {}".format(figfile))
        model.log.to_table(df_metric)

        # regional performance
        perf_df_huc2 = model_diagnose.generat_metric_by_category(gb, X_test, y_test, features=features, category='HUC2')
        # perf_df_state = model_diagnose.generat_metric_by_category(gb, X_test, y_test,features=features, category='HUC2')

        figfile = os.path.join(figures_folder, "annual_validation_error_by_huc2.pdf")
        figures.plot_huc2_map(shp_file=huc2_shp_file, usa_map=usa_conus_file, info_df=perf_df_huc2, legend_column='rmse',
                              log_scale=False, epsg=5070, cmap='cool', title='R2', figfile=figfile)

        figfile = os.path.join(figures_folder, "map_val_error.pdf")
        figures.plot_scatter_map(X_test['LONG'], X_test['LAT'], err,
                                 legend_column='per_capita', cmap='jet', title="Per Capita WU", figfile=figfile,
                                 log_scale=False)

        # plot importance
        importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        for type in importance_types:
            figfile = os.path.join(figures_folder, "annual_feature_importance_{}_v_0.pdf".format(type))
            if isinstance(gb, Pipeline):
                estm = gb.named_steps["estimator"]
                gb.named_steps["preprocess"].get_feature_names()
            else:
                estm = gb
            figures.feature_importance(estm, max_num_feature=15, type=type, figfile=figfile)

# =============================
# Simplify model by dropping
# features with little importance
# =============================
drop_features = {}
feat_selec_file = 'confirmed_selected_features.json'
if run_boruta:
    rf = xgb.XGBRFRegressor(n_estimators=500, subsample=0.8, colsample_bynode=0.5, max_depth=7, )

    # impute by mean for boruta
    final_dataset_ = final_dataset.copy()
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
        y_hat = gb.predict(X_test[curr_features])
        err = pd.DataFrame(y_test - y_hat)
        df_metric = model_diagnose.generate_metrics(y_true=y_test, y_pred=y_hat)
        df_metric['model'] = model_type
        df_metric['importance_metrix'] = imp_metric
        df_metric['nfeatures'] = len(curr_features)
        selection_eval_results.append(df_metric.copy())
    selection_eval_results = pd.concat(selection_eval_results)
    model.log.to_table(df=selection_eval_results, title="Evaluation of feature selection ",
                       header=len(selection_eval_results))
    selection_eval_results.to_csv(r"results\feature_selection_summary.csv")

if run_chi_selection:
    chi_test_df = selection.chi_square_test(X=final_dataset[features], y=final_dataset[target], nbins=20)

if run_RFECV_selection:
    from sklearn.feature_selection import RFE, RFECV

    selector = RFECV(vv, verbose=10, scoring='r2')
    selector.fit(final_dataset[features], final_dataset[target])
    features_selected = selector.get_feature_names_out()
    feature_rank = selector.ranking_

# =============================
# Outliers Detection ...
# =============================
if detect_outliers:
    dataset = model.df_train.copy()

    f = open(feat_selec_file)
    feature_selection_info = json.load(f)
    f.close()
    features_to_use = feature_selection_info['rf_gain']

    df_results = denoise.purify(dataset, target='per_capita', features=features_to_use, col_id='sample_id',
                                max_iterations=400, estimator=gb, score='neg_root_mean_squared_error',
                                min_signal_ratio=0.17, min_mse=30 ** 2.0)
# =============================
# Use denoised data
# =============================
if use_denoised_data:
    dfff = model.df_train.copy()
    outlier_info = pd.read_csv(r"outliers_7_24_22.csv")
    ids = []
    for col in outlier_info.columns:
        if col.isdigit():
            ids.append(col)

    outlier_info['signal_ratio'] = outlier_info[ids].sum(axis=1) / len(ids)
    oo = outlier_info[outlier_info['iter'] > 30]
    w = oo[ids].mean(axis=0)
    w = w.reset_index().rename(columns={'index': 'sample_id', 0: 'weight'})
    w['sample_id'] = w['sample_id'].astype(int)
    dfff = dfff.merge(w, how = 'left', on = 'sample_id')
    dfff1 = dfff[dfff['weight'] > 0.05]
    dfff1['weight'] = 1.0

    X_train, X_test, y_train, y_test = train_test_split(dfff1[features + ['weight']], dfff1[target], test_size=0.3, shuffle=True,
                                                        random_state=777, stratify=dfff1['HUC2'])


    w_train = X_train['weight'].values
    w_test = X_test['weight'].values
    gb.fit(X_train[features], y_train, sample_weight=w_train)

    plt.scatter(y_test, gb.predict(X_test[features]), marker="o", s=20, c=w_test, cmap='jet',
                alpha=0.5)
    print(r2_score(gb.predict(X_test[features]), y_test))
    cc = 1


# =============================
# Predictions
# =============================
if make_prediction:
    pred_col = 'pred' + model.target
    model_predict = joblib.load(r".\models\annual\1_initial_model.json")
    pred_features = model_predict.get_booster().feature_names
    df_prediction = pd.read_csv(prediction_file)


    df_prediction[model.target] = df_prediction['wu_rate'] / df_prediction['pop']
    cat_groups = list(encoding_summary.keys())
    for cat_feat in cat_groups:
        curr_group = encoding_summary[cat_feat]
        curr_group = pd.DataFrame.from_dict(curr_group)
        curr_group.index.rename(cat_feat, inplace=True)
        curr_group.reset_index(inplace=True)
        df_prediction = df_prediction.merge(curr_group, how='left', on=cat_feat)

    df_prediction[pred_col] = model_predict.predict(df_prediction[pred_features])
    b2010 = df_prediction[df_prediction['Year'] == 2010]
    plt.scatter(b2010['LONG'], b2010['LAT'], s=4, c=b2010[pred_col], cmap='jet')

    huc2_wu = predictions.upscale_to_area(df_prediction, pred_column= pred_col, scale='HUC2')


    ccc = 1


# =============================
# Interpret the model
# =============================
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
from lightgbm import LGBMRegressor

lgb_params = {
    'n_jobs': 1,
    'max_depth': 8,
    'min_data_in_leaf': 10,
    'subsample': 0.8,
    'n_estimators': 500,
    'learning_rate': 0.1,
    'colsample_bytree': 0.8,
    'boosting_type': 'gbdt'
}

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
    # to train a quantile regression, we change the objective parameter and
    # specify the quantile value we're interested in
    lgb = LGBMRegressor(alpha=quantile_alpha, **lgb_params)
    lgb.fit(X_train, y_train)
    lgb_quantile_alphas[quantile_alpha] = lgb
plt.figure()
for quantile_alpha, lgb in lgb_quantile_alphas.items():
    ypredict = lgb.predict(X_test)
    plt.scatter(y_test, ypredict, s=4, label="{}".format
    (quantile_alpha))
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.legend()
lim = [min(y_test), max(y_test)]
plt.plot(lim, lim, 'k')
plt.xlabel("Actual Water Use")
plt.ylabel("Estimated Water Use")
