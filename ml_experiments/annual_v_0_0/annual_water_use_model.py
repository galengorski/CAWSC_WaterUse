# ===============****===================
"""
Annual Water Use Model
Date: 7/14/2022
"""
# ===============****===================
import os
import warnings
import pandas as pd
import numpy as np
from iwateruse import (
    data_cleaning,
    report,
    splittors,
    pre_train_utils,
    make_dataset,
    figures,
)
from iwateruse import denoise, model_diagnose
import matplotlib.pyplot as plt
import xgboost as xgb
import json
import shap
import joblib
from sklearn.pipeline import Pipeline
from iwateruse.model import Model
from iwateruse import (
    targets,
    weights,
    pipelines,
    outliers_utils,
    estimators,
    featurize,
    predictions,
)
from iwateruse import selection, interpret
from sklearn.model_selection import train_test_split
import copy

warnings.filterwarnings("ignore")
xgb.set_config(verbosity=0)

# =============================
# Flags
# =============================
work_space = r"models\annual\m7_29_2022"
clean_folder = False
log_file = "train_log.log"
split_seed = 123

# features
# apply_onehot_encoding = False
# apply_summary_encoding = True
# include_all_features = False

train_all_models = True
train_initial_model = False
train_model_no_encoding = False
train_denoised_model = False
plot_diagnosis = False

run_boruta = False
use_boruta_results = False
run_permutation_selection = False
run_chi_selection = False
run_RFECV_selection = False
detect_outliers = False
make_prediction = False
interpret_model = True
quantile_regression = False

# files
files_info = {}
files_info[
    "usa_conus_file"
] = r"C:\work\water_use\mldataset\gis\nation_gis_files\conus_usa2.shp"
files_info[
    "huc2_shp_file"
] = r"C:\work\water_use\mldataset\gis\nation_gis_files\conus_huc2.shp"
files_info[
    "state_shp_file"
] = r"C:\work\water_use\mldataset\gis\nation_gis_files\conus_states.shp"
files_info[
    "county_shp_file"
] = r"C:\work\water_use\mldataset\gis\nation_gis_files\conus_county.shp"
files_info["outliers_file_used"] = r"outliers_7_24_22.csv"
files_info["outliers_weights_file_used"] = r"outliers_weight_7_24_22.csv"
files_info[
    "AWUDS_file"
] = r"C:\work\water_use\mldataset\ml\training\misc_features\awuds_all_years.csv"

# =============================
# Setup Training
# =============================
model = Model(
    name="annual_pc",
    log_file=log_file,
    feature_status_file=r"features_status.xlsx",
    model_ws=work_space,
    clean=clean_folder,
)
figures_folder = os.path.join(model.model_ws, "figs")
model.raw_target = "wu_rate"
model.target = "per_capita"
model.files_info = files_info
datafile = r"clean_train_db.csv"
outliers_weights_df = pd.read_csv(files_info["outliers_weights_file_used"])
df_train = pd.read_csv(datafile)
model.add_training_df(df_train=df_train)
base_features = model.features

# model.light_copy(model)

model.features_info = {}
model.features_info["all_base_features"] = model.features

# =============================
# Use selected features
# =============================
selected_features = model.load_features_selected(method="xgb_cover")
dropped_feat = list(set(base_features).difference(selected_features))
model.add_feature_to_skip_list(dropped_feat)
model.features_info["selected_features"] = model.features

model.apply_func(
    func=targets.compute_per_capita, type="target_func", args=None
)

# =============================
# Feature Engineering
# =============================
# make cat features int and replace missing with -999
for catfeat in model.categorical_features:
    model.df_train.loc[model.df_train[catfeat].isna(), catfeat] = -999  # NaN
    model.df_train[catfeat] = model.df_train[catfeat].astype(int)
model.add_feature_to_skip_list(model.categorical_features)
model.features_info["cat_features"] = model.categorical_features

# apply onehot encoding
cols_before_encoding = model.df_train.columns
ohc = featurize.MultiOneHotEncoder(catfeatures=model.categorical_features)
df_tr = ohc.transform(model.df_train)
cat_df = model.df_train[model.categorical_features + ["sample_id"]]
df_tr = df_tr.merge(cat_df, how="left", on="sample_id")
model.add_feature_to_skip_list(model.categorical_features)
model.add_training_df(df_train=df_tr)
cols_after_encoding = df_tr.columns
ohc_features = set(cols_after_encoding).difference(cols_before_encoding)
model.features_info["ohc_features"] = list(ohc_features)
del df_tr
del cat_df

# summary encoding
cols_before_encoding = model.df_train.columns
df_trans = featurize.summary_encode(
    model, cols=model.categorical_features, quantiles=[0.25, 0.5, 0.75]
)
model.add_feature_to_skip_list(model.categorical_features)
model.add_training_df(df_train=df_trans)
cols_after_encoding = model.df_train.columns
summary_tragte_features = set(cols_after_encoding).difference(
    cols_before_encoding
)
del df_trans
model.features_info["summary_tragte_features"] = list(summary_tragte_features)

model.get_model_options()
# =============================
# Prepare Prediction df
# =============================
model.df_pred = model.df_train.copy()





# =============================
# Outliers
# =============================
opts = ["pop<=100", "per_capita>=700", "per_capita<=20"]
model.apply_func(
    func=outliers_utils.drop_values, type="outliers_func", opts=opts
)
model.apply_func(func=outliers_utils.drop_na_target, type="outliers_func")

# =============================
# Prepare the initial estimator
# =============================
params_old = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "colsample_bytree": 0.5972129389586888,
    "learning_rate": 0.04371772054983907,
    "max_depth": 11,
    "alpha": 100,
    "n_estimators": 300,
    "subsample": 1,
    "reg_lambda": 1.2998981941716606e-08,
    "min_child_weight": 4,
    "gamma": 10,
    "max_delta_step": 0,
    "seed": 456,
    "importance_type": "total_gain",
}
params = {
    "objective": "reg:squarederror",
    "base_score": 0.5,
    "booster": "gbtree",
    "callbacks": None,
    "colsample_bylevel": 1,
    "colsample_bynode": 1,
    "colsample_bytree": 0.867202783570103,
    "early_stopping_rounds": None,
    "enable_categorical": False,
    "eval_metric": None,
    "gamma": 0,
    # 'gpu_id': 0,
    "grow_policy": "depthwise",
    "importance_type": None,
    "interaction_constraints": "",
    "learning_rate": 0.12128959372061261,
    "max_bin": 256,
    "max_cat_to_onehot": 4,
    "max_delta_step": 0,
    "max_depth": 11,
    "max_leaves": 0,
    "min_child_weight": 4,
    "monotone_constraints": "()",
    "n_estimators": 300,
    "n_jobs": -1,
    "num_parallel_tree": 1,
    "predictor": "auto",
    "random_state": 5751,
    "reg_alpha": 10.0,
    "reg_lambda": 10.0,
    "sampling_method": "uniform",
    "scale_pos_weight": 50.0,
    "subsample": 0.867555264259934,
    "tree_method": "hist",
    "validate_parameters": 1,
    "verbosity": 0,
}

gb = estimators.xgb_estimator(params)
features = model.features
target = model.target

model.log.info("\n\n ****** Data used for training & testing****")
model.log.to_table(
    model.df_train[features].describe(percentiles=[0, 0.05, 0.5, 0.95, 1]),
    "Features",
    header=-1,
)
model.log.to_table(
    model.df_train[[target]].describe(percentiles=[0, 0.05, 0.5, 0.95, 1]),
    "Target",
    header=-1,
)

# =============================
#  Train all models
# =============================
# key features_samples_encoding_model
available_model_options = [
    "all_full_NoEncoding_xgb",
    "all_full_ohc_xgb",
    "all_full_smc_xgb",
    "all_denoised_NoEncoding_xgb",
    "all_denoised_ohc_xgb",
    "all_denoised_smc_xgb",
    "reduced_full_NoEncoding_xgb",
    "reduced_full_ohc_xgb",
    "reduced_full_smc_xgb",
    "reduced_denoised_NoEncoding_xgb",
    "reduced_denoised_ohc_xgb",
    "reduced_denoised_smc_xgb",
]

models_to_train = ["all_full_NoEncoding_xgb", "all_denoised_NoEncoding_xgb", "reduced_denoised_NoEncoding_xgb",
                   "reduced_denoised_smc_xgb"]
models_to_train = available_model_options
model_samples = {}
all_light_diagnosis = []
weight_threshold = 0.05
model.df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
train_all_models = False
if train_all_models:
    for mmodel in models_to_train:
        print("Training model {}".format(mmodel))

        df_train = model.df_train

        # get samples to use in the training
        if mmodel.split("_")[1] in ['denoised']:
            df_train = df_train.merge(outliers_weights_df[['sample_id', 'weight']], how="left", on="sample_id")
            df_train = df_train[df_train["weight"] > weight_threshold]

        X_train, X_test, y_train, y_test = train_test_split(
            df_train,
            df_train[target],
            test_size=0.3,
            shuffle=True,
            random_state=split_seed,
            stratify=df_train["HUC2"],
        )

        model_samples[mmodel] = [{'train': X_train['sample_id'].values}, {'test': X_test['sample_id'].values}]
        # get features and train
        mfeatures = model.models_features[mmodel]
        curr_trained_model = gb.fit(X_train[mfeatures], y_train)

        # diagnose
        y_pred = curr_trained_model.predict(X_test[mfeatures])
        diag_df = model_diagnose.generate_metrics(y_true=y_test, y_pred=y_pred)
        diag_df['model_name'] = mmodel
        all_light_diagnosis.append(diag_df.copy())
        # save
        model_file_name = os.path.join(model.model_ws, r"{}.json".format(mmodel))
        model.log.info("\n\n Model saved at {}".format(model_file_name))
        joblib.dump(curr_trained_model, model_file_name)

        all_light_diagnosis = pd.concat(all_light_diagnosis)
        all_light_diagnosis.to_csv(os.path.join(model.model_ws, 'all_models_diagnosis.csv'))

# =============================
# Full Diagnosis of models
# =============================
models_to_diagnose = ["all_full_NoEncoding_xgb", "all_denoised_NoEncoding_xgb"]
for mmodel in models_to_diagnose:
    model_file_name = os.path.join(model.model_ws, r"{}.json".format(mmodel))
    trained_model, pfeatures = estimators.get_model_from_file(model_file_name)
    model.set_features(pfeatures)
    df_train = model.df_train
    if mmodel.split("_")[1] in ['denoised']:
        df_train = df_train.merge(outliers_weights_df[['sample_id', 'weight']], how="left", on="sample_id")
        df_train = df_train[df_train["weight"] > weight_threshold]

    X_train, X_test, y_train, y_test = train_test_split(
        df_train,
        df_train[target],
        test_size=0.3,
        shuffle=True,
        random_state=split_seed,
        stratify=df_train["HUC2"],
    )

    model.splits = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    model_diagnose.complete_model_diagnose(
        model, estimator=trained_model, basename=mmodel
    )

# =============================
# Simplify model by dropping
# features with little importance
# =============================
drop_features = {}
feat_selec_file = "confirmed_selected_features.json"
if run_boruta:
    rf = xgb.XGBRFRegressor(
        n_estimators=500,
        subsample=0.8,
        colsample_bynode=0.5,
        max_depth=7,
    )

    # impute by mean for boruta
    final_dataset_ = model.df_train.copy()
    for feat in features:
        nanMask = final_dataset_[feat].isna()
        feat_mean = final_dataset_[feat].mean()
        final_dataset_.loc[nanMask, feat] = feat_mean

    importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
    for imtype in importance_types:
        rf.set_params(importance_type=imtype)
        confirmed_features = selection.boruta(
            X=final_dataset_[features], y=final_dataset_[target], estimator=rf
        )
        keyname = "rf_{}".format(imtype)
        drop_features[keyname] = confirmed_features.values.tolist()

        gb.set_params(importance_type=imtype)
        confirmed_features = selection.boruta(
            X=final_dataset_[features], y=final_dataset_[target], estimator=gb
        )
        keyname = "xgb_{}".format(imtype)
        drop_features[keyname] = confirmed_features.values.tolist()

    seeds = [123, 5795, 2136, 5214]
    perm_confirm_features = []
    for sd in seeds:
        X_train, X_test, y_train, y_test = splittors.stratified_split(
            model, test_size=0.3, id_column="HUC2", seed=sd
        )
        estm_ = gb.fit(X_train[features], y_train)
        scoring = [
            "r2"
        ]  # , 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
        feature_importance = selection.permutation_selection(
            X_test[features],
            y_test,
            estimator=estm_,
            scoring=scoring,
            n_repeats=5,
            features=features,
        )

        best_r2 = feature_importance[feature_importance["metric"].isin(["r2"])]
        confirmed_features = best_r2[
            (best_r2["mean_reduction"] / best_r2["mean_reduction"].max())
            > 0.015
            ]["feature"]
        perm_confirm_features = (
                perm_confirm_features + confirmed_features.values.tolist()
        )

    keyname = "permutation_r2"
    drop_features[keyname] = list(set(perm_confirm_features))
    with open(feat_selec_file, "w") as convert_file:
        convert_file.write(json.dumps(drop_features))
    del final_dataset_

# =============================
# Evaluate model after features
# selection
# =============================
if use_boruta_results:

    model.log.info(
        " *** Evaluate Selection using file : {}".format(feat_selec_file)
    )
    f = open(feat_selec_file)
    feature_selection_info = json.load(f)
    f.close()

    selection_method = sorted(list(feature_selection_info.keys()))
    selection_eval_results = []
    for method in selection_method:
        print("Evaluating selection method {}".format(method))
        curr_features = feature_selection_info[method]
        if "permut" in method:
            model_type = "xgb"
            imp_metric = "r2"
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
        df_metric = model_diagnose.generate_metrics(
            y_true=y_test, y_pred=y_hat
        )
        df_metric["model"] = model_type
        df_metric["importance_metrix"] = imp_metric
        df_metric["nfeatures"] = len(curr_features)
        selection_eval_results.append(df_metric.copy())
    selection_eval_results = pd.concat(selection_eval_results)
    model.log.to_table(
        df=selection_eval_results,
        title="Evaluation of feature selection ",
        header=len(selection_eval_results),
    )
    selection_eval_results.to_csv(r"results\feature_selection_summary.csv")

if run_chi_selection:
    chi_test_df = selection.chi_square_test(
        X=model.df_train[features], y=model.df_train[target], nbins=20
    )

if run_RFECV_selection:
    from sklearn.feature_selection import RFE, RFECV

    selector = RFECV(vv, verbose=10, scoring="r2")
    selector.fit(model.df_train[features], model.df_train[target])
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
    features_to_use = feature_selection_info["rf_gain"]

    df_results = denoise.purify(
        dataset,
        target="per_capita",
        features=features_to_use,
        col_id="sample_id",
        max_iterations=400,
        estimator=gb,
        score="neg_root_mean_squared_error",
        min_signal_ratio=0.17,
        min_mse=30 ** 2.0,
    )

# =============================
# generate samples weight
# =============================
compute_noise_weights = True
burn_period = 30
if compute_noise_weights:
    noise_detect_files = []
    for f in noise_detect_files:
        dfff = model.df_train.copy()
        outlier_info = pd.read_csv(f)
        ids = []
        for col in outlier_info.columns:
            if col.isdigit():
                ids.append(col)

        outlier_info["signal_ratio"] = outlier_info[ids].sum(axis=1) / len(ids)
        oo = outlier_info[outlier_info["iter"] > 30]
        w = oo[ids].mean(axis=0)
        w = w.reset_index().rename(columns={"index": "sample_id", 0: "weight"})
        w["sample_id"] = w["sample_id"].astype(int)
        w = w[["sample_id", "weight"]]
        w.to_csv("weight_" + f)

# =============================
# Predictions
# =============================
if make_prediction:

    model_file_name = os.path.join(
        model.model_ws, r"denoised_model_with_selected_features.json"
    )
    model_predict = joblib.load(model_file_name)
    try:
        pfeatures = model_predict.get_booster().feature_names
    except:
        pfeatures = model_predict.steps[-1][1].get_booster().feature_names
    model.df_pred["est_per_capita"] = model_predict.predict(
        model.df_pred[pfeatures]
    )

    model_diagnose.complete_model_evaluate(
        model, estimator=model_predict, basename="result_denoised"
    )

# =============================
# Interpret the model
# =============================
if interpret_model:
    interp_folder = os.path.join(model.model_ws, "interp")
    if not (os.path.isdir(interp_folder)):
        os.mkdir(interp_folder)

    # =============================
    # PDPs plots
    # =============================
    # model_file_name = os.path.join(model.model_ws, r"1_initial_model.json")
    model_file_name = os.path.join(
        model.model_ws, r"1_initial_model_no_encoding.json"
    )
    # model_file_name = os.path.join(model.model_ws, r"denoised_model_with_selected_features.json")
    model_predict = joblib.load(model_file_name)
    try:
        pfeatures = model_predict.get_booster().feature_names
    except:
        pfeatures = model_predict.steps[-1][1].get_booster().feature_names
    basename = "initial_with_encoding"

    fn = os.path.join(
        interp_folder, "interp_partial_dep_plots_{}.pdf".format(basename)
    )
    interpret.pdp_plots(
        estimator=model_predict,
        df=model.df_train,
        fn=fn,
        features=pfeatures,
    )

    ffeat = [
        "median_income",
        "average_income",
        "prc_n_bachelors",
        "prc_n_masters_phd",
        "prc_n_associates",
    ]
    fn = os.path.join(
        interp_folder,
        "interp_by_huc2_partial_dep_plots_{}.pdf".format(basename),
    )
    interpret.pdp_plots_by_huc2(
        estimator=model_predict,
        df=model.df_train,
        fn=fn,
        all_features=pfeatures,
        interp_feat=ffeat,
        huc2=[2, 18, 11],
    )

    # =============================
    # SHAP plots for each HUC2
    # =============================
    fn = os.path.join(
        interp_folder, "interp_shap_values_{}.pdf".format(basename)
    )
    interpret.shap_plots(
        model,
        estimator=model_predict,
        fn=fn,
        features=pfeatures,
        data_frac=0.5,
    )

# =============================
# Quantile regression
# =============================
from lightgbm import LGBMRegressor

lgb_params = {
    "n_jobs": 1,
    "max_depth": 8,
    "min_data_in_leaf": 10,
    "subsample": 0.8,
    "n_estimators": 500,
    "learning_rate": 0.1,
    "colsample_bytree": 0.8,
    "boosting_type": "gbdt",
}

lgb_params = {
    "boosting_type": "gbdt",
    "class_weight": None,
    "colsample_bytree": 1.0,
    "importance_type": "split",
    "learning_rate": 0.0789707832086975,
    "max_depth": -1,
    "min_child_samples": 1,
    "min_child_weight": 0.001,
    "min_split_gain": 0.09505847801910139,
    "n_estimators": 300,
    "n_jobs": -1,
    "num_leaves": 256,
    "objective": "quantile",
    "random_state": 1880,
    "reg_alpha": 0.7590311640897429,
    "reg_lambda": 1.437857111206781e-10,
    "silent": "warn",
    "subsample": 1.0,
    "subsample_for_bin": 200000,
    "subsample_freq": 0,
    "bagging_fraction": 1.0,
    "bagging_freq": 7,
    "feature_fraction": 0.6210738953447875,
}
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
    plt.scatter(y_test, ypredict, s=4, label="{}".format(quantile_alpha))
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.legend()
lim = [min(y_test), max(y_test)]
plt.plot(lim, lim, "k")
plt.xlabel("Actual Water Use")
plt.ylabel("Estimated Water Use")
