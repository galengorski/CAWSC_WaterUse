import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy


def boruta(X, y, estimator):
    # load X and y
    # NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
    # X = pd.read_csv('examples/test_X.csv', index_col=0).values
    # y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
    # y = y.ravel()

    # define random forest classifier, with utilising all cores and
    # sampling in proportion to y labels
    # rf = RandomForestRegressor(n_jobs=-1, max_depth=5)

    # define Boruta feature selection method
    feat_selector = BorutaPy(
        estimator, n_estimators="auto", verbose=2, random_state=1
    )

    # find all relevant features - 5 features should be selected
    feat_selector.fit(X.values, y.ravel())

    # check selected features - first 5 features are selected
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    # X_filtered = feat_selector.transform(X)
    return X.columns[feat_selector.support_]


def permutation_selection(
    X, y, estimator, scoring, n_repeats=5, features=None
):
    scoring = [
        "r2",
        "neg_mean_absolute_percentage_error",
        "neg_mean_squared_error",
    ]
    from sklearn.inspection import permutation_importance

    r_multi = permutation_importance(
        estimator, X, y, n_repeats=n_repeats, random_state=0, scoring=scoring
    )

    all_metrics = []
    for metric in r_multi:
        print(f"{metric}")
        r = r_multi[metric]

        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                all_metrics.append(
                    [
                        features[i],
                        metric,
                        r.importances_mean[i],
                        r.importances_std[i],
                    ]
                )
    all_metrics = pd.DataFrame(
        all_metrics,
        columns=["feature", "metric", "mean_reduction", "std_reduction"],
    )
    return all_metrics


def chi_square_test(X, y, nbins=10):
    """

    :param X:
    :param y:
    :return:
    """
    import scipy.stats as sst

    features = X.columns
    result = []
    for feat in features:
        print(feat)
        var1 = X[feat]
        var2 = y
        mask = np.logical_not(np.isnan(var1))
        var1 = var1[mask]
        var2 = var2[mask]
        X_ca = pd.qcut(
            var1, nbins, labels=False, duplicates="drop"
        )  # discretize into 10 classes
        Y_ca = pd.qcut(var2, nbins, labels=False, duplicates="drop")
        crosstab = pd.crosstab(X_ca, Y_ca)
        chi2, p, _, _ = sst.chi2_contingency(crosstab)
        result.append([feat, chi2, p])
    result = pd.DataFrame(result, columns=["feature", "chi2", "pvalue"])
    return result
