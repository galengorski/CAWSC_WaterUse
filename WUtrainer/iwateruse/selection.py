import os
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
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5)

    # define Boruta feature selection method
    feat_selector = BorutaPy(estimator, n_estimators='auto', verbose=2, random_state=1)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(X.values, y.ravel())

    # check selected features - first 5 features are selected
    feat_selector.support_

    # check ranking of features
    feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    X_filtered = feat_selector.transform(X)
    return X.columns[feat_selector.support_]

def permutation_selection(X, y, estimator, scoring):

    from sklearn.inspection import permutation_importance

    # compute baseline perfromance using cross validation
    r = permutation_importance(estimator, X, y, scoring = scoring,
                               n_repeats=30,
                               random_state=0)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{diabetes.feature_names[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")

    pass