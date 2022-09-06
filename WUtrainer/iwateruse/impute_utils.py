import os, sys
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def iterative_imputer(df, **kwargs):

    """
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer
    :param df:
    :param kwargs:
    :return:
    """

    X = df.values
    imp = IterativeImputer(**kwargs)
    imp.fit(X)

    mask = np.isnan(X)
    imp_X = imp.transform(X)
    imp_X = pd.DataFrame(imp_X, columns=df.columns)
    return imp_X
