import os

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, max_error, mean_squared_error, r2_score
try :
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
except:
    from sklearn.metrics import mean_absolute_error,  median_absolute_error

def generate_metrics(y_true, y_pred):
    scores = []
    names = []

    scores.append(r2_score(y_true=y_true, y_pred=y_pred))
    names.append("r2")

    scores.append(mean_squared_error(y_true=y_true, y_pred=y_pred))
    names.append("mse")

    scores.append(np.power(mean_squared_error(y_true=y_true, y_pred=y_pred), 0.5))
    names.append("rmse")

    scores.append(mean_absolute_error(y_true=y_true, y_pred=y_pred))
    names.append("mae")

    scores.append(median_absolute_error(y_true=y_true, y_pred=y_pred))
    names.append("mdae")

    try:
        scores.append(mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred))
        names.append("mape")
    except:
        print("Mape is not available in this sklearn")

    scores.append(max_error(y_true=y_true, y_pred=y_pred))
    names.append("maxe")

    scores.append(explained_variance_score(y_true=y_true, y_pred=y_pred))
    names.append("evar")

    df = pd.DataFrame([scores], columns=names)
    return df

def generat_metric_by_category(estimator, X, y_true, category = 'HUC2'):
    ids = X[category].unique()
    all_metrics = []
    for id in ids:
        mask = X[category] == id
        X_ = X[mask].copy()
        y_hat = estimator.predict(X_)
        df_ = generate_metrics(y_true[mask],y_hat)
        df_[category] = id
        all_metrics.append(df_)
    all_metrics = pd.concat(all_metrics)
    return all_metrics










