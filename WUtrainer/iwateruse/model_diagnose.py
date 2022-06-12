import os

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, max_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error

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

    scores.append(mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred))
    names.append("mape")

    scores.append(max_error(y_true=y_true, y_pred=y_pred))
    names.append("maxe")

    scores.append(explained_variance_score(y_true=y_true, y_pred=y_pred))
    names.append("evar")

    df = pd.DataFrame([scores], columns=names)
    return df








