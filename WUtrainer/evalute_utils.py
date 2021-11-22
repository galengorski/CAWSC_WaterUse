import os, sys
import pandas as pd
from sklearn.model_selection import cross_val_score


def cross_validate(X, y, baseline_estimator, number_of_foldes = 5):
    # Train and score baseline model
    baseline_score = cross_val_score(
        baseline_estimator, X, y, cv=number_of_foldes, scoring="neg_mean_absolute_error"
    )
    mean_score = baseline_score.mean()
    std_score = baseline_score.std()

    return mean_score, std_score