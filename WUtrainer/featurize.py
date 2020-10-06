import os, sys
import pandas as pd
import numpy as np
import tsfel

def add_dayOfweek(X):
    X_ = X.copy()
    X_['DoW'] = X_['Date'].dt.dayofweek
    return X_

def add_holidays(X):
    X_ = X.copy()
    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
    start_date = X_['Date'].min()
    end_date = X_['Date'].max()
    cal = calendar()
    holidays = cal.holidays(start = start_date, end = end_date)
    X_['Holiday'] = X_['Date'].isin(holidays)
    X_['Holiday'] = X_['Holiday'].astype('int')
    return X_

def add_per_capita_wu(X):
    X_ = X.copy()
    X_['per_capita_wu'] = X_['wu_rate']/X_['population']
    return X_

def drop_na_features(X, y):
    xx = 1;
    pass

def drop_na_targets(X, y):
    pass



