import numpy as np
import pandas as pd


def pre_train(model, args):
    columns_to_drop = model.columns_to_drop
    if not (model.target in columns_to_drop):
        columns_to_drop.append(model.target)

    if not (model.raw_target in columns_to_drop):
        columns_to_drop.append(model.raw_target)
    temp_drop = []
    for feat in columns_to_drop:
        if feat in model.X_train.columns:
            temp_drop.append(feat)

    model.X_train = model.X_train.drop(temp_drop, axis=1)
    model.X_test = model.X_test.drop(temp_drop, axis=1)
