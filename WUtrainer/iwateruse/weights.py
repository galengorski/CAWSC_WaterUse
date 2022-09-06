import numpy as np


def generate_weights_ones(model):
    w_train = np.ones(shape=(len(model.y_train)))
    w_test = np.ones(shape=(len(model.y_test)))
    return w_train, w_test


def generate_weights_by_category(model, col):
    w_train = np.ones(shape=(len(model.y_train)))
    w_test = np.ones(shape=(len(model.y_test)))
    return w_train, w_test
