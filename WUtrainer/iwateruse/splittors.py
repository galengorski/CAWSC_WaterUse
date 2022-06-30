import os, sys
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, GridSearchCV


def random_split(model, args):
    """

    :param model: an object with model information
    :param args: dict with two keys:
                'frac': fraction of samples used in the training
                'seed': seed number
    :return:
    """
    model.log.info("\n\n\n Splitting the training dataset")
    frac = args['frac']
    model.log.info("Testing fraction is {}".format(1-frac))
    seed = args['seed']
    model.log.info("Seed number {}".format(seed))
    test_size = 1 - frac


    df = model.df_train
    train_dataset, test_dataset = train_test_split(df, test_size=test_size, random_state=seed)
    model.train_dataset = train_dataset
    model.test_datset = test_dataset

    model.log.to_table(train_dataset, title = "Training Dataset" )
    model.log.to_table(test_dataset, title="Testing Dataset")

    # features = list(df.columns)
    # features.remove(model.target)
    # X = df[features]
    # y = df[model.target]
    #
    #
    # model.X = X
    # model.y = y
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    # model.X_train = X_train
    # model.X_test = X_test
    # model.y_train = y_train
    # model.y_test = y_test

def stratified_split(model, test_size = 0.3,  id_column = 'HUC1', seed = 123):
    """

    :param model:
    :param args:
    :return:
    """

    df = model.df_train
    features = df.columns.to_list()
    target = model.target
    feats = features.remove(target)
    X = df[features]
    y = df[model.target]

    model.X = X
    model.y = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle = True,
                                                        random_state=seed, stratify=df[id_column])
    model.X_train = X_train
    model.X_test = X_test
    model.y_train = y_train
    model.y_test = y_test

    return X_train, X_test, y_train, y_test

def split_by_id(model, args):
    """
    Split sample such that an id specified in id_column
    exist either in training or testing. Not in both
    :param model: model object
    :param id_column: column with ids to be used in the split
    :param frac: frac of training sample
    :param seed: random seed
    :return:
    """

    frac = args['frac']
    seed = args['seed']
    id_column = args['id_column']
    test_size = 1 - frac

    df = model.df_train
    ids = pd.DataFrame(df[id_column].unique())
    train_ids = ids.sample(frac=frac, random_state=seed)
    test_ids= ids.drop(train_ids.index)

    model.X_train = df[df[id_column].isin(train_ids[0])]
    model.y_train = model.X_train[model.target]

    model.X_test = df[df[id_column].isin(test_ids[0])]
    model.y_test = model.X_test[model.target]



