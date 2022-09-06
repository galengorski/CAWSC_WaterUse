import pandas as pd
import numpy as np


def compute_per_capita(model, args=None):
    """ """
    df = model.df_train
    df[model.target] = df["wu_rate"] / df["pop"]
    model.df_train = df
    model.log.info("Target feature {} is added".format(model.target))
    model.log.to_table(model.df_train)
