import os
import numpy as np
import pandas as pd
from iwateruse import denoise


def add_normal_noise_to_col(df, col, mu=0, seg=1):
    N = len(df)
    noise = np.random.normal(mu, seg, N)
    df[col] = df[col] + noise
    return df


def add_outlier_samples(df, frac=0.1):
    """
    We assume that df set has x1,x2,..., y
    :param df:
    :return:
    """
    Nnoise = int(frac * len(df))
    df_noise = pd.DataFrame(columns=df.columns)
    for col in df_noise:
        min_val = df[col].min()
        max_val = df[col].max()
        noise = np.random.rand(Nnoise, 1)
        df_noise[col] = min_val + noise.flatten() * (max_val - min_val)

    df = pd.concat([df, df_noise], axis=0)
    df = df.reset_index().drop(["index"], axis=1)
    return df


# ====================================
# 1D test problem
# ====================================
if 1:
    df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=["x1"])
    df1["y"] = (
        -3 * df1["x1"] + np.power(df1["x1"], 2) + np.power(df1["x1"] / 3, 3.0)
    )
    df = df1

    # addnoise on data
    add_normal_noise_to_col(df, "y", mu=0, seg=1500)

    # add noise
    df = add_outlier_samples(df, frac=0.5)
    df["id"] = df.index.values

    denoise.purify(
        df, target="y", features=["x1"], col_id=["id"], max_iterations=400
    )
    stop = 1

# ====================================
# High D test problem
# ====================================
if 0:
    np.random.seed(123)
    samples = 5000
    nfeatures = 10
    features = ["x{}".format(i) for i in range * nfeatures]
    exponents = [5, 2, 3, 1, 7, 4, 0, -1, 6, 2]
    coeff = [0.5, -2, 3, 4, -2.5, 0, 7, 2, 3.5]
    df1 = pd.DataFrame(columns=[features])
    for feat in features:
        df1[feat] = 10 * np.random.rand(samples)

    df1["y"] = (
        -3 * df1["x1"] + np.power(df1["x1"], 2) + np.power(df1["x1"] / 3, 3.0)
    )
    df = df1

    # addnoise on data
    add_normal_noise_to_col(df, "y", mu=0, seg=1500)

    # add noise
    df = add_outlier_samples(df, frac=0.5)
    df["id"] = df.index.values

    purify(df, target="y", features=["x1"], col_id=["id"], max_iterations=400)
    stop = 1
