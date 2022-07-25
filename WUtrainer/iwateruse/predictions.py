import os
import pandas as pd


def upscale_to_area(df, pred_column = 'per_capita', scale = 'HUC2'):
    """

    :param df: prediction dataframe
    :param pred_column: column with predictions
    :param scale: column that is used to scale data
    :return: dataframe of upscaled data
    """

    if not("Year" in df.columns):
        raise ValueError("The 'Year' column is not in the dataframe ")

    pred_type = 'annual'
    groupby = [scale, 'Year']
    if 'Month' in df.columns:
        pred_type = 'monthly'
        groupby = [scale, 'Year', 'Month']

    df_ = df[ groupby + [pred_column]].copy()
    scaled_df = df_.groupby(by = groupby).mean()

    df_ = df[groupby + ['pop',pred_column]].copy()
    df_['total_wu_pred'] = df_['pop'] * df_[pred_column]
    df2 = df_.groupby(by = groupby).sum()

    scaled_df['total_wu_pred'] = df2['total_wu_pred']

    return scaled_df


