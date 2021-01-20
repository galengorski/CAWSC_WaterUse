import pandas as pd


FIELDS = (
    "wsa_agidf",
    "sum_gu_pop",
    "x_centroid",
    "y_centroid",
    "tot_wd_mgd",
)


def get_input_data(f, dataframe=None):
    """
    Method to read and clean input data for processing in outlier detection
    work

    Parameters
    ----------
    f : str
        csv file name to be imported by pandas
    df : None or pd.Dataframe
        if not none we can join by wsa

    Returns
    -------
        pd.DataFrame
    """
    df = pd.read_csv(f)

    drop = []
    for col in list(df):
        if col.lower() not in FIELDS:
            drop.append(col)

    lowered = {i: i.lower() for i in list(df)}
    df = df.drop(columns=drop)
    df = df.rename(columns=lowered)

    if dataframe is not None:
        df = pd.merge(
            left=dataframe,
            right=df,
            left_on='wsa_agidf',
            right_on='wsa_agidf'
        )
    return df
