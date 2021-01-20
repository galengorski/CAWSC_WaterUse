import pandas as pd


FIELDS = (
    "wsa_agidf",
    "sum_gu_pop",
    "x_centroid",
    "y_centroid"
)


def get_input_data(f):
    """
    Method to read and clean input data for processing in outlier detection
    work

    Parameters
    ----------
    f : str
        csv file name to be imported by pandas

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
    return df
