import pandas as pd
import numpy as np
import os


FWS = os.path.abspath(r"C:\work\water_use\dataset\ml_county_to_sa\ml_county_to_sa")
files = [os.path.join(FWS,  "county_data",
                      "clean_data", "income_data.csv"),
         os.path.join(FWS,  "county_data",
                      "clean_data", "population_data.csv"),
         os.path.join(FWS,  "county_data",
                      "clean_data", "economic_profile_data.csv"),
         os.path.join(FWS,  "county_data",
                      "clean_data", "education_data.csv"),
         os.path.join(FWS,  "county_data",
                      "clean_data", "employment_data.csv"),
         os.path.join(FWS,  "county_data",
                      "clean_data", "gdp_data.csv"),
         os.path.join(FWS,  "county_data",
                      "clean_data", "gini_index_data.csv"),
         os.path.join(FWS,  "county_data",
                      "clean_data", "poverty_data.csv"),
         os.path.join(FWS,  "county_data",
                      "clean_data", "ruralurbancodes2003.csv"),
         os.path.join(FWS,  "county_data",
                      "clean_data", "ruralurbancodes2013.csv"),
         os.path.join(FWS,  "county_data",
                      "clean_data", "unemployment_data.csv")]

dbf_file = os.path.join(FWS,
                        "wsa_county_map",
                        "wsa_county_map.dat.csv")


def load_training_data(f):
    """
    Method to load existing training data

    Parameters
    ----------
    f : str
        file name

    Returns
    -------
        df.DataFrame
    """
    df = pd.read_csv(f)
    return df


def load_parameters(f):
    """
    Method to load the parameter list for joins

    Parameters
    ----------
    f : str

    Returns
    -------
        list
    """
    parameters = []
    with open(f) as foo:
        for line in foo:
            line = line.strip()
            if line:
                if line.startswith("#"):
                    continue
                parameters.append(line)

    return parameters


def join_data(wsa, df, left_on='fips', right_on='fips'):
    """
    Method to join dataframe and remove duplicate data

    Parameters
    ----------
    wsa : pd.DataFrame
    df : pd.DataFrame
    left_on : str
        left join field
    right_on : str
        right join field

    Returns
    -------
        pd.DataFrame
    """
    df = df.drop_duplicates(subset=right_on)
    merged = pd.merge(left=wsa,
                      right=df,
                      how="left",
                      left_on=left_on,
                      right_on=right_on)

    cols = list(merged)
    # drop duplicate data
    drop_list = []
    for c in cols:
        if c.endswith("_x") or c.endswith("_y"):
            drop_list.append(c)

    merged = merged.drop(columns=drop_list)

    return merged


def load_all(parameters):
    """
    Method to load all composite csv files, wu_annual_training file,
    and map data into pandas csvs

    Returns
    -------
        None
    """
    dflist = []
    for f in files:
        dflist.append(load_csv(f, parameters))

    return dflist


def load_wsas(parameters):
    """

    Returns
    -------
        None
    """
    parameters = [i.lower() for i in parameters] + ['geoid', 'fips']
    df = pd.read_csv(dbf_file)
    cols = {i: i.lower() for i in list(df)}
    df = df.rename(columns=cols)
    df = df.rename(columns={'geoid': 'fips'})

    drop_list = []
    for col in list(df):
        if col not in parameters:
            drop_list.append(col)

    df = df.drop(columns=drop_list)

    for col in list(df):
        if col not in ('geoid', 'fips', 'wsa_agidf'):
            df = df.astype({col: 'float'})

    return df


def load_csv(f, parameters):
    """
    Method to load csv files into a pandas dataframe

    Parameters
    ----------
    f : str
        file name

    Returns
    -------
        pd.DataFrame
    """
    parameters = list(parameters) + ['fips', 'sys_']
    parameters = [i.lower() for i in parameters]
    df = pd.read_csv(f)
    cols = {i: i.lower() for i in list(df)}
    df = df.rename(columns=cols)

    drop_list = []
    for col in list(df):
        if col not in parameters:
            drop_list.append(col)

    df = df.drop(columns=drop_list)

    for col in list(df):
        if col not in ('geoid', 'fips', 'wsa_agidf'):
            df = df.astype({col: 'float'})

    return df


def _format_data(f):
    """
    Method to format non-standard dataframes into a pandas readable
    object.

    Parameters
    ----------
    f : str
        file name
    """
    ws, fname = os.path.split(f)
    ofname = fname.split(".")[0] + "_clean.csv"
    d = {'fips': []}
    df = pd.read_csv(f)
    attrs = df.Attribute.unique()
    fips = df.FIPStxt.unique()
    for attr in attrs:
        d[attr] = []

    for fip in fips:
        df1 = df[df.FIPStxt == fip]
        for k in d.keys():
            if k == "fips":
                if df1.FIPStxt.values[0] not in d['fips']:
                    d[k].append(df1.FIPStxt.values[0])
            else:
                dft = df1[df1.Attribute == k]
                if len(dft) == 0:
                    d[k].append(np.nan)
                else:
                    d[k].append(dft.Value.values[0])

    dfout = pd.DataFrame.from_dict(d)
    dfout.to_csv(os.path.join(ws, ofname), index=False)


def _format_data2(f, r0, r1):
    """
    Method 2 to format non-standard dataframes into a pandas readable
    object.

    Parameters
    ----------
    f : str
        file name
    """
    ws, fname = os.path.split(f)
    oname = fname.split(".")[0]
    oname = "_".join(oname.split("_")[:-1]) + "_clean2.csv"
    x = [str(i) for i in range(r0, r1)]

    units = {"Thousands of dollars": "th_dol",
             "Dollars": "dol",
             "Number of jobs": "n_job",
             "Number of persons": "n_person"}

    df = pd.read_csv(f)
    fips = df.GeoFIPS.unique()
    isin = []
    d = {'fips': []}
    for iloc, row in df.iterrows():
        desc = row.Description
        unit = units[row.Unit]
        cbase = " ".join([desc, unit])
        if cbase not in isin:
            isin.append(cbase)
            for year in x:
                k = "{} {}".format(year, cbase)
                d[k] = []

    for fip in fips:
        d['fips'].append(fip)
        df1 = df[df.GeoFIPS == fip]
        hits = ['fips']
        for iloc, row in df1.iterrows():
            desc = row.Description
            unit = units[row.Unit]
            cbase = " ".join([desc, unit])
            for year in x:
                k = "{} {}".format(year, cbase)
                if k in hits:
                    continue
                hits.append(k)
                try:
                    d[k].append(float(row[year]))
                except ValueError:
                    d[k].append(np.nan)

        for k in d.keys():
            if k not in hits:
                d[k].append(np.nan)

    df = pd.DataFrame.from_dict(d)
    df.to_csv(os.path.join(ws, oname), index=False)
