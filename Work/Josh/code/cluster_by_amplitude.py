import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from unidip import UniDip


columns = (
    "jan_norm",
    "feb_norm",
    "mar_norm",
    "apr_norm",
    "may_norm",
    "jun_norm",
    "jul_norm",
    "aug_norm",
    "sep_norm",
    "oct_norm",
    "nov_norm",
    "dec_norm"
)


def describe_amplitude(record):
    ts = [record[c] for c in columns]
    maximum = np.max(ts)
    minimum = np.min(ts)
    amplitude = maximum - minimum
    max_month = ts.index(maximum) + 1
    min_month = ts.index(minimum) + 1
    return maximum, amplitude, max_month, minimum, min_month


def describe_modality(record):

    ts = np.array([record[c] for c in columns])
    intervals = UniDip(ts).run()
    if len(intervals) > 1:
        return 0
    else:
        return 1


if __name__ == "__main__":
    ws = os.path.abspath(os.path.dirname(__file__))
    norm_file = os.path.join(ws, "..", "output", "monthly",
                             "wu_mean_spatial_2010_normalized.csv")

    df = pd.read_csv(norm_file)
    df = df.fillna(value=0.)
    amplitude = []
    peak = []
    max_mo = []
    minimum = []
    min_mo = []
    modal = []
    for iloc, row in df.iterrows():
        result = describe_amplitude(row)
        peak.append(result[0])
        amplitude.append(result[1])
        max_mo.append(result[2])
        minimum.append(result[3])
        min_mo.append(result[4])
        modal.append(describe_modality(row))

    df['max_val'] = peak
    df['amplitude'] = amplitude
    df['max_mo'] = np.array(max_mo) / 12.
    df['minimum'] = minimum
    df['min_mo'] = np.array(min_mo) /12.
    df['freq'] = df["max_mo"] - df["min_mo"]
    df["freq"] = df["freq"].abs() / 12
    df['tot_wu_norm'] = df["tot_wu_mgd"] / df['tot_wu_mgd'].max()
    df['x_norm'] = df['x_centroid'] / df['x_centroid'].max()
    df['y_norm'] = df['y_centroid'] / df['y_centroid'].max()
    df['modal'] = modal

    df = df[df['minimum'] > 0.]
    fields = ['minimum', 'amplitude', 'tot_wu_norm', 'modal',
              'x_norm', 'y_norm']
    km = KMeans(n_clusters=4, n_init=100)
    km.fit(df[fields])
    Xt = km.transform(df[fields])
    labels = km.labels_

    for f in fields:
        for ff in fields:
            plt.scatter(df[f], df[ff], c=labels, cmap='viridis')
            plt.xlabel(f)
            plt.ylabel(ff)
            plt.show()

    for field in fields:
        for i in range(4):
            plt.scatter(df[field], Xt[:, i], c=labels, cmap='viridis')
            plt.xlabel(field)
            plt.ylabel("Dimension {}".format(i + 1))
            plt.show()

    plt.scatter(df["x_centroid"], df['y_centroid'], c=labels, cmap="viridis")
    plt.show()