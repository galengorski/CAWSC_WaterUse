import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
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
    "dec_norm",
)


def describe_modality(record, inflection="max", max_pts=2):

    ts = np.array([record[c] for c in columns])
    intervals = UniDip(ts).run()
    if len(intervals) > 1:
        plt.plot(range(12), ts)
        plt.show()
        print("break")
    tmp = list(sorted(ts))[::-1]
    ts = list(ts)
    idx0 = ts.index(tmp[0])
    idx1 = ts.index(tmp[3])
    midx = ts.index(tmp[-1])

    bimodal = False
    bi_label = ""
    if abs(idx0 - idx1) > 3:
        bimodal = True
        bi_label = columns[idx1]

    return bimodal, columns[idx0], columns[midx], bi_label


if __name__ == "__main__":
    ws = os.path.abspath(os.path.dirname(__file__))
    norm_file = os.path.join(
        ws, "..", "output", "monthly", "wu_mean_spatial_2010_normalized.csv"
    )

    df = pd.read_csv(norm_file)
    df = df.fillna(value=0.0)
    for iloc, row in df.iterrows():
        result = describe_modality(row)
        print("break")
