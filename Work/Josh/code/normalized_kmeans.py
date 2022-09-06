from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

interest = (
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
ws = os.path.abspath(os.path.dirname(__file__))
norm_file = os.path.join(
    ws, "..", "output", "monthly", "wu_mean_spatial_2010_normalized.csv"
)

df = pd.read_csv(norm_file)
# df = df[df['tot_wu_mgd'] < 1e+4]
df1 = df[list(interest)]
df1 = df1.fillna(value=0.0)

kmeans = KMeans(n_clusters=4).fit(df1)
print(kmeans.labels_)
df1["cluster"] = kmeans.labels_
for i in interest:
    for k in interest:
        plt.scatter(df1[i], df1[k], c=df1["cluster"], cmap="viridis")
        plt.show()

print("break")
