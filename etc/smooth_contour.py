import os, sys
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.ndimage.filters import gaussian_filter

df_feat = pd.read_csv(
    r"C:\work\water_use\dataset\spatial_analysis\df_pc_percentile_N.csv"
)
df = df_feat[["tmmx_warm", "pr", "pc_median"]]
df = df.dropna()
x = df["tmmx_warm"].values
y = df["pr"].values
values = df["pc_median"].values

mask = x > 286
x = x[mask]
y = y[mask]
z = values[mask]

nbins = 100

xi = np.linspace(np.min(x), np.max(x), nbins)
yi = np.linspace(np.min(y), np.max(y), nbins)

triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, z)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)
cntr2 = plt.tricontourf(x, y, z, levels=5, cmap="RdBu_r")
bin_means = binned_statistic_2d(
    x, y, z, bins=nbins, statistic="median"
).statistic


def interpolate_nans(X):
    """Overwrite NaNs with column value interpolations."""
    for j in range(X.shape[1]):
        mask_j = np.isnan(X[:, j])
        X[mask_j, j] = np.interp(
            np.flatnonzero(mask_j), np.flatnonzero(~mask_j), X[~mask_j, j]
        )
    return X


interpolate_nans(bin_means)
xx = 1
