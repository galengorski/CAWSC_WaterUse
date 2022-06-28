import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from flopy.plot import styles

x = np.random.rand(100)
y = np.random.rand(100)
xlim = [min(x), max(x)]
ylim = [min(y), max(y)]
with styles.USGSPlot():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, "o", mfc="None", mec="k")
    # ax.plot(xol, yol, "o", mfc="None", mec="r")
    # ax.plot(xl, yl, "b--")
    styles.ylabel(ax=ax, label="{} water use in gallons".format(1), fontsize=12)
    styles.xlabel(ax=ax, label="number of {} {}".format(5,5), fontsize=12)
    styles.heading(ax=ax,
                   heading="TX water use for {} group".format(1),
                   idx=0, fontsize=16)
    styles.add_text(ax=ax, text=r"$r^{2}$" + " = {:.2f}".format(5),
                    x=0.75, y=0.90)
    styles.add_text(ax=ax, text="y = {:.2f}x + {:.2e}".format(1, 1),
                    x=0.75, y=0.93)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()
    # plt.savefig(os.path.join(fig_out, fig_base.format(xfield, group)))
    # plt.close(fig)


vv = 1