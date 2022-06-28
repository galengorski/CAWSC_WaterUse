import os
import numpy as np
import matplotlib.pyplot as plt
from flopy.plot import styles
from sklearn.metrics import r2_score, mean_squared_error

def one_to_one(y_actual, y_hat, heading, xlabel, ylabel, figfile):
    with styles.USGSPlot():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(y_actual, y_hat, marker="o", s=20, alpha=0.5)
        plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r')
        accuracy = r2_score(y_actual, y_hat)
        rmse = np.power(mean_squared_error(y_actual,y_hat), 0.5)
        styles.heading(ax=ax,
                       heading=heading,
                       idx=0, fontsize=16)
        msg = r"$r^{2}$" + " = {:.2f}".format(np.round(accuracy, 2))
        msg = msg + "\nrmse = {} GPCD".format(np.round(rmse, 2))
        styles.add_text(ax=ax, text=msg,
                        x=0.02, y=0.92, backgroundcolor = 'white')

        styles.xlabel(ax = ax, fontsize=12,  label = xlabel)
        styles.ylabel(ax = ax, fontsize=12,  label = ylabel)
        plt.savefig(figfile)
        plt.close(fig)


def feature_importance(estimator, max_num_feature , type , figfile):
    from xgboost import plot_importance
    with styles.USGSPlot():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        f = plot_importance(ax = ax, booster=estimator, max_num_features=max_num_feature, importance_type=type)
        styles.heading(ax=ax,
                       heading='Feature Importance',
                       idx=0, fontsize=16)
        styles.xlabel(ax=ax, fontsize=16, label=type)
        styles.ylabel(ax=ax, fontsize=16, label= 'Feature')
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plt.savefig(figfile)
        plt.close(fig)



