import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from matplotlib.backends.backend_pdf import PdfPages
import shap


def pdp_plots_by_huc2(estimator, df, fn, all_features, interp_feat, huc2=[1]):
    common_params = {
        # "subsample": 50,
        "n_jobs": 5,
        # "grid_resolution": 20,
        # "random_state": 0,
    }
    hucs = df["HUC2"].unique()
    with PdfPages(fn) as pdf:
        for feat in interp_feat:
            print("Explaining : {}".format(feat))
            all_data = {}

            for h in huc2:
                interp_feat_ = [feat]
                XX = df[df["HUC2"] == h].copy()
                XX = XX[~XX[feat].isna()]

                PartialDependenceDisplay.from_estimator(
                    features=interp_feat_,
                    estimator=estimator,
                    X=XX[all_features],
                    grid_resolution=50,
                    **common_params
                )
                ax = plt.gca()
                x_ = ax.lines[0].get_xdata()
                y_ = ax.lines[0].get_ydata()
                all_data[h] = [x_, y_]
                plt.close()

            ifig = plt.figure(figsize=(5, 5))
            axx = ifig.gca()
            for h in huc2:
                x_, y_ = all_data[h]
                axx.plot(x_, y_, label=h)

            ifig.suptitle(feat, fontsize=14)
            plt.xlabel("Feature Value")
            plt.ylabel("Annual Water Use\n (GPCD)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def pdp_plots(estimator, df, fn, features):
    common_params = {
        # "subsample": 50,
        "n_jobs": 5,
        # "grid_resolution": 20,
        # "random_state": 0,
    }

    with PdfPages(fn) as pdf:
        for feat in features:
            print("Explaining : {}".format(feat))
            interp_feat = [feat]
            XX = df[~df[feat].isna()]
            ifig = plt.figure(figsize=(5, 5))

            PartialDependenceDisplay.from_estimator(
                ax=ifig.gca(),
                features=interp_feat,
                estimator=estimator,
                X=XX[features],
                grid_resolution=50,
                **common_params
            )

            ifig.suptitle(feat, fontsize=14)
            plt.xlabel("Feature Value")
            plt.ylabel("Annual Water Use\n (GPCD)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()


def shap_plots(model, estimator, fn, features, data_frac=0.5):

    df = model.df_train
    with PdfPages(fn) as pdf:

        # Global
        X100 = df.sample(frac=data_frac).copy()
        X100 = X100[features]
        explainer = shap.Explainer(estimator, X100)
        shap_values = explainer(X100)
        shap.summary_plot(
            shap_values, X100, max_display=15, show=False, color_bar=False
        )
        cb = plt.colorbar(shrink=0.2, label="Feature Value")
        cb.ax.tick_params(labelsize=500)
        cb.set_ticks([])
        cb.ax.text(
            0.5, -0.01, "Low", transform=cb.ax.transAxes, va="top", ha="center"
        )
        cb.ax.text(
            0.5,
            1.01,
            "High",
            transform=cb.ax.transAxes,
            va="bottom",
            ha="center",
        )
        plt.gca().figure.suptitle("Global SHAP Values", fontsize=14)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        for feat in features:
            try:
                fig = plt.figure()
                cax = fig.gca()
                shap.plots.scatter(
                    shap_values[:, feat],
                    color=shap_values,
                    ax=cax,
                    hist=False,
                    show=False,
                )
                fig.suptitle("SHAP Values for {}".format(feat), fontsize=14)
                vrange = np.abs(
                    X100[feat].quantile(0.90) / X100[feat].quantile(0.1)
                )
                if vrange > 0:
                    pvals = X100[feat].values
                    pvals = pvals[pvals != -np.inf]
                    pvals = pvals[pvals != np.inf]
                    max_limit = np.nanmax(pvals)
                    min_limit = np.nanmin(pvals)
                    plt.xlim([min_limit, max_limit])
                    if (max_limit > 0) & (min_limit > 0):
                        plt.xscale("log")
                parname = fig.axes[-1].get_ylabel()
                fig.axes[-1].remove()
                PCM = cax.get_children()[2]
                cb = plt.colorbar(PCM, ax=cax, shrink=1.0, label=parname)
                cb.ax.tick_params(labelsize=10)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            except:
                print("Issues with SHAP values for {}".format(feat))

        # per HUC2
        hru2s = np.sort(df["HUC2"].unique())
        for h in hru2s:
            print("Explaining HUC2: {}".format(h))
            XX = df[df["HUC2"] == h]
            XX = XX[features]
            explainer = shap.Explainer(estimator, XX)
            shap_values = explainer(XX)
            shap.summary_plot(
                shap_values, XX, max_display=15, color_bar=False, show=False
            )
            cb = plt.colorbar(shrink=0.2, label="Feature Value")
            cb.ax.tick_params(labelsize=12)
            cb.set_ticks([])
            cb.ax.text(
                0.5,
                -0.01,
                "Low",
                transform=cb.ax.transAxes,
                va="top",
                ha="center",
            )
            cb.ax.text(
                0.5,
                1.01,
                "High",
                transform=cb.ax.transAxes,
                va="bottom",
                ha="center",
            )
            fig = plt.gca().figure
            fig.suptitle("SHAP Values for HUC2 = {}".format(h), fontsize=12)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
