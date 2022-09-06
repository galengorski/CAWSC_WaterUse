import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
import xgboost as xgb
from numpy.random import choice
import time


# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import HalvingRandomSearchCV


def add_normal_noise_to_col(df, col, mu=0, seg=1):
    N = len(df)
    noise = np.random.normal(mu, seg, N)
    df[col] = df[col] + noise
    return df


def add_outlier_samples(df, frac=0.1):
    """
    We assume that df set has x1,x2,..., y
    :param df:
    :return:
    """
    Nnoise = int(frac * len(df))
    df_noise = pd.DataFrame(columns=df.columns)
    for col in df_noise:
        min_val = df[col].min()
        max_val = df[col].max()
        noise = np.random.rand(Nnoise, 1)
        df_noise[col] = min_val + noise.flatten() * (max_val - min_val)

    df = pd.concat([df, df_noise], axis=0)
    df.reset_index().drop(["index"], axis=1)
    return df


def mse():
    def _mse(y_true, y_pred):
        errors = y_pred - y_true

        grad = 2 * errors
        hess = -2 * np.ones_like(errors)

        return grad, hess

    return _mse


def xgb_estimator(params={}):
    if params is None:
        objective = mse()
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "colsample_bytree": 0.8,
            "learning_rate": 0.20,
            "max_depth": 7,
            "alpha": 100,
            "n_estimators": 500,
            "rate_drop": 0.9,
            "skip_drop": 0.5,
            "subsample": 0.8,
            "reg_lambda": 10,
            "min_child_weight": 5,
            "gamma": 10,
            "max_delta_step": 0,
            "seed": 123,
        }
    results = {}
    gb = xgb.XGBRegressor(**params, evals_result=results)
    return gb


def remove_signal(df_noise, features, test_frac=0.3, signal_model=None):
    df_noise["err"] = np.power(
        (signal_model.predict(df_noise[features]) - df_noise["y"]), 2
    )
    df_noise.sort_values("err", inplace=True)
    del df_noise["err"]
    df_noise.reset_index(inplace=True)
    del df_noise["index"]

    sig_samples = df_noise.index.values[0:5]
    df_signal = pd.concat([df_signal, df_noise.iloc[sig_samples]], axis=0)
    df_signal.reset_index(inplace=True)
    del df_signal["index"]
    df_noise.drop(index=sig_samples, inplace=True)
    df_noise.reset_index(inplace=True)
    del df_noise["index"]
    pass


def cross_validate(new_df, features, score):
    score = "neg_mean_absolute_percentage_error"
    gb = xgb_estimator(params=None)
    signal_scores = cross_val_score(
        gb, new_df[features], new_df["y"], scoring=score, cv=5
    )
    new_score = np.mean(signal_scores)
    return new_score


def propose_sample_removal(df_, features, test_frac=0.3, isnoise=0):
    """
    - Add or remove samples to a dataset
    - check score before and after alteration
    :return:
    """
    # shuffle and split
    df = df_.copy()
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df.sample(frac=(1 - test_frac))
    test_df = df.drop(index=train_df.index)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # fit a model to flag possible samples to removing
    gb1 = xgb_estimator(params=None)
    gb1.fit(train_df[features], train_df["y"])
    err = np.power((gb1.predict(test_df[features]) - test_df["y"]), 2.0)

    if isnoise == 0:
        q70 = np.quantile(err, 0.7)
        w = np.zeros_like(err)
        w[err >= q70] = 1
        w = w / np.sum(w)
    else:
        q30 = np.quantile(err, 0.7)
        w = np.zeros_like(err)
        w[err < q30] = 1
        w = w / np.sum(w)

    # propose samples to remove
    nn = int(len(test_df.index.values) * 0.05)
    move_samples = choice(test_df.index.values, nn, p=w, replace=False)

    df_to_move = test_df.iloc[move_samples]
    test_df_new = test_df.drop(index=move_samples)
    new_df = pd.concat([train_df, test_df_new], axis=0)
    score = "neg_mean_absolute_percentage_error"
    gb = xgb_estimator(params=None)
    signal_scores = cross_val_score(
        gb, new_df[features], new_df["y"], scoring=score, cv=5
    )
    new_score = np.mean(signal_scores)
    return new_score, new_df, df_to_move, gb1


def propose_sample_removal2(df_, features, test_frac=0.3, isnoise=0):
    """
    - Add or remove samples to a dataset
    - check score before and after alteration
    :return:
    """
    # shuffle and split
    df = df_.copy()
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df.sample(frac=(1 - test_frac))
    test_df = df.drop(index=train_df.index)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # fit a model to flag possible samples to removing
    gb1 = xgb_estimator(params=None)
    gb1.fit(train_df[features], train_df["y"])
    err = np.power((gb1.predict(test_df[features]) - test_df["y"]), 2.0)

    if isnoise == 0:
        q70 = np.quantile(err, 0.7)
        w = np.zeros_like(err)
        w[err >= q70] = 1
        w = w / np.sum(w)
    else:
        q30 = np.quantile(err, 0.7)
        w = np.zeros_like(err)
        w[err < q30] = 1
        w = w / np.sum(w)

    # propose samples to remove
    nn = int(len(test_df.index.values) * 0.05)
    move_samples = choice(test_df.index.values, nn, p=w, replace=False)

    df_to_move = test_df.iloc[move_samples]
    test_df_new = test_df.drop(index=move_samples)
    new_df = pd.concat([train_df, test_df_new], axis=0)
    score = "neg_mean_absolute_percentage_error"
    gb = xgb_estimator(params=None)
    signal_scores = cross_val_score(
        gb, new_df[features], new_df["y"], scoring=score, cv=5
    )
    new_score = np.mean(signal_scores)
    return new_score, new_df, df_to_move, gb1


def algo(df):
    score = "neg_mean_absolute_percentage_error"
    figure, ax = plt.subplots(figsize=(4, 5))
    figure2, ax2 = plt.subplots(figsize=(4, 5))
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df_signal = df.sample(frac=0.5)
    df_noise = df.drop(index=df_signal.index)

    features = [feat for feat in df.columns if "x" in feat]
    gb = xgb_estimator(params=None)

    signal_scores = cross_val_score(
        gb, df_signal[features], df_signal["y"], scoring=score, cv=5
    )

    noise_scores = cross_val_score(
        gb, df_noise[features], df_noise["y"], scoring=score, cv=5
    )

    if np.mean(signal_scores) < np.mean(noise_scores):
        signal_average_scroe = [np.mean(noise_scores)]
        noise_average_score = [np.mean(signal_scores)]
        temp = df_signal.copy()
        df_signal = df_noise.copy()
        df_noise = temp.copy()
        del temp
    else:
        signal_average_scroe = [np.mean(signal_scores)]
        noise_average_score = [np.mean(noise_scores)]

    plt.ion()

    for iter in range(200):

        # remove outliers from  signal
        new_score, new_df, df_move = propose_sample_removal(
            df_signal, features, test_frac=0.3, isnoise=0
        )

        frac_of_noise = len(df_noise) / (len(df_signal) + len(df_noise))
        signal2noise_leakage = 1.6 * len(df_signal) / len(df_noise)
        if new_score >= signal2noise_leakage * signal_average_scroe[-1]:
            new_score = (5 * signal_average_scroe[-1] + new_score) / 6
            df_noise = pd.concat([df_noise, df_move], axis=0)
            df_signal = new_df
            signal_average_scroe.append(new_score)

        # remove signal from outlier
        new_score, new_df, df_move = propose_sample_removal(
            df_noise, features, test_frac=0.3, isnoise=1
        )
        noise2signal_leakage = 0.6 * len(df_signal) / len(df_noise)
        if new_score < noise2signal_leakage * noise_average_score[-1]:
            new_score = (5 * noise_average_score[-1] + new_score) / 6
            df_signal = pd.concat([df_signal, df_move], axis=0)
            df_noise = new_df
            noise_average_score.append(new_score)
        (plot1,) = ax.plot(np.log10(np.abs(noise_average_score)), "r")
        (plot2,) = ax.plot(np.log10(np.abs(signal_average_scroe)), "b")

        ax2.clear()
        (plot4,) = ax2.plot(
            df_noise[features],
            df_noise["y"],
            c="r",
            alpha=0.5,
            linestyle="None",
            marker="o",
        )
        (plot3,) = ax2.plot(
            df_signal[features],
            df_signal["y"],
            c="b",
            alpha=0.5,
            linestyle="None",
            marker="o",
        )

        plot1.set_ydata(np.log10(np.abs(noise_average_score)))
        plot2.set_ydata(np.log10(np.abs(signal_average_scroe)))

        plot1.set_ydata(np.log10(np.abs(noise_average_score)))
        plot2.set_ydata(np.log10(np.abs(signal_average_scroe)))

        plot3.set_ydata(df_signal["y"])
        plot4.set_ydata(df_noise["y"])

        plot3.set_xdata(df_signal[features])
        plot4.set_xdata(df_noise[features])

        figure2.canvas.draw()
        plt.title(iter)
        figure2.canvas.flush_events()
        time.sleep(0.1)

        figure.canvas.draw()
        plt.title("Noise = {}".format(len(df_noise) / len(df_signal)))
        figure.canvas.flush_events()
        time.sleep(0.1)

        ax.set_title(str(iter))
        plt.show()
        return None, None

    x = 1


def visu():
    pass


def visulize(
    initialize=True,
    fig=None,
    axs=None,
    noise_average_score=None,
    signal_average_scroe=None,
    df_signal=None,
    df_noise=None,
    features=None,
    iter=1,
    frac_noise_list=None,
    df_move=None,
):
    if initialize:
        fig, axs = plt.subplots(ncols=2, nrows=2)
        return fig, axs

    plt.ion()
    (plot1,) = axs[0, 0].plot(
        np.log10(np.abs(noise_average_score)),
        "r",
        label="Error in Noise Model",
    )
    (plot2,) = axs[0, 0].plot(
        np.log10(np.abs(signal_average_scroe)),
        "b",
        label="Error in Signal Model",
    )
    if iter == 0:
        axs[0, 0].legend(loc="upper right", bbox_to_anchor=(0, 0.5))

    axs[0, 1].clear()
    (plot4,) = axs[0, 1].plot(
        df_noise[features].values,
        df_noise["y"],
        c="r",
        alpha=0.5,
        linestyle="None",
        marker="o",
        label="noise",
    )
    (plot3,) = axs[0, 1].plot(
        df_signal[features].values,
        df_signal["y"],
        c="b",
        alpha=0.5,
        linestyle="None",
        marker="o",
        label="signal",
    )
    (plot3_1,) = axs[0, 1].plot(
        df_move[features].values,
        df_move["y"],
        c="k",
        linestyle="None",
        marker=".",
        label="signal",
    )

    (plot5,) = axs[1, 0].plot(frac_noise_list, c="k")

    axs[0, 1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plot1.set_ydata(np.log10(np.abs(noise_average_score)))
    plot2.set_ydata(np.log10(np.abs(signal_average_scroe)))
    #
    # plot1.set_ydata(np.log10(np.abs(noise_average_score)))
    # plot2.set_ydata(np.log10(np.abs(signal_average_scroe)))

    plot3.set_ydata(df_signal["y"])
    plot4.set_ydata(df_noise["y"])
    plot3_1.set_ydata(df_move["y"])

    plot3.set_xdata(df_signal[features])
    plot4.set_xdata(df_noise[features])
    plot3_1.set_xdata(df_move[features])

    plot5.set_ydata(frac_noise_list)

    axs[0, 0].set_title(str(iter))
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)

    plt.tight_layout()
    plt.show()

    pass


def smoother(old_value, new_value, weight):
    score = (weight * old_value + new_value) / (weight + 1)
    return score


def algo_mcmc(df):
    score = "neg_mean_absolute_percentage_error"
    fig, axs = visulize(initialize=True)

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df_signal = df.sample(frac=0.5)
    df_noise = df.drop(index=df_signal.index)

    features = [feat for feat in df.columns if "x" in feat]
    gb = xgb_estimator(params=None)

    signal_scores = cross_val_score(
        gb, df_signal[features], df_signal["y"], scoring=score, cv=5
    )

    noise_scores = cross_val_score(
        gb, df_noise[features], df_noise["y"], scoring=score, cv=5
    )

    signal_average_scroe = [np.mean(signal_scores)]
    noise_average_score = [np.mean(noise_scores)]

    traget_noise_signal_ratio = 0.2

    N = 400
    frac_noise_list = []
    for iter in range(N):
        out_df_mov = pd.DataFrame(columns=df_signal.columns)
        # remove outliers from  signal
        new_score, new_df, df_move, signal_model = propose_sample_removal(
            df_signal, features, test_frac=0.3, isnoise=0
        )
        frac_of_noise = len(df_noise) / len(df_signal)
        gamma = signal_average_scroe[-1] / new_score
        # gamma = gamma * (traget_noise_signal_ratio/frac_of_noise)
        u = np.random.rand(1)
        # signal_frac = len(df_signal) / (len(df_signal) + len(df_noise))
        window = 1 - 0.75 * iter / N
        if window < 0.1:
            window = 0.1

        signal_frac = len(df_signal) / len(df_noise)
        if u <= gamma * window:
            # if gamma >= 0.80:
            if signal_frac > 0.3:
                # new_score = (1* signal_average_scroe[-1] + new_score) / 2
                new_score = smoother(signal_average_scroe[-1], new_score, 5)
                df_noise = pd.concat([df_noise, df_move], axis=0)
                df_signal = new_df
                signal_average_scroe.append(new_score)
                out_df_mov = df_move.copy()

        # remove signal from outlier

        if 1:
            new_score, new_df, df_move, outlier_model = propose_sample_removal(
                df_noise, features, test_frac=0.3, isnoise=1
            )
            frac_of_noise = len(df_noise) / (len(df_signal))
            gamma = new_score / noise_average_score[-1]
            # gamma =  gamma  * (frac_of_noise/traget_noise_signal_ratio)
            u = np.random.rand(1)

            if u <= gamma * window:
                # if gamma>=0.90:
                if frac_of_noise > 0.3:
                    # new_score = (1 * noise_average_score[-1] + new_score) / 2
                    new_score = smoother(noise_average_score[-1], new_score, 5)
                    df_signal = pd.concat([df_signal, df_move], axis=0)
                    df_noise = new_df
                    noise_average_score.append(new_score)
        else:
            # if (np.mod(iter, 10) == 0) and (frac_of_noise > 0.30):
            df_noise["err"] = np.power(
                (signal_model.predict(df_noise[features]) - df_noise["y"]), 2
            )
            df_noise.sort_values("err", inplace=True)
            del df_noise["err"]
            df_noise.reset_index(inplace=True)
            del df_noise["index"]

            sig_samples = df_noise.index.values[0:5]
            df_signal = pd.concat(
                [df_signal, df_noise.iloc[sig_samples]], axis=0
            )
            df_signal.reset_index(inplace=True)
            del df_signal["index"]
            df_noise.drop(index=sig_samples, inplace=True)
            df_noise.reset_index(inplace=True)
            del df_noise["index"]
        frac_noise_list.append(frac_of_noise)
        visulize(
            initialize=False,
            fig=fig,
            axs=axs,
            noise_average_score=noise_average_score,
            signal_average_scroe=signal_average_scroe,
            df_signal=df_signal,
            df_noise=df_noise,
            features=features,
            iter=iter,
            frac_noise_list=frac_noise_list,
            df_move=out_df_mov,
        )

    pass


def algo_mcmc2(df):
    score = "neg_mean_absolute_percentage_error"
    fig, axs = visulize(initialize=True)

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df_signal = df.sample(frac=0.5)
    df_noise = df.drop(index=df_signal.index)

    features = [feat for feat in df.columns if "x" in feat]
    gb = xgb_estimator(params=None)

    signal_scores = cross_val_score(
        gb, df_signal[features], df_signal["y"], scoring=score, cv=5
    )

    noise_scores = cross_val_score(
        gb, df_noise[features], df_noise["y"], scoring=score, cv=5
    )

    signal_average_scroe = [np.mean(signal_scores)]
    noise_average_score = [np.mean(noise_scores)]

    N = 250
    frac_noise_list = []
    for iter in range(N):
        out_df_mov = pd.DataFrame(columns=df_signal.columns)
        # remove outliers from  signal
        new_score, new_df, df_move, signal_model = propose_sample_removal(
            df_signal, features, test_frac=0.3, isnoise=0
        )
        frac_of_noise = len(df_noise) / len(df_signal)
        gamma = signal_average_scroe[-1] / new_score
        # gamma = gamma * (traget_noise_signal_ratio/frac_of_noise)
        u = np.random.rand(1)
        # signal_frac = len(df_signal) / (len(df_signal) + len(df_noise))
        window = 1 - 0.75 * iter / N
        if window < 0.1:
            window = 0.1

        signal_frac = len(df_signal) / len(df_noise)
        if u <= gamma:
            # if gamma >= 0.80:
            if signal_frac > 0.5:
                # new_score = (1* signal_average_scroe[-1] + new_score) / 2
                new_score = smoother(signal_average_scroe[-1], new_score, 5)
                df_noise = pd.concat([df_noise, df_move], axis=0)
                df_signal = new_df
                signal_average_scroe.append(new_score)
                out_df_mov = df_move.copy()

        # remove signal from outlier
        df_noise["err"] = np.power(
            (signal_model.predict(df_noise[features]) - df_noise["y"]), 2
        )
        df_noise.sort_values("err", inplace=True)
        del df_noise["err"]
        df_noise.reset_index(inplace=True)
        del df_noise["index"]

        sig_samples = df_noise.index.values[0:10]
        df_move = df_noise.iloc[sig_samples]
        df_signal_ = pd.concat([df_signal, df_move], axis=0)
        df_signal_.reset_index(inplace=True)
        del df_signal_["index"]
        new_score = cross_validate(df_signal_, features, score)

        u = np.random.rand(1)
        window = 1 - 0.4 * iter / N
        if window < 0.1:
            window = 0.1

        gamma = signal_average_scroe[-1] / new_score
        signal_frac = len(df_signal) / len(df_noise)
        if u <= gamma * window:
            # if gamma >= 0.80:
            if signal_frac < 20:
                # new_score = (1* signal_average_scroe[-1] + new_score) / 2
                df_signal = df_signal_.copy()
                signal_average_scroe.append(new_score)
                df_noise.drop(index=sig_samples, inplace=True)

        frac_noise_list.append(frac_of_noise)
        visulize(
            initialize=False,
            fig=fig,
            axs=axs,
            noise_average_score=noise_average_score,
            signal_average_scroe=signal_average_scroe,
            df_signal=df_signal,
            df_noise=df_noise,
            features=features,
            iter=iter,
            frac_noise_list=frac_noise_list,
            df_move=out_df_mov,
        )

    pass


def _get_args(arg, default=None, **kwargs):
    if arg in kwargs["keys"]:
        val = kwargs[arg]
        if val is None:
            val = default
    else:
        val = default
    return val


def algo_mcmc_official(df, **kwargs):
    alpha = _get_args("initial_split_frac", default=0.5, **kwargs)
    target = _get_args("target", default="y", **kwargs)
    features = _get_args("features", default=df.columns, **kwargs)
    if target in features:
        features.remove(target)
    score = _get_args(
        "score", default="neg_mean_absolute_percentage_error", **kwargs
    )
    N = _get_args("max_iterations", default=200, **kwargs)
    kfolds = _get_args("kfolds", default=5, **kwargs)
    test_frac = _get_args("test_frac", default=0.3, **kwargs)
    max_signal_ratio = _get_args("max_signal_ratio", default=30, **kwargs)
    min_signal_ratio = _get_args("min_signal_ratio", default=0.5, **kwargs)
    cooling_rate = _get_args("cooling_rate", default=1.0, **kwargs)
    damping_weight = _get_args("damping_weight", default=5.0, **kwargs)

    kwargs["initial_df"] = df.copy()
    fig, axs = visulize(initialize=True)
    kwargs["fig"] = fig
    kwargs["axs"] = axs

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df_signal = df.sample(frac=alpha)
    df_noise = df.drop(index=df_signal.index)

    gb = xgb_estimator(params=None)
    signal_scores = cross_val_score(
        gb, df_signal[features], df_signal[target], scoring=score, cv=kfolds
    )
    signal_average_score = [np.mean(signal_scores)]

    frac_noise_list = []
    for iter in range(N):
        out_df_mov = pd.DataFrame(columns=df_signal.columns)
        # remove outliers from  signal
        new_score, new_df, df_move, signal_model = propose_sample_removal(
            df_signal, features, test_frac=test_frac, isnoise=0
        )
        frac_of_noise = len(df_noise) / len(df_signal)
        gamma = signal_average_score[-1] / new_score
        u = np.random.rand(1)
        window = 1 - cooling_rate * iter / N
        if window < 0.1:
            window = 0.1

        signal_frac = len(df_signal) / len(df_noise)
        if u <= gamma:
            if signal_frac > min_signal_ratio:
                new_score = smoother(
                    signal_average_score[-1], new_score, damping_weight
                )
                df_noise = pd.concat([df_noise, df_move], axis=0)
                df_signal = new_df
                signal_average_score.append(new_score)
                out_df_mov = df_move.copy()

        # remove signal from outlier
        df_noise["err"] = np.power(
            (signal_model.predict(df_noise[features]) - df_noise["y"]), 2
        )
        df_noise.sort_values("err", inplace=True)
        del df_noise["err"]
        df_noise.reset_index(inplace=True)
        del df_noise["index"]

        sig_samples = df_noise.index.values[0:10]
        df_move = df_noise.iloc[sig_samples]
        df_signal_ = pd.concat([df_signal, df_move], axis=0)
        df_signal_.reset_index(inplace=True)
        del df_signal_["index"]
        new_score = cross_validate(df_signal_, features, score)

        u = np.random.rand(1)
        window = 1 - cooling_rate * iter / N
        if window < 0.1:
            window = 0.1

        gamma = signal_average_score[-1] / new_score
        signal_frac = len(df_signal) / len(df_noise)
        if u <= gamma * window:

            if signal_frac < max_signal_ratio:
                df_signal = df_signal_.copy()
                signal_average_score.append(new_score)
                df_noise.drop(index=sig_samples, inplace=True)

        frac_noise_list.append(frac_of_noise)
        visulize(
            initialize=False,
            fig=fig,
            axs=axs,
            noise_average_score=noise_average_score,
            signal_average_scroe=signal_average_score,
            df_signal=df_signal,
            df_noise=df_noise,
            features=features,
            iter=iter,
            frac_noise_list=frac_noise_list,
            df_move=out_df_mov,
        )

    pass


def simple_train_test(df_):
    objective = mse()
    # objective = "reg:squarederror"
    params = {
        "objective": objective,  # "reg:squarederror"
        "tree_method": "hist",
        "colsample_bytree": 0.8,
        "learning_rate": 0.20,
        "max_depth": 7,
        "alpha": 100,
        "n_estimators": 500,
        "rate_drop": 0.9,
        "skip_drop": 0.5,
        "subsample": 0.8,
        "reg_lambda": 10,
        "min_child_weight": 100,
        "gamma": 10,
        "max_delta_step": 0,
        "seed": 123,
    }

    df = df_.copy()
    test_frac = 0.3
    gb = xgb.XGBRegressor(**params)
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df.sample(frac=(1 - test_frac))
    test_df = df.drop(index=train_df.index)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # fit a model to flag possible samples to removing
    gb.fit(train_df[["x1"]], train_df["y"])
    plt.scatter(gb.predict(test_df[["x1"]]), test_df["y"])
    err = np.power((gb.predict(test_df["x1"]) - test_df["y"]), 2.0)

    return gb


if __name__ == "__main__":

    if 1:  # 1D example

        # generate model
        df1 = pd.DataFrame(np.arange(-100, 100, 0.4), columns=["x1"])
        df1["y"] = (
            -3 * df1["x1"]
            + np.power(df1["x1"], 2)
            + np.power(df1["x1"] / 3, 3.0)
        )
        # x = df1['x1']
        # y = np.power(x,5.0) - 3 * np.power(x, 4.0) + 3 * np.power(x, 3) - 2 * np.power(x,2.0) - 5
        # df1['y'] = y

        if 0:
            simple_train_test(df1)

        df = df1
        add_normal_noise_to_col(df, "y", mu=0, seg=1500)

        # add noise
        df = add_outlier_samples(df, frac=1.5)
        algo_mcmc2(df)

        x = 1

    pass
