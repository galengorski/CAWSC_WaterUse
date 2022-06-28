# =============================
# Quantile regression
# =============================
# log cosh quantile is a regularized quantile loss function
def log_cosh_quantile(alpha):
    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = 0.99 * np.tanh(err)

        hess = 0.001 + 1 / np.cosh(err) ** 2

        #hess = np.ones_like(hess)
        return grad, hess
    return _log_cosh_quantile

def oo(alpha, delta):
    def original_quantile_loss(y_true,y_pred):
        x = y_true - y_pred
        grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
        hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta
        return grad,hess
    return original_quantile_loss

def quantile_loss( alpha, delta, threshold, var):
    def _quantile_loss(y_true,y_pred):
        x = y_true - y_pred
        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
                    (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta

        grad = (np.abs(x) < threshold) * grad - (np.abs(x) >= threshold) * (
                    2 * np.random.randint(2, size=len(y_true)) - 1.0) * var
        hess = (np.abs(x) < threshold) * hess + (np.abs(x) >= threshold)
        return grad, hess
    return _quantile_loss

quantile_alphas = [0.5, 0.90]

params_qauntiles = params2.copy()
del(params_qauntiles['objective'])
params_qauntiles['reg_lambda'] = 10

xgb_quantile_alphas = {}
for quantile_alpha in quantile_alphas:
    # to train a quantile regression, we change the objective parameter and
    # specify the quantile value we're interested in
    #params_qauntiles['objective'] = log_cosh_quantile(quantile_alpha),
    gb_ = xgb.XGBRegressor(objective=oo(quantile_alpha, 10), **params_qauntiles )
    gb_.fit(X_train, y_train)
    xgb_quantile_alphas[quantile_alpha] = gb_
    print("Done Quantile {}".format(quantile_alpha))
plt.figure()
for quantile_alpha, lgb in xgb_quantile_alphas.items():
    ypredict = lgb.predict(X_test)
    plt.scatter(y_test, ypredict, s = 4, label = "{}".format
                (quantile_alpha))
# plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')
plt.legend()
lim = [min(y_test), max(y_test)]
plt.plot(lim, lim, 'k')
plt.xlabel("Actual Water Use")
plt.ylabel("Estimated Water Use")
