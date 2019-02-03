import numpy as np
from arch import arch_model


def normalize(y_mat):
    return  (y_mat-y_mat.mean(axis=0))/y_mat.std(axis=0)


def GARCH_normalize(ret, winsorize=True):

    ret = ret-ret.mean()
    garch11 = arch_model(ret, p=1, q=1)
    res = garch11.fit(update_freq=5)
    ret_std = ret/res.conditional_volatility

    # winsorize

    if winsorize:
        q99 = np.quantile(ret_std, 0.99)
        q01 = np.quantile(ret_std, 0.01)
        ret_std[ret_std>q99] = q99
        ret_std[ret_std<q01] = q01

    return ret_std

