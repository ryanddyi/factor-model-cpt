import numpy as np
import scipy
import scipy.linalg
import scipy.stats
import math
import sklearn
from sklearn import linear_model
import pandas as pd
import random
import cpt_functions as cpt
import argparse
import itertools
import os
from arch import arch_model


# read sp 100 data 
ret_mat_100 = pd.read_csv('sp100retMat.csv', index_col=0)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
args = parser.parse_args()

job_id = args.integers[0]

def GARCH_normalize(ret,winsorize):

    #ret = retMat.iloc[:,10]
    ret = ret-ret.mean()
    garch11 = arch_model(ret, p=1, q=1)
    res = garch11.fit(update_freq=5)
    ret_std = ret/res.conditional_volatility

    # winsorize

    if winsorize:
        q99 = ret_std.quantile(0.99)
        q01 = ret_std.quantile(0.01)
        ret_std[ret_std>q99] = q99
        ret_std[ret_std<q01] = q01

    return ret_std

def GARCH_vol(ret):
    #ret = retMat.iloc[:,10]
    ret = ret-ret.mean()
    garch11 = arch_model(ret, p=1, q=1)
    res = garch11.fit(update_freq=5)
    vol = res.conditional_volatility
    return vol[len(vol)-1]


# read sp 100 data 
ret_mat_100 = pd.read_csv('sp100retMat.csv', index_col=0)


random.seed(1000+job_id)
ret_mat = ret_mat_100.iloc[0:job_id]

y_mat = np.ndarray(ret_mat.shape)
for j in range(ret_mat.shape[1]):
    y_mat[:,j] = GARCH_normalize(ret_mat.iloc[:,j],winsorize=True)

garch_vol = np.ndarray(ret_mat_100.shape[1])

for j in range(ret_mat.shape[1]):
    garch_vol[j] = GARCH_vol(ret_mat.iloc[:,j])

np.savetxt("garch/vol"+str(job_id)+".csv", garch_vol, delimiter=",")
np.savetxt("garch/cov"+str(job_id)+".csv", np.cov(y_mat,rowvar=False), delimiter=",")

