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

def EM_iterate(PXL, subsetting):
    global Beta, Lambda2, sigma2, k_plus, tau, E_Gamma, Q_list, lambda2_0
    ## conditional expectation gamma
    Theta_rep = np.repeat(Theta,n_name, axis=1).transpose()
    E_Gamma = 1/(1+delta0/delta1*(1-Theta_rep)/Theta_rep*np.exp(-abs(Beta)*(delta0-delta1)))

    ## conditional expectation F
    M=[];
    M_sum=0

    Sigma_inv=np.diag(1/sigma2)
    for j in range(len(tau)-1):
        M.append(np.linalg.inv(np.diag(1/Lambda2[j]) + Beta.transpose().dot(Sigma_inv).dot(Beta)))
        M_sum = M_sum+(tau[j+1]-tau[j])*M[j]

        for t in range(tau[j],tau[j+1]):
            E_F[t,:] = M[j].dot(Beta.transpose()).dot(Sigma_inv).dot(y_mat[t,:])
            E_F2[t,:]=E_F[t,:]**2 + np.diag(M[j])

    M_u = scipy.linalg.sqrtm(M_sum)

    # change-point detection
    if subsetting:
        cpt_m = cpt.cpt_detect_PELT(eta, s2_lambda, E_F2[:,range(k_plus)], k_plus, cpt_set)
    else:
        cpt_m = cpt.cpt_detect_minseglen_PELT(eta, s2_lambda, E_F2[:,range(k_plus)], k_plus, minseglen)

    Q1_list = [];
    for k_star in range(1, k_max+1):
        if subsetting:
            Q1 = -0.5*(len(cpt_m)-1)*k_star*math.log(len(cpt_set)-1)
        else:
            Q1 = -0.5*(len(cpt_m)-1)*k_star*math.log(n_date)
        
        Q1 = Q1 + (E_Gamma[:,range(k_star)].sum()*math.log(theta1) + 
                   (1-E_Gamma[:,range(k_star)]).sum()*math.log(1-theta1))
        if k_star < k_max:
            Q1 = Q1 + (E_Gamma[:,range(k_star,k_max)].sum()*math.log(theta0) + 
                       (1-E_Gamma[:,range(k_star,k_max)]).sum()*math.log(1-theta0))
            Q1 = Q1 - cpt.cost_function_inactive_factors(eta, s2_lambda, E_F2[:,range(k_star,k_max)])

        for j in range(len(cpt_m)-1):
            Q1 = Q1 - cpt.cost_function(eta, s2_lambda, E_F2[cpt_m[j]:cpt_m[j+1],:], k_star)
        Q1_list.append(Q1)

    k_plus = np.argmax(Q1_list)+1

    lambda2_m = [];
    l2_m = np.zeros(k_plus)
    for j in range(len(cpt_m)-1):
        lambda2_m.append((eta*s2_lambda+E_F2[cpt_m[j]:cpt_m[j+1],range(k_plus)].sum(axis=0))/(eta+2+cpt_m[j+1]-cpt_m[j]))

    lambda2_0 = (eta*s2_lambda+E_F2[:,range(k_star,k_max)].sum())/(eta+2+E_F2[:,range(k_star,k_max)].size)

    Lambda2 = [];
    for j in range(len(cpt_m)-1):
        Lambda2.append(np.append(lambda2_m[j],lambda2_0*np.ones(k_max-k_plus)))

    tau = cpt_m

    ## maximize Q2(B,Sigma)

    tilde_Y = np.append(y_mat, np.zeros((k_max,n_name)), axis=0)
    tilde_F = np.append(E_F,M_u, axis=0)  
    tilde_F_rw = np.ndarray(tilde_F.shape)

    for j in range(n_name):
        # penalty term
        lambda_j = (1-E_Gamma[j,:])*delta0+E_Gamma[j,:]*delta1

        # reweight tilde_F
        for k in range(k_max):
            tilde_F_rw[:,k] = tilde_F[:,k]/lambda_j[k]

        # lasso
        clf = linear_model.Lasso(alpha = sigma2[j]/(n_date+k_max), normalize = False, fit_intercept = False)
        clf.fit(tilde_F_rw, tilde_Y[:,j])
        Beta[j,:]=clf.coef_/lambda_j

        # sum of square of residuls
        SSR = sum((tilde_Y[:,j]-tilde_F.dot(Beta[j,:]))**2)

        # update 
        sigma2[j]=(SSR+xi*s2_sigma)/(n_date+xi+2)
    
    if PXL:
        # lower triangular 
        A_l = scipy.linalg.cholesky(1/n_date*(E_F.transpose().dot(E_F)+M_sum), lower=True)
        Beta = Beta.dot(A_l)
    
    order = np.argsort(abs(Beta).sum(axis=0))[::-1]
    Beta = Beta[:,order]
    for j in range(len(cpt_m)-1):
        Lambda2[j] = Lambda2[j][order]    
    return

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

# read sp 100 data 
ret_mat_100 = pd.read_csv('sp100retMat.csv', index_col=0)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
args = parser.parse_args()

job_id = args.integers[0]
random.seed(1000+job_id)
ret_mat = ret_mat_100.iloc[0:job_id]

y_mat = np.ndarray(ret_mat.shape)
for j in range(ret_mat.shape[1]):
    y_mat[:,j] = GARCH_normalize(ret_mat.iloc[:,j],winsorize=True)
    
## PXL-EM
## parameters and initialization

y_mat = (y_mat-y_mat.mean(axis=0))/y_mat.std(axis=0)

n_date=y_mat.shape[0]
n_name=y_mat.shape[1]

k_max = 20
k_plus = 1

Beta = np.random.randn(n_name,k_max)*10
sigma2 = np.ones(n_name) # diagonal of Sigma

E_F = np.ndarray(shape = (n_date,k_max))
E_F2= np.ndarray(shape = (n_date,k_max))

tau = [0,n_date];
Lambda2=[];
Lambda2.append(np.ones(k_max));

theta1 = 0.5
theta0 = 0.001
Theta = theta1*np.ones((k_max,1))

delta1=0.001
s2_lambda=1
s2_sigma=1

eta=1
xi=1
minseglen=5

cpt_set = minseglen*np.array(range(int((n_date-1)/minseglen)+1))
cpt_set = np.append(cpt_set,n_date)

subsetting = True
log_like = [];
delta0_steps = [1,2,5,10,20]
for delta0 in delta0_steps:
    for i in range(200):
        Beta_old = Beta
        EM_iterate(True, subsetting)
        if (i+1)%50 == 0:
            print(k_plus)
            print(tau)
        if i>10 and abs(Beta-Beta_old).max()<0.001:
            break

    for i in range(200):
        Beta_old = Beta
        EM_iterate(False, subsetting)
        if (i+1)%50 == 0:
            print(k_plus)
            print(tau)
        if i>10 and abs(Beta-Beta_old).max()<0.001:
            break

delta0=50
for i in range(200):
    Beta_old = Beta
    EM_iterate(True, subsetting)
    if i>10 and abs(Beta-Beta_old).max()<0.001:
        break

for i in range(1000):
    Beta_old = Beta
    EM_iterate(False, subsetting)
    if i>10 and abs(Beta-Beta_old).max()<0.001:
        break
        
# rescale Beta and Lambda2
Lambda_ts = np.tile(Lambda2[0],(tau[1]-tau[0],1))
if len(tau)>2:
    for j in range(1,len(tau)-1):
        Lambda_tile = np.tile(Lambda2[j],(tau[j+1]-tau[j],1))
        Lambda_ts = np.concatenate((Lambda_ts,Lambda_tile),axis=0)

lambda2_mean = Lambda_ts.mean(axis=0)
Lambda_ts = Lambda_ts/lambda2_mean
Beta = Beta*np.sqrt(lambda2_mean)

np.savetxt("sp100/Beta"+str(job_id)+".csv", Beta, delimiter=",")
np.savetxt("sp100/Lambda_ts"+str(job_id)+".csv", Lambda_ts, delimiter=",")
np.savetxt("sp100/tau"+str(job_id)+".csv", tau, delimiter=",")
np.savetxt("sp100/sigma2"+str(job_id)+".csv", sigma2, delimiter=",")
