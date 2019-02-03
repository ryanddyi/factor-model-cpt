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

def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

params = expand_grid({'nu':[2,5,10,20],'s2':[1/4,1/2,1,2]})

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
args = parser.parse_args()
job_id = args.integers[0]
random.seed(1000+job_id)

batch = int(job_id/50)
directory = 'batch' + str(batch)
if not os.path.exists(directory):
    os.makedirs(directory)

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

# log-likelihood
def log_likelihood(subsetting):
    global Beta, Lambda2, sigma2, k_plus, tau, lambda2_0
    if subsetting:
        loglik = -0.5*(len(tau)-1)*k_plus*np.log(len(cpt_set)-1)
    else:
        loglik = -0.5*(len(tau)-1)*k_plus*np.log(n_date)

    if k_max>k_plus:
        loglik = loglik - (eta*s2_lambda/lambda2_0 + (eta+2)*np.log(lambda2_0))/2

    loglik = loglik - sum(xi*s2_sigma/sigma2 + (xi+2)*np.log(sigma2))/2

    Sigma_t = [];
    for j in range(len(tau)-1):
        loglik = loglik - sum(eta*s2_lambda/Lambda2[j][0:k_plus] + (eta+2)*np.log(Lambda2[j][0:k_plus]))/2
        Sigma_t.append(Beta.dot(np.diag(Lambda2[j])).dot(Beta.transpose()) + np.diag(sigma2))
        for t in range(tau[j],tau[j+1]):
            loglik = loglik + scipy.stats.multivariate_normal.logpdf(y_mat[t,:],np.zeros(n_name), Sigma_t[j])

    loglik = loglik + (np.log(theta1*delta0*np.exp(-delta0*abs(Beta[:,0:k_plus])) + 
                              (1-theta1)*delta1*np.exp(-delta1*abs(Beta[:,0:k_plus])))).sum()
    if k_max>k_plus:
        loglik = loglik + (np.log(theta0*delta0*np.exp(-delta0*abs(Beta[:,k_plus:k_max])) + 
                              (1-theta0)*delta1*np.exp(-delta1*abs(Beta[:,k_plus:k_max])))).sum()
    return loglik

# simulation
n_factor = 5
block_size = 30
overlap_size = 5
n_name = overlap_size+(block_size-overlap_size)*n_factor

# B_0
B_mat =  np.zeros((n_name,n_factor))
for k in range(n_factor):
    B_mat[(block_size-overlap_size)*k:block_size*(k+1)-overlap_size*k,k] = 1

# static covariance matrix
#covmat_true = B_mat.dot(B_mat.transpose())+np.diag(np.ones(n_name))*sd_idio**2

# segments
n_segment = 4
tau_true = [0, 80, 160, 200, 300]
n_date = 300

nu = params.nu[batch]
s2 = params.s2[batch]
sd_idio = np.sqrt(s2*np.random.uniform(0.5,1.5,n_name))
# generate factors with change-points
f_mat = np.ndarray((n_date, n_factor))
for k in range(n_factor):
    for j in range(n_segment):
        lambda_0 = np.sqrt(np.exp(np.sqrt(np.log(nu)/2)*np.random.randn(1)))
        f_mat[tau_true[j]:tau_true[j+1],k] = np.random.randn(tau_true[j+1]-tau_true[j])*lambda_0

y_mat = f_mat.dot(B_mat.transpose()) + np.random.randn(n_date,n_name)*sd_idio

y_mat = (y_mat-y_mat.mean(axis=0))/y_mat.std(axis=0)

## PXL-EM
## parameters and initialization

n_date=y_mat.shape[0]
n_name=y_mat.shape[1]

k_max = 10
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

subsetting = False
log_like = [];
delta0_steps = [1,5,10]
for delta0 in delta0_steps:
    for i in range(200):
        Beta_old = Beta
        EM_iterate(True, subsetting)
        if i>10 and abs(Beta-Beta_old).max()<0.001:
            break

    for i in range(200):
        Beta_old = Beta
        EM_iterate(False, subsetting)
        if i>10 and abs(Beta-Beta_old).max()<0.001:
            break

delta0=20
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


np.savetxt(directory+"/tau"+str(job_id)+"k"+str(k_plus)+".csv", tau, delimiter=",")
#np.savetxt("Lambda1.csv", Lambda2, delimiter=",")
#np.savetxt("Beta1.csv", Beta, delimiter=",")
#np.savetxt("loglik1.csv", log_like, delimiter=",")



