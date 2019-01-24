# EM algorithm for Bayesian factor model with multiple changepoints

import numpy as np
import scipy
import scipy.linalg
import scipy.stats
import math
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
import pandas as pd
import random
import cpt_functions as cpt

# to-do
# idea on refactoring this piece of code
# a class centered around y_mat
# initialize the set of parameters in constructor
# config hyper parameters





class FactorModel():
    """
    A class centered around multivariated time series y_mat (n_time x n_name)
    """
    def __init__(self, y_mat, k_max, delta=[0.001,5], theta=[0.001,0.5]):
        # observed data: y_mat is 2d-array like
        self.y_mat = y_mat
        self.n_date = y_mat.shape[0]
        self.n_name = y_mat.shape[1]

        # hyper-param: k_max, theta, delta
        self.k_max = k_max
        self.theta = theta
        self.delta = delta

        # non-adjustable hyper-params: s2_lambda, s2_sigma, eta, xi
        self.s2_lambda = 1
        self.s2_sigma = 1
        self.eta = 1
        self.xi = 1


    def param_init(self):
        # parameters: Beta, Lambda2, k_plus, tau, lambda2_0
        self.Beta = np.random.randn(self.n_name, self.k_max)
        self.sigma2 = np.ones(self.n_name)

        self.Lambda2 = []
        self.Lambda2.append(np.ones(self.k_max))

        self.tau = [0, n_date]
        self.k_plus = 1
        self.Theta = 


    def _e_step_gamma(self):
        """
        conditional expectation gamma
        """
        Theta_rep = self.theta[0] * np.ones((self.k_max, self.n_name))
        Theta_rep[:k_plus] = self.theta[1]
        ratio = self.delta[0]/self.delta[1]*(1-Theta_rep)/Theta_rep*np.exp(-abs(self.Beta)*(self.delta[0]-self.delta[1]))
        E_Gamma = 1/(1+ratio)
        return E_Gamma


    def _e_step_f(self):
        """
        conditional expectation F
        """
        M = []
        M_sum = 0

        Sigma_inv=np.diag(1/self.sigma2)
        for j in range(len(tau)-1):
            M.append(np.linalg.inv(np.diag(1/self.Lambda2[j]) +
                     self.Beta.transpose().dot(Sigma_inv).dot(self.Beta)))
            M_sum = M_sum + (self.tau[j+1] - self.tau[j])*M[j]

            for t in range(tau[j],tau[j+1]):
                E_F[t,:] = M[j].dot(Beta.transpose()).dot(Sigma_inv).dot(y_mat[t,:])
                E_F2[t,:]=E_F[t,:]**2 + np.diag(M[j])

        M_u = scipy.linalg.sqrtm(M_sum)
        return E_F, E_F2, M_u


    def EM_iterate(PXL, subsetting):
        """
        Main solver of the EM algorithm.
        PXL indicates whether we use PXL-EM.
        """
        # references of all parameters
        Beta = self.Beta
        sigma2 = self.sigma2
        Lambda2 = self.Lambda2
        tau = self.tau
        k_plus = self.k_plus

        # change-point detection
        if subsetting:
            cpt_m = cpt.cpt_detect_PELT(eta, s2_lambda, E_F2[:,range(k_plus)], k_plus, cpt_set)
        else:
            cpt_m = cpt.cpt_detect_minseglen_PELT(eta, s2_lambda, E_F2[:,range(k_plus)], k_plus, minseglen)

        Q1_list = []
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

        lambda2_m = []
        l2_m = np.zeros(k_plus)
        for j in range(len(cpt_m)-1):
            lambda2_m.append((eta*s2_lambda+E_F2[cpt_m[j]:cpt_m[j+1],range(k_plus)].sum(axis=0))/(eta+2+cpt_m[j+1]-cpt_m[j]))

        lambda2_0 = (eta*s2_lambda+E_F2[:,range(k_star,k_max)].sum())/(eta+2+E_F2[:,range(k_star,k_max)].size)

        Lambda2 = []
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

# log-likelihood
def log_likelihood(subsetting):
    global Beta, Lambda2, sigma2, k_plus, tau, lambda2_0
    if subsetting:
        loglik = -0.5*(len(tau)-1)*k_plus*np.log(len(cpt_set)-1)
    else:
        loglik = -0.5*(len(tau)-1)*k_plus*np.log(n_date)

    if k_max > k_plus:
        loglik = loglik - (eta*s2_lambda/lambda2_0 + (eta+2)*np.log(lambda2_0))/2

    loglik = loglik - sum(xi*s2_sigma/sigma2 + (xi+2)*np.log(sigma2))/2

    Sigma_t = []
    for j in range(len(tau)-1):
        loglik = loglik - sum(eta*s2_lambda/Lambda2[j][0:k_plus] + (eta+2)*np.log(Lambda2[j][0:k_plus]))/2
        Sigma_t.append(Beta.dot(np.diag(Lambda2[j])).dot(Beta.transpose()) + np.diag(sigma2))
        for t in range(tau[j],tau[j+1]):
            loglik = loglik + scipy.stats.multivariate_normal.logpdf(y_mat[t,:],np.zeros(n_name), Sigma_t[j])

    loglik = loglik + (np.log(theta1*delta0*np.exp(-delta0*abs(Beta[:,0:k_plus])) + 
                              (1-theta1)*delta1*np.exp(-delta1*abs(Beta[:,0:k_plus])))).sum()
    if k_max > k_plus:
        loglik = loglik + (np.log(theta0*delta0*np.exp(-delta0*abs(Beta[:,k_plus:k_max])) + 
                              (1-theta0)*delta1*np.exp(-delta1*abs(Beta[:,k_plus:k_max])))).sum()
    return loglik


def data_toy_simulation():
    """
    Toy simulation for testing
    """
    # simulation
    n_factor = 5
    block_size = 30
    overlap_size = 5
    n_name = overlap_size+(block_size-overlap_size)*n_factor
    sd_idio = 1

    # B_0
    B_mat =  np.zeros((n_name,n_factor))
    for k in range(n_factor):
        B_mat[(block_size-overlap_size)*k:block_size*(k+1)-overlap_size*k,k] = 1 + np.random.randn(block_size)/10

    # segments 
    n_segment = 4
    tau_true = [0, 50, 80, 100, 150]
    n_date = 150
    lambda_0 = [4,1,3,1]

    # generate factors with change-points
    f_mat = np.ndarray((n_date, n_factor))
    for k in range(n_factor):
        for j in range(n_segment):
            f_mat[tau_true[j]:tau_true[j+1],k] = np.random.randn(tau_true[j+1]-tau_true[j])*np.sqrt(lambda_0[j])

    y_mat = f_mat.dot(B_mat.transpose()) + np.random.randn(n_date,n_name)*sd_idio

    return y_mat


np.savetxt("Beta_true.csv", B_mat, delimiter=",")


plt.plot(f_mat)


f2_mat = f_mat**2
eta=1
s2=1
k_plus=n_factor
cpt_set = range(n_date+1)
cpt.cpt_detect_minseglen_PELT(eta, s2, f2_mat, k_plus, 10)



## PXL-EM
## parameters and initialization

def normalize(y_mat):
    return  (y_mat-y_mat.mean(axis=0))/y_mat.std(axis=0)

y_mat = normalize(y_mat)
n_date=y_mat.shape[0]
n_name=y_mat.shape[1]

k_max = 10
k_plus = 1

Beta = np.random.randn(n_name,k_max)*10
sigma2 = np.ones(n_name) # diagonal of Sigma

E_F = np.ndarray(shape = (n_date,k_max))
E_F2= np.ndarray(shape = (n_date,k_max))

tau = [0,n_date]
Lambda2=[]
Lambda2.append(np.ones(k_max))

theta1 = 0.5
theta0 = 0.001
Theta = theta1*np.ones((k_max,1))

delta1=0.001
delta0=5
s2_lambda=1
s2_sigma=1

eta=1
xi=1
minseglen=5

subsetting = False
log_like = [];
for i in range(200):
    Beta_old = Beta
    EM_iterate(True, subsetting)
    if (i+1)%50 == 0:
        log_like.append(log_likelihood(subsetting))
    if i>50 and abs(Beta-Beta_old).max()<0.001:
        break

for i in range(1000):
    Beta_old = Beta
    EM_iterate(False, subsetting)
    if (i+1)%10 == 0:
        log_like.append(log_likelihood(subsetting))
    if i>200 and abs(Beta-Beta_old).max()<0.001:
        break

delta0 = 10
for i in range(200):
    Beta_old = Beta
    EM_iterate(True, subsetting)
    if (i+1)%50 == 0:
        log_like.append(log_likelihood(subsetting))
    if i>50 and abs(Beta-Beta_old).max()<0.001:
        break

for i in range(1000):
    Beta_old = Beta
    EM_iterate(False, subsetting)
    if (i+1)%10 == 0:
        log_like.append(log_likelihood(subsetting))
    if i>200 and abs(Beta-Beta_old).max()<0.001:
        break

delta0 = 20
for i in range(200):
    Beta_old = Beta
    EM_iterate(True, subsetting)
    if (i+1)%50 == 0:
        log_like.append(log_likelihood(subsetting))
    if i>50 and abs(Beta-Beta_old).max()<0.001:
        break

for i in range(1000):
    Beta_old = Beta
    EM_iterate(False, subsetting)
    if (i+1)%10 == 0:
        log_like.append(log_likelihood(subsetting))
    if i>200 and abs(Beta-Beta_old).max()<0.001:
        break

delta0 = 50
for i in range(200):
    Beta_old = Beta
    EM_iterate(True, subsetting)
    if (i+1)%50 == 0:
        log_like.append(log_likelihood(subsetting))
    if i>50 and abs(Beta-Beta_old).max()<0.001:
        break

for i in range(1000):
    Beta_old = Beta
    EM_iterate(False, subsetting)
    if (i+1)%10 == 0:
        log_like.append(log_likelihood(subsetting))
    if i>200 and abs(Beta-Beta_old).max()<0.001:
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



plt.plot(Lambda_ts)



print(k_plus)
print(tau)
print(Lambda2)
plt.plot(log_like)



plt.plot(abs(Beta))
np.savetxt("/Users/ydd/Documents/covariance/output/simulation/Beta_delta"+str(delta0)+".csv", Beta, delimiter=",")
np.savetxt("/Users/ydd/Documents/covariance/output/simulation/Lambda2_delta"+str(delta0)+".csv", Lambda_ts, delimiter=",")


from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(y_mat)
np.savetxt("Beta_pca.csv", pca.components_.transpose(), delimiter=",")


from sklearn.decomposition import SparsePCA
pca = SparsePCA(n_components=10)
pca.fit(y_mat)
np.savetxt("Beta_spca.csv", pca.components_.transpose(), delimiter=",")


