# EM algorithm for Bayesian factor model with multiple changepoints

import numpy as np
import scipy.linalg
import scipy.stats
import math
import sklearn
from sklearn import linear_model
import cpt_functions as cpt
import copy

# idea on refactoring this piece of code
# a class centered around y_mat
# initialize the set of parameters in constructor
# config hyper-parameters

class FactorModel():
    """
    A class centered around multivariate time series y_mat (n_time x n_name)
    """
    def __init__(self, y_mat, k_max, delta=[5,0.001], theta=[0.001,0.5]):
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


    def cpt_config(self, subsetting=False, minseglen=5, cpt_set=None):
        # specify change-point candidate set
        self.subsetting = subsetting
        if subsetting:
            self.cpt_set = cpt_set
        else:
            self.minseglen = minseglen


    def param_init(self):
        # parameters: Beta, Lambda2, k_plus, tau, lambda2_0
        self.Beta = np.random.randn(self.n_name, self.k_max)
        self.sigma2 = np.ones(self.n_name)

        self.Lambda2 = []
        self.Lambda2.append(np.ones(self.k_max))

        self.tau = [0, self.n_date]
        self.k_plus = self.k_max


    def delta0_reconfig(self, delta0):
        self.delta[0] = delta0


    def _e_step_gamma(self):
        """
        conditional expectation: sufficient statistics of gamma
        """
        Theta_rep = self.theta[0] * np.ones((self.n_name, self.k_max))
        Theta_rep[:,:self.k_plus] = self.theta[1]
        tmp = np.exp(-abs(self.Beta)*(self.delta[0]-self.delta[1]))
        ratio = self.delta[0]/self.delta[1]*(1-Theta_rep)/Theta_rep*tmp
        E_Gamma = 1/(1+ratio)
        return E_Gamma


    def _e_step_f(self):
        """
        conditional expectation: sufficient statistics of F
        """
        M = []
        M_sum = 0
        E_F = np.ndarray(shape = (self.n_date,self.k_max))
        E_F2= np.ndarray(shape = (self.n_date,self.k_max))
        tau = self.tau
        Beta = self.Beta

        Sigma_inv=np.diag(1/self.sigma2)
        for j in range(len(tau)-1):
            M.append(np.linalg.inv(np.diag(1/self.Lambda2[j]) +
                     Beta.transpose().dot(Sigma_inv).dot(Beta)))
            M_sum = M_sum + (tau[j+1] - tau[j])*M[j]

            for t in range(tau[j],tau[j+1]):
                E_F[t,:] = M[j].dot(Beta.transpose()).dot(Sigma_inv).dot(y_mat[t,:])
                E_F2[t,:]=E_F[t,:]**2 + np.diag(M[j])

        M_u = scipy.linalg.sqrtm(M_sum)
        return E_F, E_F2, M_u, M_sum


    def _m_step_q1(self, E_F2, E_Gamma):
        """
        M-step of Q1. Change-point detection
        """
        k_plus = self.k_plus
        s2_lambda = self.s2_lambda
        eta = self.eta
        k_max = self.k_max
        theta = self.theta

        if self.subsetting:
            cpt_m = cpt.cpt_detect_PELT(eta, s2_lambda, E_F2[:,:k_plus], k_plus, self.cpt_set)
        else:
            cpt_m = cpt.cpt_detect_minseglen_PELT(eta, s2_lambda, E_F2[:,:k_plus], k_plus, self.minseglen)

        Q1_list = []
        for k_star in range(1, k_max+1):
            if self.subsetting:
                Q1 = -0.5*(len(cpt_m)-1)*k_star*math.log(len(self.cpt_set)-1)
            else:
                Q1 = -0.5*(len(cpt_m)-1)*k_star*math.log(self.n_date)

            Q1 += (E_Gamma[:,:k_star].sum()*math.log(theta[1]) + 
                    (1-E_Gamma[:,:k_star]).sum()*math.log(1-theta[1]))
            if k_star < k_max:
                Q1 += (E_Gamma[:,k_star:k_max].sum()*math.log(theta[0]) + 
                        (1-E_Gamma[:,k_star:k_max]).sum()*math.log(1-theta[0]))
                Q1 -= cpt.cost_function_inactive_factors(eta, s2_lambda, E_F2[:,k_star:k_max])

            for j in range(len(cpt_m)-1):
                Q1 -= cpt.cost_function(eta, s2_lambda, E_F2[cpt_m[j]:cpt_m[j+1],:], k_star)

            Q1_list.append(Q1)

        k_plus = np.argmax(Q1_list)+1

        lambda2_m = []

        for j in range(len(cpt_m)-1):
            lambda2_m.append((eta*s2_lambda+E_F2[cpt_m[j]:cpt_m[j+1],:k_plus].sum(axis=0))/(eta+2+cpt_m[j+1]-cpt_m[j]))

        lambda2_0 = (eta*s2_lambda+E_F2[:,k_plus:k_max].sum())/(eta+2+E_F2[:,k_plus:k_max].size)

        Lambda2 = []
        for j in range(len(cpt_m)-1):
            Lambda2.append(np.append(lambda2_m[j],lambda2_0*np.ones(k_max-k_plus)))

        self.tau = cpt_m
        self.k_plus = k_plus
        self.Lambda2 = Lambda2
        print(self.tau)


    def _m_step_q2(self, E_F, M_u, M_sum, E_Gamma, PXL=False):
        """
        maximize Q2(B,Sigma)
        """
        Beta = self.Beta
        n_date = self.n_date
        n_name = self.n_name
        k_max = self.k_max
        xi = self.xi
        delta = self.delta

        tilde_Y = np.append(self.y_mat, np.zeros((k_max,n_name)), axis=0)
        tilde_F = np.append(E_F, M_u, axis=0)
        tilde_F_rw = np.ndarray(tilde_F.shape)

        for j in range(n_name):
            # penalty term
            lambda_j = (1-E_Gamma[j,:])*delta[0]+E_Gamma[j,:]*delta[1]

            # reweight tilde_F
            for k in range(k_max):
                tilde_F_rw[:,k] = tilde_F[:,k]/lambda_j[k]

            # lasso
            clf = linear_model.Lasso(alpha = self.sigma2[j]/(n_date+k_max), normalize = False, fit_intercept = False)
            clf.fit(tilde_F_rw, tilde_Y[:,j])
            Beta[j,:] = clf.coef_/lambda_j

            # sum of square of residuls
            SSR = sum((tilde_Y[:,j]-tilde_F.dot(Beta[j,:]))**2)

            # update 
            self.sigma2[j]=(SSR+xi*self.s2_sigma)/(n_date+xi+2)

        if PXL:
            # lower triangular
            A_l = scipy.linalg.cholesky(1/n_date*(E_F.transpose().dot(E_F)+M_sum), lower=True)
            Beta = Beta.dot(A_l)

        order = np.argsort(abs(Beta).sum(axis=0))[::-1]
        Beta = Beta[:,order]

        for j in range(len(self.tau)-1):
            self.Lambda2[j] = self.Lambda2[j][order]


    def em_iterator(self, nstep, PXL):
        """
        Main solver of the EM algorithm.
        PXL indicates whether we use PXL-EM.
        """
        for i in range(nstep):
            Beta_old = copy.deepcopy(self.Beta)
            E_Gamma = self._e_step_gamma()
            E_F, E_F2, M_u, M_sum = self._e_step_f()
            self._m_step_q1(E_F2, E_Gamma)
            self._m_step_q2(E_F, M_u, M_sum, E_Gamma, PXL)
            #print((self.Beta-Beta_old).max())
            if i > nstep/5 and (self.Beta-Beta_old).max()<0.0001:
                print(Q1_list)
                print(k_plus)
                break

    def log_likelihood(self):
        """ Evaluate likelihood"""
        Beta = self.Beta
        Lambda2 = self.Lambda2
        sigma2 = self.sigma2
        k_max = self.k_max
        k_plus = self.k_plus
        tau = self.tau
        lambda2_0 = self.lambda2_0

        if self.subsetting:
            loglik = -0.5*(len(tau)-1)*k_plus*np.log(len(self.cpt_set)-1)
        else:
            loglik = -0.5*(len(tau)-1)*k_plus*np.log(self.n_date)

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
    sd_idio = 0.1

    # B_0
    B_mat =  np.zeros((n_name,n_factor))
    for k in range(n_factor):
        B_mat[(block_size-overlap_size)*k:block_size*(k+1)-overlap_size*k,k] = 1 + np.random.randn(block_size)/10

    # segments 
    n_segment = 4
    tau_true = [0, 50, 80, 100, 150]
    n_date = 150
    lambda_0 = [10,1,10,1]

    # generate factors with change-points
    f_mat = np.ndarray((n_date, n_factor))
    for k in range(n_factor):
        for j in range(n_segment):
            f_mat[tau_true[j]:tau_true[j+1],k] = np.random.randn(tau_true[j+1]-tau_true[j])*np.sqrt(lambda_0[j])

    y_mat = f_mat.dot(B_mat.transpose()) + np.random.randn(n_date,n_name)*sd_idio
    print(B_mat)
    f2_mat = f_mat**2
    print(cpt.cpt_detect_minseglen_PELT(1, 1, f2_mat,n_factor, 5))

    return y_mat


## PXL-EM
## parameters and initialization

def normalize(y_mat):
    return  (y_mat-y_mat.mean(axis=0))/y_mat.std(axis=0)


if __name__ == "__main__":
    y_mat = data_toy_simulation()
    y_mat = normalize(y_mat)

    model = FactorModel(y_mat, k_max=10)

    model.cpt_config()
    model.param_init()
    model.em_iterator(200, True)
    print(model.k_plus)
    print(model.Beta[:,:model.k_plus])
    model.em_iterator(200, False)
    print(model.k_plus)
    print(model.Beta[:,:model.k_plus])
    model.delta0_reconfig(10)
    model.em_iterator(200, True)
    print(model.k_plus)
    print(model.Beta[:,:model.k_plus])
    model.em_iterator(200, False)
    print(model.k_plus)
    print(model.Beta[:,:model.k_plus])
    model.delta0_reconfig(20)
    model.em_iterator(200, True)
    print(model.k_plus)
    print(model.Beta[:,:model.k_plus])
    model.em_iterator(200, False)
    print(model.k_plus)
    print(model.Beta[:,:model.k_plus])
    model.delta0_reconfig(50)
    model.em_iterator(200, True)
    print(model.k_plus)
    print(model.Beta[:,:model.k_plus])
    model.em_iterator(200, False)
    print(model.k_plus)
    print(model.Beta[:,:model.k_plus])



