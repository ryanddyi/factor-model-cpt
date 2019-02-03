import numpy as np
import scipy
import scipy.linalg
import scipy.stats
import math
import sklearn
from sklearn import linear_model
import pandas as pd

# max log-likelihood cost function with scaled inverse chi-square prior
# for each k_plus & t, f2[t,:] follow multivariate normal distribution

def cost_function(eta, s2, f2_mat, k_plus):
    n = f2_mat.shape[0]

    cost=0
    for k in range(k_plus):
        a = eta*s2 + f2_mat[:,k].sum()
        b = eta+2+n
        lambda2_hat = a/b
        cost = cost + 0.5*(a/lambda2_hat + b*math.log(lambda2_hat))

    return cost

# max log-likelihood cost function with flat prior

def cost_function_no_prior(f2_mat, k_plus):
    n = f2_mat.shape[0]

    cost=0
    for k in range(k_plus):
        a = f2_mat[:,k].sum()
        b = n
        lambda2_hat = a/b
        cost = cost + 0.5*(a/lambda2_hat + b*math.log(lambda2_hat))

    return cost

# max log-likelihood cost function with scaled inverse chi-square prior
# for each k_plus & t, f2[t,:] follow multivariate normal distribution with common variance

def cost_function_inactive_factors(eta, s2, f2_mat):
    n = f2_mat.shape[0]
    k_plus = f2_mat.shape[1]

    a = eta*s2
    b = eta+2
    for k in range(k_plus):
        a = a + f2_mat[:,k].sum()
        b = b + n
    lambda2_hat = a/b
    cost = 0.5*(a/lambda2_hat + b*math.log(lambda2_hat))

    return cost

# change-point detection functions
# PELT algorithm is applied with a pre-determined candidate cpt_set

def cpt_detect_PELT(eta, s2, f2_mat, k_plus, cpt_set):

    # change-point candidate sets: tau_sets
    n_all = len(cpt_set) - 1

    pen = 0.5*k_plus*math.log(n_all)
    lastchangecpts = np.ndarray(n_all+1, dtype='int');
    lastchangelike = np.ndarray(n_all+1);
    numchangecpts = np.ndarray(n_all+1, dtype='int');
    checklist = np.ndarray(n_all+1, dtype='int');
    tmplike = np.ndarray(n_all+1);

    # initialize F(0) = -penalty, cp(0) = NULL

    lastchangelike[0] = -pen;
    lastchangecpts[0] = 0;
    numchangecpts[0] = 0;

    # F(1) = C(y_0), cp(1) = 0,
    lastchangelike[1] = cost_function(eta, s2, f2_mat[cpt_set[0]:cpt_set[1],:], k_plus);
    lastchangecpts[1] = 0;
    numchangecpts[1] = 1;

    nchecklist=2;
    checklist[0] = 0;
    checklist[1] = 1;

    for tstar in range(2, n_all+1):
        for i in range(nchecklist):
            tmplike[i] = (lastchangelike[checklist[i]]
                          + cost_function(eta, s2, f2_mat[cpt_set[checklist[i]]:cpt_set[tstar],:], k_plus)
                          + pen);

        whichout = tmplike[range(nchecklist)].argmin()
        minout = tmplike[whichout]
        lastchangelike[tstar] = minout
        lastchangecpts[tstar] = checklist[whichout]
        numchangecpts[tstar] = numchangecpts[lastchangecpts[tstar]]+1;

        nchecktmp=0;
        for i in range(nchecklist):
            if tmplike[i]<=(lastchangelike[tstar]+pen):
                checklist[nchecktmp] = checklist[i]
                nchecktmp = nchecktmp+1

        nchecklist = nchecktmp
        checklist[nchecklist] = tstar
        nchecklist = nchecklist+1

    ncpts = 0
    last = n_all
    cptsout = [];
    while(last!=0):
        cptsout.append(cpt_set[last]);
        last = lastchangecpts[last]
        ncpts = ncpts+1
    cptsout.append(0)
    cptsout.reverse()
    return cptsout

# change-point detection functions
# minseglen determines the minimum distance between two successive cpts.

def cpt_detect_minseglen_PELT(eta, s2, f2_mat, k_plus, minseglen):
    
    n_all = f2_mat.shape[0]
    
    pen = 0.5*k_plus*math.log(n_all)
    lastchangecpts = np.ndarray(n_all+1, dtype='int');
    lastchangelike = np.ndarray(n_all+1);
    numchangecpts = np.ndarray(n_all+1, dtype='int');
    checklist = np.ndarray(n_all+1, dtype='int');
    tmplike = np.ndarray(n_all+1);
    
    # initialize F(0) = -penalty, cp(0) = NULL
    
    lastchangelike[0] = -pen;
    lastchangecpts[0] = 0;
    numchangecpts[0] = 0;
    
    # F(1) = C(y_0), cp(1) = 0,
    for j in range(minseglen, 2*minseglen):
        lastchangelike[j] = cost_function(eta, s2, f2_mat[0:j,:], k_plus)
        lastchangecpts[j] = 0
        numchangecpts[j] = 1
    
    nchecklist=2;
    checklist[0] = 0;
    checklist[1] = minseglen;
    
    for tstar in range(2*minseglen, n_all+1):
        
        for i in range(nchecklist):
            tmplike[i] = (lastchangelike[checklist[i]]
                          + cost_function(eta, s2, f2_mat[checklist[i]:tstar,:], k_plus)
                          + pen);
        
        whichout = tmplike[range(nchecklist)].argmin()
        minout = tmplike[whichout]
        lastchangelike[tstar] = minout
        lastchangecpts[tstar] = checklist[whichout]
        numchangecpts[tstar] = numchangecpts[lastchangecpts[tstar]]+1;
        
        nchecktmp=0;
        for i in range(nchecklist):
            if tmplike[i]<=(lastchangelike[tstar]+pen):
                checklist[nchecktmp] = checklist[i]
                nchecktmp = nchecktmp+1
        
        nchecklist = nchecktmp
        checklist[nchecklist] = tstar - (minseglen-1)
        nchecklist = nchecklist+1

    ncpts = 0
    last = n_all
    cptsout = [];
    while(last!=0):
        cptsout.append(last);
        last = lastchangecpts[last]
        ncpts = ncpts+1
    cptsout.append(0)
    cptsout.reverse()
    return cptsout


