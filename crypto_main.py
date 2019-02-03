import argparse
import numpy as np
import solver
import lib.utils
import random

random.seed(1)

parser = argparse.ArgumentParser(description='Unsupervised learning\
                                    on crypto price time series data')
parser.add_argument('data', metavar='.csv', help='time series data file')

args = parser.parse_args()

if __name__ == '__main__':

    with open(args.data) as f:
        ncols = len(f.readline().split(','))

    y_mat = np.loadtxt(open(args.data, 'rb'), delimiter=',', skiprows=1, usecols=range(1,ncols))
    for i in range(y_mat.shape[1]):
        y_mat[:,i] = utils.GARCH_normalize(y_mat[:,i])
    y_mat = solver.normalize(y_mat)
    print(y_mat)

    model = solver.FactorModel(y_mat, k_max=20)

    model.cpt_config(subsetting=True)
    model.param_init()
    #model.Beta = np.loadtxt(open("Beta_init.csv", "rb"), delimiter=",") # for debug

    nstep = 200
    for delta0 in [1, 2, 5, 10, 20, 50]:
        model.delta0_reconfig(delta0)
        for PXL in [True, False]:
            model.em_iterator(nstep, PXL)
            #print(model.log_likelihood())
            print(model.tau)
            print(model.k_plus)
            print(model.Beta)

    model.em_iterator(500, False)
    #print(model.log_likelihood())
    print(model.tau)
    print(model.k_plus)
    print(model.Beta)

    model.em_iterator(500, False)
    #print(model.log_likelihood())
    print(model.tau)
    print(model.k_plus)
    print(model.Beta)

    Beta, Lambda_ts = model.final_rescale()

    np.savetxt('/output/crypto/Beta.csv', Beta, delimiter=',')
    np.savetxt('/output/crypto/Lambda_ts.csv', Lambda_ts, delimiter=',')
    np.savetxt('/output/crypto/tau.csv', model.tau, delimieter=',')
    np.savetxt('/output/crypto/sigma2.csv', model.sigma2, delimieter=',')
