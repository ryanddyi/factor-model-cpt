import argparse
import numpy as np
import solver
from lib.utils import GARCH_normalize
import random

random.seed(1)

parser = argparse.ArgumentParser(description='Unsupervised learning\
                                    on multivariate time series data')
parser.add_argument('data', metavar='.csv', help='time series data file')
parser.add_argument('-out', help='output directory')

args = parser.parse_args()

if __name__ == '__main__':

    with open(args.data) as f:
        ncols = len(f.readline().split(','))

    y_mat = np.loadtxt(open(args.data, 'rb'), delimiter=',', skiprows=1, usecols=range(1,ncols))

    # preprocess with garch model
    for i in range(y_mat.shape[1]):
        y_mat[:,i] = GARCH_normalize(y_mat[:,i])
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

    model.em_iterator(1000, False)

    Beta, Lambda_ts = model.final_rescale()

    np.savetxt(args.out+'Beta.csv', Beta, delimiter=',')
    np.savetxt(args.out+'Lambda_ts.csv', Lambda_ts, delimiter=',')
    np.savetxt(args.out+'tau.csv', model.tau, delimiter=',')
    np.savetxt(args.out+'sigma2.csv', model.sigma2, delimiter=',')
