import argparse
import numpy as np
import solver

parser = argparse.ArgumentParser(description='Unsupervised learning\
                                    on crypto price time series data')
parser.add_argument('data', metavar='.csv', help='time series data file')

args = parser.parse_args()

if __name__ == '__main__':

    with open(args.data) as f:
        ncols = len(f.readline().split(','))

    y_mat = np.loadtxt(open(args.data, 'rb'), delimiter=',', skiprows=1, usecols=range(1,ncols))
    y_mat = solver.normalize(y_mat)

    model = solver.FactorModel(y_mat, k_max=10)

    model.cpt_config()
    model.param_init()

    nstep = 20
    for delta0 in [5,10,20,50]:
        for PXL in [True, False]:
            model.em_iterator(nstep, PXL)
            print(model.tau)
            print(model.k_plus)
            print(model.Beta)
