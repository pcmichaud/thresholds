import numpy as np
import pandas as pd
from iv import monte_carlo
from itertools import product

if __name__ == '__main__':
    n_i = 3
    n_e = 3
    etas = np.linspace(0.0,1.0,n_i)
    epss = np.linspace(0.0,1.0,n_e)
    labels_eti  = ['eps','eta']
    runs = [x for x in range(n_i*n_e)]
    labels = list(product(*[runs,labels_eti]))
    ids = pd.MultiIndex.from_tuples(labels)
    results = pd.DataFrame(index=ids,columns=['true value','mean','sd'])
    k = 0
    for i,eps in enumerate(epss):
        for j,eta in enumerate(etas):
            mc = monte_carlo(obs = 350000,reps=1000,n_cpu=10)
            mc.set_eti(eps=eps, eta = eta)
            mc.set_controls(local=False)
            mc.set_taxes(d_tau = [0,0.0,0.025,0.025], thres=[125e3,150e3,175e3,100e3])
            mc.set_dgp(mu=100e3,rho=0.9,sigma=0.5)
            betas = np.array(mc.run())
            results.loc[(k,'eps'),'true value'] = eps
            results.loc[(k,'eps'),'mean'] = np.mean(betas[:,0])
            results.loc[(k,'eps'),'sd'] = np.std(betas[:,0])
            results.loc[(k,'eta'),'true value'] = -eta
            results.loc[(k,'eta'),'mean'] = np.mean(betas[:,1])
            results.loc[(k,'eta'),'sd'] = np.std(betas[:,1])
            k +=1 
    print(np.mean(betas,axis=0))
    print(np.std(betas,axis=0))
    print(results)
    for c in results.columns:
        results[c] = results[c].astype('float64')
    results.round(3).to_latex('output/eta_experiment.tex')
