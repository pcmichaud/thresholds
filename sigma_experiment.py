import numpy as np
import pandas as pd
from iv import monte_carlo

if __name__ == '__main__':
    n_s = 3
    sigmas = np.linspace(0.1,0.7,n_s)
    results = pd.DataFrame(index=np.arange(n_s),columns=['sigma','mean','sd'])
    for i,sigma in enumerate(sigmas):
        mc = monte_carlo(obs = 200000,reps=500,n_cpu=250)
        mc.set_eti(eps=0.0)
        mc.set_dgp(mu=100e3,rho=0.9,sigma=sigma)
        betas = mc.run()
        results.loc[i,'sigma'] = sigma
        results.loc[i,'mean'] = np.mean(betas)
        results.loc[i,'sd'] = np.std(betas)
    print(results)
    results.to_csv('sigma_experiment.csv')

