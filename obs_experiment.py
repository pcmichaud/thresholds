import numpy as np
import pandas as pd
from iv import monte_carlo

if __name__ == '__main__':
    n_s = 5
    obs = np.linspace(100,500,n_s)
    results = pd.DataFrame(index=np.arange(n_s),columns=['N','mean','sd'])
    for i,n in enumerate(obs):
        mc = monte_carlo(obs = n*1000,reps=500,n_cpu=50)
        mc.set_eti(eps=0.5)
        mc.set_taxes(d_tau = [0,0.0,0.025,0.025])
        mc.set_dgp(mu=100e3,rho=0.9,sigma=0.5)
        betas = mc.run()
        results.loc[i,'N'] = n
        results.loc[i,'mean'] = np.mean(betas)
        results.loc[i,'sd'] = np.std(betas)
    print(results)
    for c in results.columns:
        results[c] = results[c].astype('float64')
    results.round(3).to_latex('output/obs_experiment.tex')
