import numpy as np
import pandas as pd
from iv import monte_carlo

if __name__ == '__main__':
    n_s = 4
    rhos = np.linspace(0.85,0.95,n_s)
    results = pd.DataFrame(index=np.arange(n_s),columns=['rho','mean','sd'])
    for i,rho in enumerate(rhos):
        mc = monte_carlo(obs = 200000,reps=500,n_cpu=50)
        mc.set_eti(eps=0.5)
        mc.set_taxes(d_tau = [0,0.0,0.025,0.025])
        mc.set_dgp(mu=100e3,rho=rho,sigma=0.5)
        betas = mc.run()
        results.loc[i,'rho'] = rho
        results.loc[i,'mean'] = np.mean(betas)
        results.loc[i,'sd'] = np.std(betas)
    print(results)
    for c in results.columns:
        results[c] = results[c].astype('float64')
    results.round(3).to_latex('output/rho_experiment.tex')
