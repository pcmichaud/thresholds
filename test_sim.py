import numpy as np
import pandas as pd
from iv import monte_carlo

if __name__ == '__main__':
    #n_s = 5
    #etis = [1.0]
    #np.linspace(0.0,0.1,n_s)
    #results = pd.DataFrame(index=np.arange(n_s),columns=['eps','mean','sd'])
    #for i,eti in enumerate(etis):
    mc = monte_carlo(obs = 100000,reps=10,n_cpu=1)
    mc.set_eti(eps=0.5)
    mc.set_dgp(mu=125e3,rho=0.91,sigma=0.6)
    z = mc.simulate_dgp()
    print(np.mean(z,axis=0))

    #betas = mc.run()
    #print(np.mean(betas,axis=0))
    #print(np.std(betas,axis=0))
        #results.loc[i,''] = eti
        #results.loc[i,'mean'] = np.mean(betas)
        #results.loc[i,'sd'] = np.std(betas)
    #print(results)
    #results.to_csv('eps_experiment.csv')

