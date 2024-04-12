import numpy as np 

def ini_normal(N,mu,sigma):
    X = np.exp(np.log(mu) + sigma*np.random.normal(size=N) - 0.5*sigma**2)
    return X

def rnd_normal(X,mu, rho,sigma):
    N = X.shape[0]
    u = np.random.normal(size=N)
    Xp = np.exp(np.log(mu)*(1-rho) + rho*np.log(X) + sigma * u)
    return Xp

def dgp(N, T, mu, rho, sigma, n_p = 4, burn_in = 10):
    X = np.zeros((N,T),dtype='float64')
    sigma_s = sigma/np.sqrt((1-rho**2))
    x = ini_normal(N,mu,sigma_s)
    for t in range(burn_in):
    	x = rnd_normal(x,mu,rho,sigma)
    X[:,0] = x
    for t in range(1,T):
        X[:,t] = rnd_normal(X[:,t-1],mu,rho,sigma)
    prov = np.random.choice(np.arange(n_p),size=(N))
    return X, prov
