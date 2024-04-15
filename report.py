import numpy as np
import pandas as pd

def tax_system(thres, tau_lo, tau_hi, d_tau,  T = 10, t_r = 4, n_p = 4):
       _thres = np.zeros((n_p,T),dtype='float64')
       _tau_lo = np.zeros((n_p,T),dtype='float64')
       _tau_hi = np.zeros((n_p,T),dtype='float64')
       for  t in range(T):
             _thres[:,t] = thres[:]
             _tau_lo[:,t] = tau_lo[:]
             _tau_hi[:,t] = tau_hi[:]
             if t>=t_r:
                     _tau_hi[:,t] = _tau_hi[:,t] + d_tau
       return _thres, _tau_lo, _tau_hi

def lag(df,var,j):
        val = df.groupby('id').shift(j)[var]
        return df.index.map(val)

def report(n, prov, thres, tau_lo, tau_hi, t_r,  eps = 0.5, eta = 0.25):
        N = n.shape[0]
        T = n.shape[1]
        z = np.zeros((N,T),dtype='float64')
        z[:,0] = n[:,0]
        for t in range(1,T):
            k = thres[prov,t]
            tau = tau_hi[prov,t]
            if t>=t_r:
                last_tau = tau_hi[prov,t_r-1]
            else :
                last_tau = tau_hi[prov,t-1]
            d_tau = (np.log(1-tau) - np.log(1-last_tau))*(n[:,t]>k)
            d_atr = (tau - last_tau)*(n[:,t]-k)*(n[:,t]>k)/n[:,t]
            z[:,t] = n[:,t]*np.exp(eps*d_tau - eta*d_atr)
        return z
