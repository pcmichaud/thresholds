import numpy as np
import pandas as pd
from dgp import dgp
from report import tax_system, report
from multiprocessing import Pool
from functools import partial

class monte_carlo:
    def __init__(self, obs = 100000, T = 5, reps = 8, n_cpu = 8):
        self.N = obs
        self.T = T
        self.n_reps = reps
        self.n_cpu = n_cpu
        self.set_dgp()
        self.set_eti()
        self.set_taxes()
        self.set_controls()
        return
    def set_dgp(self, mu = 125e3, rho = 0.95, sigma = 0.3):
        self.mu = mu
        self.rho = rho
        self.sigma = sigma
        return
    def set_eti(self,eps = 1.0, eta = 0.0):
        self.eps = eps
        self.eta = eta
        return
    def set_controls(self,n_bins = 25,local = False):
        self.n_bins = n_bins
        self.local = local
        return
    def set_taxes(self, thres=[125e3,125e3,125e3,125e3],
           tau_lo = [0.4,0.4,0.4,0.4],
           tau_hi = [0.5,0.5,0.5,0.5],
           d_tau = [0,0,0.025,0.025],
           t_r = 3, n_p = 4):
        thres, tau_lo, tau_hi = tax_system(thres, tau_lo, tau_hi, d_tau,  self.T, t_r, n_p)
        self.thres = thres
        self.tau_lo = tau_lo
        self.tau_hi = tau_hi
        self.d_tau = d_tau
        self.t_r = t_r
        self.n_p = n_p
        return
    def simulate_dgp(self):
        n, prov = dgp(self.N, self.T, self.mu, self.rho, self.sigma)
        z  = report(n, prov, self.thres,
            self.tau_lo, self.tau_hi, self.t_r, eps = self.eps, eta = self.eta)
        return z, prov

    def one_iv(self, x, hs):
        z, prov  = self.simulate_dgp()
        n_b = self.n_bins
        n_obs = z.shape[0]*z.shape[1]

        L_z = np.c_[np.nan*np.ones(z.shape[0]),z[:,:-1]]
        D_logz = np.log(z) - np.log(L_z)

        D_nwr = np.zeros(z.shape,dtype='float64')
        D_atr = np.zeros(z.shape,dtype='float64')
        D_nwr[:,0] = np.nan
        D_atr[:,0] = np.nan
        for t in range(1,self.T):
            treated = (z[:,t]>=self.thres[prov,t])
            D_nwr[:,t] = (np.log(1-self.tau_hi[prov,t]) - np.log(1-self.tau_hi[prov,t-1]))*treated
            D_atr[:,t] = (self.tau_hi[prov,t] - self.tau_hi[prov,t-1])*np.where(treated,z[:,t]-self.thres[prov,t],0)/z[:,t]

        D_nwr_s = np.zeros(z.shape,dtype='float64')
        D_atr_s = np.zeros(z.shape,dtype='float64')

        D_nwr_s[:,0] = np.nan
        D_atr_s[:,0] = np.nan
        for t in range(1,self.T):
            D_nwr_s[:,t] = (np.log(1-self.tau_hi[prov,t])
                            - np.log(1-self.tau_hi[prov,t-1]))*(L_z[:,t]>=self.thres[prov,t])
            D_atr_s[:,t] = (self.tau_hi[prov,t]
                            - self.tau_hi[prov,t-1])*(L_z[:,t]-self.thres[prov,t])*(L_z[:,t]>self.thres[prov,t])/L_z[:,t]

        # years
        yrs = np.ones((self.N,1)) @ np.arange(self.T).reshape((1,self.T))

        # prov
        pro = prov.reshape((self.N,1)) @ np.ones((1,self.T))

        # bins
        n_bins = self.n_bins
        L_logz = np.log(L_z)
        q_z = np.quantile(L_logz[:,1:],q=np.linspace(0.01,0.99,n_bins))
        q_z = np.r_[np.min(L_logz[:,1:]),q_z,np.max(L_logz[:,1:])]
        bins = np.digitize(L_logz,q_z)
        # reshape long, drop t=0
        n_obs = (self.T-1)*self.N
        D_logz = D_logz[:,1:].reshape((n_obs,1))
        D_nwr = D_nwr[:,1:].reshape((n_obs,1))
        D_atr = D_atr[:,1:].reshape((n_obs,1))
        D_nwr_s = D_nwr_s[:,1:].reshape((n_obs,1))
        D_atr_s = D_atr_s[:,1:].reshape((n_obs,1))
        L_logz = L_logz[:,1:].reshape((n_obs,1))
        yrs = yrs[:,1:].reshape((n_obs,1))
        pro = pro[:,1:].reshape((n_obs,1))
        bins = bins[:,1:].reshape((n_obs,1))

        # creates dummies
        bins = pd.get_dummies(bins[:,0],dtype='int64').values
        bins = np.delete(bins,4,axis=1)
        n_b = bins.shape[1]
        yrs = pd.get_dummies(yrs[:,0],dtype='int64').values
        yrs = np.delete(yrs,0,axis=1)
        n_yrs = yrs.shape[1]
        pro = pd.get_dummies(pro[:,0],dtype='int64').values
        pro = np.delete(pro,0,axis=1)
        n_pro = pro.shape[1]
        yrs_bins = np.zeros((n_obs,n_yrs*n_b),dtype='int64')
        i = 0
        for t in range(n_yrs):
            for b in range(n_b):
                yrs_bins[:,i] = yrs[:,t]*bins[:,b]
                i +=1
        yrs_pro = np.zeros((n_obs,n_yrs*n_pro),dtype='int64')
        i = 0
        for t in range(n_yrs):
            for p in range(n_pro):
                yrs_pro[:,i] = yrs[:,t]*pro[:,p]
                i +=1
        bins_pro = np.zeros((n_obs,n_b*n_pro),dtype='int64')
        i = 0
        for b in range(n_b):
            for p in range(n_pro):
                bins_pro[:,i] = bins[:,b]*pro[:,p]
                i +=1
        const = np.ones((n_obs,1))
        dummies = np.concatenate([yrs,pro,bins,yrs_pro,bins_pro,yrs_bins,const],axis=1)
        # iv regression
        if self.local:
            betas = np.zeros(len(hs))
            thres = 0.9*self.thres[0,0]
            for i,h in enumerate(hs):
                logh = np.log(h)
                D_nwr_h = D_nwr * (L_logz - logh)
                D_nwr_s_h = D_nwr_s * (L_logz - logh)
                X = np.concatenate([D_nwr,D_nwr_h,dummies],axis=1)
                Z = np.concatenate([D_nwr_s,D_nwr_s_h,dummies],axis=1)
                y = D_logz
                d = L_logz[:,0] - logh
                d = d/0.20
                w = np.zeros(n_obs)
                d = np.absolute(d)
                w = np.where(d<=1.0,d,w)
                w = np.where(L_logz[:,0]<np.log(thres),1,w)
                X = X[w>0,:]
                Z = Z[w>0,:]
                y = y[w>0,0]
                w = w[w>0]
                for k in range(X.shape[1]):
                    X[:,k] = np.multiply(X[:,k],w)
                for k in range(Z.shape[1]):
                    Z[:,k] = np.multiply(Z[:,k],w)
                y = np.multiply(y,w)
                means = np.mean(X,axis=0)
                idx = np.where(np.absolute(means)<1e-4)
                X = np.delete(X,idx,axis=1)
                Z = np.delete(Z,idx,axis=1)
                beta = np.linalg.inv(Z.T @  X) @ (Z.T  @ y)
                betas[i] = beta[0]
        else :
                X = np.concatenate([D_nwr,D_atr,dummies],axis=1)
                Z = np.concatenate([D_nwr_s,D_atr_s,dummies],axis=1)
                y = D_logz
                means = np.mean(X,axis=0)
                idx = np.where(np.absolute(means)<1e-4)
                X = np.delete(X,idx,axis=1)
                Z = np.delete(Z,idx,axis=1)
                beta = np.linalg.inv(Z.T @  X) @ (Z.T  @ y)
                #print(beta[0:2,0].T)

        return beta[0:2,0].T


    def run(self):
        if self.local:
            one_iv = partial(self.one_iv,hs=[175e3,225e3,275e3,325e3])
        else :
            one_iv = partial(self.one_iv,hs=[])
        with Pool(self.n_cpu) as p:
            betas = p.map(one_iv,np.arange(self.n_reps))
        return betas
