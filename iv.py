import numpy as np 
import pandas as pd 
from dgp import dgp
from report import tax_system, report 
from multiprocessing import Pool

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
	def set_controls(self,n_bins = 25):
		self.n_bins = n_bins 
		return 
	def set_taxes(self, thres=[100e3,100e3,100e3,100e3],
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
		df  = report(n, prov, self.thres, 
		    self.tau_lo, self.tau_hi, eps = self.eps, eta = self.eta)
		return df
	def one_iv(self, x):
		df = self.simulate_dgp()
		n_b = self.n_bins
		n_obs = len(df)
		d_logz = df['D_logz'].values.reshape((n_obs,1))
		d_net = df['D_net'].values.reshape((n_obs,1))
		d_s_net = df['D_s_net'].values.reshape((n_obs,1))
		n_obs = d_logz.shape[0]
		yr = pd.get_dummies(df.index.get_level_values(1),drop_first=True,dtype='int64').values
		n_yr = yr.shape[1]
		pr = pd.get_dummies(df['prov'],drop_first=True,dtype='int64').values
		n_pr = pr.shape[1]
		df['bins'] = pd.qcut(df['L_logz'],q=n_b,labels=[x for x in range(n_b)])
		bins = pd.get_dummies(df['bins'],drop_first=True,dtype='int64').values
		n_b -=1 
		yr_bins = np.zeros((n_obs,n_yr*n_b),dtype='int64')
		i = 0
		for t in range(n_yr):
			for b in range(n_b):
				yr_bins[:,i] = yr[:,t]*bins[:,b]
				i +=1 
		yr_pr = np.zeros((n_obs,n_yr*n_pr),dtype='int64')
		i = 0
		for t in range(n_yr):
			for p in range(n_pr):
				yr_pr[:,i] = yr[:,t]*pr[:,p]
				i +=1 
		bins_pr = np.zeros((n_obs,n_b*n_pr),dtype='int64')
		i = 0
		for b in range(n_b):
			for p in range(n_pr):
				bins_pr[:,i] = bins[:,b]*pr[:,p]
				i +=1 
		const = np.ones((n_obs,1))
		dummies = np.concatenate([yr,pr,bins,yr_pr,bins_pr,yr_bins,const],axis=1)
		# first stage
		X = np.concatenate([d_net,dummies],axis=1)
		Z = np.concatenate([d_s_net,dummies],axis=1)
		beta = np.linalg.inv(Z.T @ X) @ (Z.T @ d_logz)
		return beta[0,0]
	def run(self):
		with Pool(self.n_cpu) as p: 
			betas = p.map(self.one_iv,np.arange(self.n_reps))
		return betas 
