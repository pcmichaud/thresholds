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

def report(n, prov, thres, tau_lo, tau_hi, eps = 0.5, eta = 0.25):
	N = n.shape[0]
	T = n.shape[1]
	_z = np.zeros((N,T),dtype='float64')
	_tau_hi = np.zeros((N,T),dtype='float64')
	_tau_lo = np.zeros((N,T),dtype='float64')
	_thres = np.zeros((N,T),dtype='float64')
	for i in range(N):
		p = prov[i]
		_z[i,0] = n[i,0]
		_tau_hi[i,0] = tau_hi[p,0]
		_tau_lo[i,:] = tau_lo[p,:]
		_thres[i,:] = thres[p,:]
		for t in range(1,T):
			k, tau, last_tau = thres[p,t], tau_hi[p,t], tau_hi[p,t-1]
			d_tau = (np.log(1-tau) - np.log(1-last_tau))*(n[i,t]>k)
			d_atr = (tau - last_tau)*max(n[i,t]-k,0)
			_z[i,t] = n[i,t]*np.exp(eps*d_tau - eta*d_atr)
			_tau_hi[i,t] = tau





        _z = pd.DataFrame(_z,index=np.arange(N),columns=np.arange(T))
	_tau_hi = pd.DataFrame(_tau_hi,index=np.arange(N),columns=np.arange(T))
	_tau_lo = pd.DataFrame(_tau_lo,index=np.arange(N),columns=np.arange(T))
	_thres = pd.DataFrame(_thres,index=np.arange(N),columns=np.arange(T))
	df = _z.stack().to_frame()
	df = df.merge(_tau_hi.stack().to_frame(),left_index=True,right_index=True)
	df.columns = ['z','tau_1']
	df = df.merge(_tau_lo.stack().to_frame(),left_index=True,right_index=True)
	df.columns = ['z','tau_1','tau_0']
	df = df.merge(_thres.stack().to_frame(),left_index=True,right_index=True)
	df.columns = ['z','tau_1','tau_0','thres']
	df.index.names = ['id','year']
	df['L_tau_1'] = lag(df,'tau_1',1)
	df['L_z'] = lag(df,'z',1)
	df['prov'] = np.repeat(prov,T)
	df['atr'] = (df['tau_0']*df[['z','thres']].min(axis=1) + df['tau_1']*(df['z']-df['thres']).clip(lower=0.0))/df['z']
	df['net'] = np.where(df['z']<df['thres'],np.log(1-df['tau_0']),np.log(1-df['tau_1']))

	df['D_s_net'] = (np.log(1-df['tau_1']) - np.log(1-df['L_tau_1']))*(df['L_z']>=df['thres'])
	df['D_s_atr'] = (df['tau_1'] - df['L_tau_1'])*(df['L_z']-df['thres']).clip(lower=0.0)/df['L_z']
	df['L_atr'] = lag(df,'atr',1)
	df['L_net'] = lag(df,'net',1)
	df['D_atr'] = df['atr'] - df['L_atr']
	df['D_net'] = df['net'] - df['L_net']
	df['L_logz'] = np.log(df['L_z'])
	df['logz'] = np.log(df['z'])
	df['D_logz'] = df['logz'] - df['L_logz']
	df = df.loc[df.index.get_level_values(1)!=0,:]
	df = df[df.z>=80e3]
	return df



