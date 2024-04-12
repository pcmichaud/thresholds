import numpy as np 
import pandas as pd 
from iv import monte_carlo 

if __name__ == '__main__':
	mc = monte_carlo()
	betas = mc.run()
	print(mc.eps, np.mean(betas),np.std(betas))


