import numpy as np
from scipy.special import erfinv

def inverse_lognormal(pr,mu,sigma):
	power = mu + 2.**0.5*sigma * erfinv(2.*pr - 1.)
	x = np.exp(power)
	return x