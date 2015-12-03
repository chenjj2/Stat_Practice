import numpy as np
from scipy.stats import norm

### fix the number of different populations
n_pop = 4

### recover c with continuous criteria
def split_hyper(hyper):
	c0, power, sigma, trans = \
	hyper[0], hyper[1:1+n_pop], hyper[1+n_pop:1+2*n_pop], hyper[1+2*n_pop:]
	
	c = np.zeros_like(power)
	c[0] = c0
	for i in range(1,n_pop):
		c[i] = c[i-1] * trans[i-1]**(power[i-1]-power[i])
	return c, power, sigma, trans	


### indicate which M belongs to population i given transition parameter
def indicate(M, trans, i):
	ts = np.insert(np.insert(trans, n_pop-1, np.inf), 0, -np.inf)
	ind = (M>=ts[i]) & (M<ts[i+1])
	return ind



### piece power law
def piece_power(hyper, M, prob_R):
	c, power, sigma, trans = split_hyper(hyper)
	
	R = np.zeros_like(M)
	for i in range(4):
		ind = indicate(M, trans, i)
		mu = c[i] * M[ind]**power[i]
		R[ind] = norm.ppf(prob_R[ind], mu, sigma[i])
		
	return R


### piece power law, different sigma
def piece_power_frac(hyper, M, prob_R):
	c, power, sigma, trans = split_hyper(hyper)
	
	R = np.zeros_like(M)
	for i in range(4):
		ind = indicate(M, trans, i)
		mu = c[i] * M[ind]**power[i]
		''' note the change here, sigma is now mu*sigma '''
		R[ind] = norm.ppf(prob_R[ind], mu, mu*sigma[i])
		
	return R

### turn mass to log(m) and merr to merr/m, likewise for radius
def convert_data(dat):

	nrow, ncol = np.shape(dat)

	mass = dat[:,0].reshape(nrow,1)
	merr = dat[:,1].reshape(nrow,1)
	rad = dat[:,2].reshape(nrow,1)
	raderr = dat[:,3].reshape(nrow,1)

	log_dat = np.hstack(( np.log10(mass), np.log10(np.e) * merr/mass, np.log10(rad), np.log10(np.e) * raderr/rad ))

	return log_dat



### split hyper and derive c
def split_hyper_linear(hyper):
	c0, slope,sigma, trans = \
	hyper[0], hyper[1:1+n_pop], hyper[1+n_pop:1+2*n_pop], hyper[1+2*n_pop:]

	c = np.zeros_like(slope)
	c[0] = c0
	for i in range(1,n_pop):
		# trans[0] * slope[0] + c[0] = trans[0] * slope[1] + c[1]
		# c[1] = c[0] + trans[0] * (slope[0]-slope[1])
		c[i] = c[i-1] + trans[i-1]*(slope[i-1]-slope[i])

	return c, slope, sigma, trans

### model: straight line
def piece_linear(hyper, M, prob_R):
	c, slope, sigma, trans = split_hyper_linear(hyper)
	R = np.zeros_like(M)
	for i in range(4):
		ind = indicate(M, trans, i)
		mu = c[i] + M[ind]*slope[i]
		R[ind] = norm.ppf(prob_R[ind], mu, sigma[i])

	return R

### based on piece_linear
### but the intrinsic scatter in degenerate group is linear rather than constant
def piece_linear_complex(hyper, M, prob_R):
	c, slope, sigma, trans = split_hyper_linear(hyper)
	R = np.zeros_like(M)
	for i in range(4):
		ind = indicate(M, trans, i)
		mu = c[i] + M[ind]*slope[i]
		if i==2: # degenerate group
			sig = sigma[2] + (sigma[1]-sigma[2]) * (trans[2]-M[ind]) / (trans[2]-trans[1])
		else:
			sig = sigma[i]
		R[ind] = norm.ppf(prob_R[ind], mu, sig)

	return R	



