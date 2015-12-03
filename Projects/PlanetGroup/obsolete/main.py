'''
fit the Radius-Mass relation in a wide mass range with power law
categorize objects
find the transition point
find the relation in each category
add intrinsic scatter in the power law, like 2015-Wolfgang
fix number of category
'''

### import 
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
from NaiveMC.mcmc import hbm_joint_cdf
from scipy.stats import norm, uniform
from func import split_hyper, indicate, piece_power

### fix the number of different populations
n_pop = 4

### data
#file = '/Users/jingjing/Work/DK_project/Data/Mine/PlanetGroup.txt'
file = 'PlanetGroup.txt'
dat = np.loadtxt(file)
m_exact = (dat[:,1]==0.)
dat_fixm = dat[m_exact] # 23 objects
M0 = dat_fixm[:,0]
dat_varm = dat[~m_exact] # 265 objects


### some static parameter 
n_fixm = 23
n_varm = 265




### inverse sampling
def inverse_hyper(hyper_prob):
	prob_C0, prob_power, prob_sigma, prob_trans = \
	hyper_prob[0], hyper_prob[1:1+n_pop], hyper_prob[1+n_pop:1+2*n_pop], hyper_prob[1+2*n_pop:3*n_pop]
	
	C0 = np.exp( uniform.ppf(prob_C0,-3.,6.) )
	power = norm.ppf(prob_power, 0.,5.)
	sigma = 10.**( uniform.ppf(prob_sigma, -2., 2.) )
	trans = np.sort( 10.**( uniform.ppf(prob_trans, -4., 10.) ) ) # sort

	hyper = np.hstack(( C0, power, sigma, trans ))

	return hyper

# R(0,i) for fix mass, and then M(1,i), R(1,i) for variable mass, 0/1 indicates fix/var
def inverse_local(local_prob, hyper):
	# R(0,i) for fix mass
	prob_R0 = local_prob[0:n_fixm]
	R0 = piece_power(hyper, M0, prob_R0)

	# M(1,i) for variable mass
	prob_M1 = local_prob[n_fixm:n_fixm+n_varm]
	M1 = 10.**( uniform.ppf(prob_M1, -4., 10.) )

	# R(1,i) for varibable mass
	prob_R1 = local_prob[n_fixm+n_varm:]
	R1 = piece_power(hyper, M1, prob_R1)

	local = np.hstack((R0, M1, R1))

	return local


### return sigma corresponding to M/R
def split_group(hyper, local):
	c, power, sigma, trans = split_hyper(hyper)

	M1 = local[n_fixm:n_fixm+n_varm]
	
	sig_like_M0 = np.zeros_like(M0)
	for i in range(n_pop):
		sig_like_M0 += sigma[i] * indicate(M0,trans,i)

	sig_like_M1 = np.zeros_like(M1)
	for i in range(n_pop):
		sig_like_M1 += sigma[i] * indicate(M1,trans,i)

	return sig_like_M0, sig_like_M1


### likelihood
def loglike_func(hyper,local, dat_fixm, dat_varm):
	sigma_like_R0, sigma_like_R1 = split_group(hyper, local)

	# fix mass
	Rob0 = dat_fixm[:,2]
	Rerr0 = dat_fixm[:,3]
	Rth0 = local[0:n_fixm]

	L0 = 0. - np.sum( (Rob0-Rth0)**2./(Rerr0**2.+sigma_like_R0**2.) ) \
		- 0.5* np.sum( np.log( Rerr0**2.+sigma_like_R0**2. ) )

	# variable mass
	Mob1 = dat_varm[:,0]
	Merr1 = dat_varm[:,1]
	Rob1 = dat_varm[:,2]
	Rerr1 = dat_varm[:,3]
	Mth1 = local[n_fixm:n_fixm+n_varm]
	Rth1 = local[n_fixm+n_varm:]

	L1 = 0. - np.sum( (Mob1-Mth1)**2./Merr1**2. ) \
		- np.sum( (Rob1-Rth1)**2./(Rerr1**2.+sigma_like_R1**2. ) ) \
		- 0.5* np.sum( np.log( Rerr1**2.+sigma_like_R1**2. ) )

	L = L0 + L1
	return L


### mcmc

n_step = int(5e5)

hyper_prob0 = np.array([0.5, 0.52, 0.54, 0.49, 0.58, 0.2, 0.4, 0.6, 0.8, 0.5, 0.65, 0.8])
hyper_stepsize = 1e-4 * np.ones(3*n_pop)
local_prob0 = 0.5 * np.ones(n_fixm + 2*n_varm)
local_stepsize = 1e-4 * np.ones(n_fixm + 2*n_varm)

import time
print 'start:', time.asctime()

hyper_prob_chain, hyper_chain, local_prob_chain, local_chain, \
loglike_chain, repeat_chain, stop_step = \
hbm_joint_cdf(hyper_prob0, hyper_stepsize, local_prob0, local_stepsize, n_step,\
			inverse_hyper, inverse_local, \
			loglike_func, data = [dat_fixm, dat_varm], \
			trial_upbound = 10*n_step)

print 'end', time.asctime()
print 'stop', stop_step

### plot
np.savetxt('hyper_prob.out',hyper_prob_chain[:stop_step,:])
np.savetxt('hyper.out',hyper_chain[:stop_step,:])
np.savetxt('loglike.out',loglike_chain[:stop_step])
np.savetxt('repeat.out',repeat_chain[:stop_step])


