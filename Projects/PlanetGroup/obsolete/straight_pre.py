'''
just fit the log(M) and log(R) with a staight line
treat the 2D error as gaussian for simplicity, 
which is definitely not true, but close when S/N is high
'''

import numpy as np
import sys
sys.path.append('../..')
from NaiveMC.mcmc import hbm_joint_cdf
from scipy.stats import norm, uniform
from func import indicate, split_hyper_linear, piece_linear, convert_data

### parse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float)
parser.add_argument("--lindir")
args = parser.parse_args()

if args.temperature:
	temperature = args.temperature
else:
	temperature = 1.
if args.lindir:
	lindir = args.lindir
else:
	lindir = ''

### fix the number of different populations
n_pop = 4

### data
#file = '/Users/jingjing/Work/DK_project/Data/Mine/PlanetGroup.txt'
file = 'PlanetGroup.txt'
dat = np.loadtxt(file)
m_exact = (dat[:,1]==0.)
dat_fixm = convert_data(dat[m_exact]) 
M0 = dat_fixm[:,0]
dat_varm = convert_data(dat[~m_exact]) 


### some static parameter 
n_fixm = np.shape(dat_fixm)[0]
n_varm = np.shape(dat_varm)[0]
m_max = np.max(np.hstack((dat_fixm[:,0], dat_varm[:,0])))
m_min = np.min(np.hstack((dat_fixm[:,0], dat_varm[:,0])))


### inverse sampling
def inverse_hyper(hyper_prob):
	prob_C0, prob_slope, prob_sigma, prob_trans = \
	hyper_prob[0], hyper_prob[1:1+n_pop], hyper_prob[1+n_pop:1+2*n_pop], hyper_prob[1+2*n_pop:3*n_pop]
	
	C0 = uniform.ppf(prob_C0,-1.,2.)
	slope = norm.ppf(prob_slope, 0.,5.)
	sigma = 10.**( uniform.ppf(prob_sigma, -3., 3.) )
	trans = np.sort( uniform.ppf(prob_trans, m_min, m_max-m_min) ) # sort

	hyper = np.hstack(( C0, slope, sigma, trans ))

	return hyper

# R(0,i) for fix mass, and then M(1,i), R(1,i) for variable mass, 0/1 indicates fix/var
def inverse_local(local_prob, hyper):
	# R(0,i) for fix mass
	prob_R0 = local_prob[0:n_fixm]
	R0 = piece_linear(hyper, M0, prob_R0)

	# M(1,i) for variable mass
	prob_M1 = local_prob[n_fixm:n_fixm+n_varm]
	M1 = uniform.ppf(prob_M1, -4., 10.)

	# R(1,i) for varibable mass
	prob_R1 = local_prob[n_fixm+n_varm:]
	R1 = piece_linear(hyper, M1, prob_R1)

	local = np.hstack((R0, M1, R1))

	return local


### return sigma corresponding to M/R
def split_group(hyper, local):
	c, slope, sigma, trans = split_hyper_linear(hyper)

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

	L0 = 0. - 0.5* np.sum( (Rob0-Rth0)**2./(Rerr0**2.+sigma_like_R0**2.) ) \
		- 0.5* np.sum( np.log( Rerr0**2.+sigma_like_R0**2. ) )

	# variable mass
	Mob1 = dat_varm[:,0]
	Merr1 = dat_varm[:,1]
	Rob1 = dat_varm[:,2]
	Rerr1 = dat_varm[:,3]
	Mth1 = local[n_fixm:n_fixm+n_varm]
	Rth1 = local[n_fixm+n_varm:]

	L1 = 0. - 0.5* np.sum( (Mob1-Mth1)**2./Merr1**2. ) \
		- 0.5* np.sum( (Rob1-Rth1)**2./(Rerr1**2.+sigma_like_R1**2. ) ) \
		- 0.5* np.sum( np.log( Rerr1**2.+sigma_like_R1**2. ) )

	L = L0 + L1
	return L/temperature

### mcmc

n_step = int(5e5)

hyper_prob0 = np.random.uniform(0.,1.,3*n_pop)
hyper_stepsize = 1e-3 * np.ones(3*n_pop)
local_prob0 = 0.5 * np.ones(n_fixm + 2*n_varm)
local_stepsize = 1e-3 * np.ones(n_fixm + 2*n_varm)

import time
#print 'start:', time.asctime()

hyper_prob_chain, hyper_chain, local_prob_chain, local_chain, \
loglike_chain, repeat_chain, stop_step = \
hbm_joint_cdf(hyper_prob0, hyper_stepsize, local_prob0, local_stepsize, n_step,\
			inverse_hyper, inverse_local, \
			loglike_func, data = [dat_fixm, dat_varm], \
			trial_upbound = 10*n_step)

#print 'end', time.asctime()
#print 'stop', stop_step

### plot
np.savetxt(lindir+'/str_hyper_prob_'+str(int(temperature))+'.out',hyper_prob_chain[:stop_step,:])
np.savetxt(lindir+'/str_hyper_'+str(int(temperature))+'.out',hyper_chain[:stop_step,:])
np.savetxt(lindir+'/str_loglike_'+str(int(temperature))+'.out',loglike_chain[:stop_step])
np.savetxt(lindir+'/str_repeat_'+str(int(temperature))+'.out',repeat_chain[:stop_step])


