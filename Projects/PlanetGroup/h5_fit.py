'''
based on straight_fit (final model)
use three transition points, all constant intrinsic scatter
change the log likelihood, because we suspect the previous one is doing double prior on R_theory
'''

import numpy as np
import sys
sys.path.append('../..')
from NaiveMC.mcmc import hbm_joint_cdf
from scipy.stats import norm, uniform
from func import indicate, split_hyper_linear, piece_linear, convert_data

###
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("hyperic")
parser.add_argument("hyperstep")
args = parser.parse_args()

### seed
import os
pid = os.getpid()
np.random.seed(pid)

print 'output directory', args.dir
print 'random seed', pid
print 'hyper prob0', args.hyperic
print 'hyper stepsize', args.hyperstep

### fix the number of different populations
n_pop = 3   

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


### inverse sampling
def inverse_hyper(hyper_prob):
	prob_C0, prob_slope, prob_sigma, prob_trans = \
	hyper_prob[0], hyper_prob[1:1+n_pop], hyper_prob[1+n_pop:1+2*n_pop], hyper_prob[1+2*n_pop:3*n_pop]
	
	C0 = uniform.ppf(prob_C0,-1.,2.)
	slope = norm.ppf(prob_slope, 0.,5.)
	sigma = 10.**( uniform.ppf(prob_sigma, -3., 5.) )
	trans = np.sort( uniform.ppf(prob_trans, -4., 10.) )

	hyper = np.hstack(( C0, slope, sigma, trans ))

	return hyper

# R(0,i) for fix mass, and then M(1,i), R(1,i) for variable mass, 0/1 indicates fix/var
def inverse_local(local_prob, hyper):	
	# M(1,i) for variable mass
	M1 = uniform.ppf(local_prob, -4., 10.)

	return M1


### return sigma corresponding to M/R
def split_group(hyper, local):
	c, slope, sigma, trans = split_hyper_linear(hyper, n_pop)

	M1 = local + 0.
	
	sig_like_M0 = np.zeros_like(M0)
	for i in range(n_pop):
		sig_like_M0 += sigma[i] * indicate(M0,trans,i, n_pop)

	sig_like_M1 = np.zeros_like(M1)
	for i in range(n_pop):
		sig_like_M1 += sigma[i] * indicate(M1,trans,i, n_pop)

	return sig_like_M0, sig_like_M1

### likelihood
def loglike_func(hyper,local, dat_fixm, dat_varm):
	sigma_like_R0, sigma_like_R1 = split_group(hyper, local)

	# fix mass
	Rob0 = dat_fixm[:,2]
	Rerr0 = dat_fixm[:,3]
	fM0 = piece_linear(hyper, M0, 0.5*np.ones_like(M0), n_pop)

	L0 = 0. - 0.5* np.sum( (Rob0-fM0)**2./(Rerr0**2.+sigma_like_R0**2.) ) \
			- 0.5* np.sum( np.log(Rerr0**2.+sigma_like_R0**2.) )

	# variable mass
	Mob1 = dat_varm[:,0]
	Merr1 = dat_varm[:,1]
	Rob1 = dat_varm[:,2]
	Rerr1 = dat_varm[:,3]
	Mth1 = local + 0.
	fM1 = piece_linear(hyper, Mth1, 0.5*np.ones_like(Mth1), n_pop)

	L1 = 0. - 0.5* np.sum( (Mob1-Mth1)**2./Merr1**2. ) \
			- 0.5* np.sum( (Rob1-fM1)**2./(Rerr1**2.+sigma_like_R1**2.) ) \
			- 0.5* np.sum( np.log(Rerr1**2.+sigma_like_R1**2.) )	

	L = L0 + L1
	return L

### mcmc

n_step = int(5e2)
#n_step = int(5e5)

hyper_prob0 = np.loadtxt(args.hyperic)
hyper_stepsize = np.loadtxt(args.hyperstep)
local_prob0 = 0.5 * np.ones(n_varm)
local_stepsize = (1.*1e-4) * np.ones(n_varm)


hyper_prob_chain, hyper_chain, local_prob_chain, local_chain, \
loglike_chain, repeat_chain, stop_step = \
hbm_joint_cdf(hyper_prob0, hyper_stepsize, local_prob0, local_stepsize, n_step,\
			inverse_hyper, inverse_local, \
			loglike_func, data = [dat_fixm, dat_varm], \
			trial_upbound = 20*n_step)

### these should belong to hbm_joint_cdf, just be here temporarily to fix the output
if np.any(loglike_chain == -np.inf):
	stop_step = np.where(loglike_chain == -np.inf)[0][0]
else:
	stop_step = n_step

if repeat_chain[stop_step-1] == 0:
	repeat_chain[stop_step-1] =1

### save
np.savetxt(args.dir+'hyper_prob.out',hyper_prob_chain[:stop_step,:])
np.savetxt(args.dir+'hyper.out',hyper_chain[:stop_step,:])
np.savetxt(args.dir+'loglike.out',loglike_chain[:stop_step])
np.savetxt(args.dir+'repeat.out',repeat_chain[:stop_step])

### save top for restart
top_ind = np.argsort(loglike_chain)[-100:]
np.savetxt(args.dir+'local_prob_top.out',local_prob_chain[top_ind,:])
np.savetxt(args.dir+'local_top.out',local_chain[top_ind,:])
np.savetxt(args.dir+'hyper_prob_top.out',hyper_prob_chain[top_ind,:])
np.savetxt(args.dir+'hyper_top.out',hyper_chain[top_ind,:])
np.savetxt(args.dir+'index_top.out',top_ind)
np.savetxt(args.dir+'loglike_top.out', loglike_chain[top_ind])

### save last for resume
last_ind = stop_step-1
np.savetxt(args.dir+'local_prob_last.out',local_prob_chain[last_ind,:])
np.savetxt(args.dir+'hyper_prob_last.out',hyper_prob_chain[last_ind,:])




