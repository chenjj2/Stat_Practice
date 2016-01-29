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

###
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("input")
parser.add_argument("top", type=int)
args = parser.parse_args()


### output dir
dir = args.dir

### input hyper prob start position
input = args.input

### which [1-10] top hyper/local
#top = args.top*10-6
top = args.top

### seed
import os
pid = os.getpid()
np.random.seed(pid)
print 'input', input
print 'output', dir
print 'top', top
print 'random seed', pid

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


### inverse sampling
def inverse_hyper(hyper_prob):
	prob_C0, prob_slope, prob_sigma, prob_trans = \
	hyper_prob[0], hyper_prob[1:1+n_pop], hyper_prob[1+n_pop:1+2*n_pop], hyper_prob[1+2*n_pop:3*n_pop]
	
	C0 = uniform.ppf(prob_C0,-1.,2.)
	slope = norm.ppf(prob_slope, 0.,5.)
	sigma = 10.**( uniform.ppf(prob_sigma, -3., 5.) )
	trans = np.sort( uniform.ppf(prob_trans, -4., 10.) ) # sort

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

### change the intrinsic scatter of the degenerate group
def split_group_complex(hyper, local):
	#sig_const_M0, sig_const_M1 = split_group(hyper, local)

	c, slope, sigma, trans = split_hyper_linear(hyper)
	M1 = local[n_fixm:n_fixm+n_varm]

	sig_M0 = np.zeros_like(M0)
        for i in range(n_pop):
                sig_M0 += sigma[i] * indicate(M0,trans,i)

        sig_M1 = np.zeros_like(M1)
        for i in range(n_pop):
                sig_M1 += sigma[i] * indicate(M1,trans,i)

	# fix the intrinsic scatter in the 2nd(0,1,2) group(degenerate group)
	# from constant to a straight line that smoothly goes from sigma_const_giant to sigma_const_degen
	sig_M0 = sig_M0 + (sigma[1]-sigma[2]) * (trans[2] - M0) / (trans[1] - trans[2]) * indicate(M0, trans, 2)
	sig_M1 = sig_M1 + (sigma[1]-sigma[2]) * (trans[2] - M1) / (trans[1] - trans[2]) * indicate(M1, trans, 2)	

	return sig_M0, sig_M1

### likelihood
def loglike_func(hyper,local, dat_fixm, dat_varm):
	sigma_like_R0, sigma_like_R1 = split_group_complex(hyper, local)

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
	return L

### mcmc

n_step = int(1e1)
#n_step = int(5e5)

#hyper_prob0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.6, 0.8])
#hyper_prob0 = np.array([0.5, 0.5, 0.55, 0.5, 0.55, 0.3, 0.5, 0.4, 0.3, 0.45, 0.6, 0.8])
hyper_prob0 = np.loadtxt(input+'hyper_prob_top.out')[0-top,:]
hyper_stepsize = (2.*1e-4) * np.ones(3*n_pop)
local_prob0 = np.loadtxt(input+'local_prob_top.out')[0-top,:]
local_stepsize = (2.*1e-4) * np.ones(n_fixm + 2*n_varm)

np.savetxt(dir+'hyper_prob_start.out', hyper_prob0)
np.savetxt(dir+'local_prob_start.out', local_prob0)

hyper_prob_chain, hyper_chain, local_prob_chain, local_chain, \
loglike_chain, repeat_chain, stop_step = \
hbm_joint_cdf(hyper_prob0, hyper_stepsize, local_prob0, local_stepsize, n_step,\
			inverse_hyper, inverse_local, \
			loglike_func, data = [dat_fixm, dat_varm], \
			trial_upbound = 20*n_step)

### save
np.savetxt(dir+'hyper_prob.out',hyper_prob_chain[:stop_step,:])
np.savetxt(dir+'hyper.out',hyper_chain[:stop_step,:])
np.savetxt(dir+'loglike.out',loglike_chain[:stop_step])
np.savetxt(dir+'repeat.out',repeat_chain[:stop_step])

### save top for restart
top_ind = np.argsort(loglike_chain)[-100:]
np.savetxt(dir+'local_prob_top.out',local_prob_chain[top_ind,:])
np.savetxt(dir+'local_top.out',local_chain[top_ind,:])
np.savetxt(dir+'hyper_prob_top.out',hyper_prob_chain[top_ind,:])
np.savetxt(dir+'hyper_top.out',hyper_chain[top_ind,:])

### save last for resume
last_ind = stop_step-1
np.savetxt(dir+'local_prob_last.out',local_prob_chain[last_ind,:])
np.savetxt(dir+'hyper_prob_last.out',hyper_prob_chain[last_ind,:])




