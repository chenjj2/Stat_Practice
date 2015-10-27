'''
use 5 (mass, radius) from Marcy 2014 to test the composition project
alpha/beta are the fraction of iron/silicate

(unif) & (log10 sigma ~ unif[-3,0])
=> alpha,beta,sigma
=> alpha_i, beta_i
+ radius_i (<= unif)
-> mass_i
=> radius_obs_i, mass_obs_i
'''


### import 
import numpy as np
import matplotlib.pyplot as plt
from mcmc import hbm_joint_cdf, hbm_joint
from scipy.stats import norm, uniform


### data
data = np.loadtxt('/Users/jingjing/Work/Data/2014-Marcy/2014-Marcy-TestSample.txt', skiprows=2, usecols=(1,2,5,6), delimiter=',')

mass_obs = data[:,0]
mass_err = data[:,1]
rad_obs = data[:,2]
rad_err = data[:,3]

n_group = len(mass_obs)


### radius as a function of mass, alpha, beta
# using nearest neighbor to interpolate 
radius_table = np.loadtxt('/Users/jingjing/Work/Model/Composition_LZeng/Radius.out', delimiter=';', unpack=True)

def rad_function(mass,alpha,beta):
	index = np.round([ mass*4., alpha*100., beta*100. ]).astype(int)
	mass_ind, iron_ind, sili_ind = index[0]-2, index[1], index[2]
	row_ind = mass_ind * 100 + iron_ind
	col_ind = sili_ind
	rad = radius_table[row_ind, col_ind]
	
	return rad


### loglikelihood
def data_given_local( local, mass_obs, mass_err, rad_obs, rad_err ):
	n_group = len(local)/3

	a_local = local[:n_group]
	b_local = local[n_group:2*n_group]
	mass_th = local[2*n_group:]
	rad_th = np.zeros(n_group)
	for i_group in range(n_group):
		rad_th[i_group] = rad_function(mass_th[i_group], a_local[i_group], b_local[i_group])

	loglikelihood = 0. - np.sum( (mass_th-mass_obs)**2./mass_err**2. ) \
					- np.sum( (rad_th-rad_obs)**2./rad_err**2. )
	return loglikelihood


def local_given_hyper(hyper, local):
	alpha, beta, sigma = hyper
	
	n_group = len(local)/3
	a_local, b_local = local[:n_group], local[n_group:2*n_group]

	loglikelihood = 0. - 2.* n_group * np.log(sigma) \
					- np.sum( (a_local-alpha)**2./sigma**2. ) \
					- np.sum( (b_local-beta)**2./sigma**2. )
	return loglikelihood


def hyper_prior(hyper):
	alpha, beta, sigma = hyper
	loglikelihood = 0. - np.log(sigma)

	return loglikelihood


### domain check
def hyper_domain(hyper_tmp, hyper_old):
	hyper_new = hyper_tmp +0.

	alpha, beta, sigma = hyper_tmp
	if (alpha < 0.) or (beta < 0.) or (alpha+beta > 1.):
		hyper_new[0:2] = hyper_old[0:2] + 0.
	if (sigma > 1.) or (sigma < 0.001):
		hyper_new[2] = hyper_old[2] + 0.

	return hyper_new

def local_domain(local_tmp, local_old):
	local_new = local_tmp + 0.
	
	n_group = len(local_tmp)/3
	a_local, b_local = local_tmp[:n_group], local_tmp[n_group:2*n_group]
	for i_group in range(n_group):
		if (a_local[i_group]<0.) or (b_local[i_group]<0.) \
		or (a_local[i_group]+b_local[i_group] > 1.):
			local_new[i_group]  = local_old[i_group] + 0. 
			local_new[n_group+i_group] = local_old[n_group+i_group] +0.

	return local_new


### mcmc
import time
print 'start', time.asctime()

n_step = int(1e5)

hyper0 = np.array([0.3, 0.3, 0.1])
hyper_stepsize = 1e-2 * np.ones(3)
local0 = np.hstack(( 0.3*np.ones(n_group), 0.3*np.ones(n_group), 3. *np.ones(n_group) ))
local_stepsize = np.hstack(( 1e-2 * np.ones(2*n_group), 0.3 * np.ones(n_group) ))


hyper_chain, local_chain, loglikelihood_chain, repeat_chain, stop_step = \
hbm_joint(hyper0, hyper_stepsize, local0, local_stepsize, n_step, \
hyper_prior, local_given_hyper, data_given_local, data=[mass_obs, mass_err, rad_obs, rad_err], \
hyper_domain=hyper_domain, local_domain=local_domain, \
trial_upbound = n_step*10)

print 'end', time.asctime()
			

### plot
row = 2
col = 4

f, ((a00,a01,a02,a03),(a10,a11,a12,a13))=plt.subplots(row,col,figsize=(col*5,row*5))
ax = ((a00,a01,a02,a03),(a10,a11,a12,a13))

ax[0][3].plot(loglikelihood_chain[:stop_step])
ax[0][3].set_xlabel('loglikelihood')

for i in range(3):
	ax[1][i].plot(hyper_chain[:stop_step,i])
ax[1][0].set_xlabel(r'$\alpha$')
ax[1][1].set_xlabel(r'$\beta$')
ax[1][2].set_xlabel(r'$\sigma$')


ax[1][3].plot(repeat_chain[:stop_step])
ax[1][3].set_xlabel('repeat times')
ax[1][3].set_yscale('log')

plt.savefig('Figure/test_marcy'+str(int(time.time()))+'.png')


