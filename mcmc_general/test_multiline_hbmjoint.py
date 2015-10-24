'''
fit multiple lines with hbm_joint
'''



### import package
import numpy as np
import matplotlib.pyplot as plt
from model_fit import model_poly1 as model
from mcmc import hbm_joint, hbm_joint_cdf
from scipy.stats import norm, uniform
from inverse_sampling import inverse_lognormal


### generate line
seed = 2357
np.random.seed(seed)

hyper_c1 = 3.
hyper_c0 = 5.
n_group = 10
n_point = 1
hyper_sigc1 = 1e0
hyper_sigc0 = 1e0

local_c1 = np.random.normal( loc=hyper_c1, scale = hyper_sigc1, size=n_group )
local_c0 = np.random.normal( loc=hyper_c0, scale = hyper_sigc0, size=n_group )

x_real = np.zeros((n_point,n_group))
y_data = np.zeros((n_point,n_group))
y_err = np.zeros((n_point,n_group))
err_size = 2.

for i in range(n_group):
	x_real[:,i] = np.sort(np.random.uniform(-5,5,n_point))
	y_real = model(x_real[:,i],(local_c0[i], local_c1[i]))
	y_err[:,i] = err_size * np.random.random(n_point)
	y_data[:,i] = y_real + np.random.normal(loc=0., scale=y_err[:,i], size=n_point)



'''
log likelihood depends on the assumption of the data
assuming y_data ~ N(y_real, y_err)
'''
def data_given_local(local, model, x_real, y_data, y_err):
	n_group = len(local)/2
	local_c1 = local[:n_group]
	local_c0 = local[n_group:]

	total_loglikelihood = 0.
	for i_group in range(n_group):
		y_model = model(x_real[:,i_group],(local_c0[i_group], local_c1[i_group]))
		loglikelihood = 0. - np.sum( (y_model - y_data[:,i_group])**2. / (2.*y_err[:,i_group]**2.) )
		total_loglikelihood = total_loglikelihood + loglikelihood
	

	return total_loglikelihood


def local_given_hyper(hyper, local):
	hyper_c1,hyper_c0,hyper_sigc1,hyper_sigc0 = hyper

	n_group = len(local)/2
	local_c1 = local[:n_group]
	local_c0 = local[n_group:]

	total_loglikelihood = n_group * ( np.log(hyper_sigc1) + np.log(hyper_sigc0) ) \
	- np.sum( (local_c1-hyper_c1)**2. / (2.*hyper_sigc1**2.) ) \
	- np.sum( (local_c0-hyper_c0)**2. / (2.*hyper_sigc0**2.) )

	return total_loglikelihood

''' inverse cdf '''
def inverse_hyper(hyper_prob):
	pr_c1, pr_c0, pr_sigc1, pr_sigc0 = hyper_prob

	c1, c0 = uniform.ppf([pr_c1, pr_c0], -10., 30.) # min, range
	sigc1, sigc0 = inverse_lognormal( np.array([pr_sigc1, pr_sigc0]), 0., 0.25)

	hyper = np.array([ c1, c0, sigc1, sigc0 ])
	return hyper

def inverse_local(local_prob, hyper):
	n_group = len(local_prob)/2
	c1 = norm.ppf(local_prob[:n_group], hyper[0], hyper[2] )
	c0 = norm.ppf(local_prob[n_group:], hyper[1], hyper[3] )
	local = np.hstack((c1, c0))
	return local

### mcmc
import time
print 'start:', time.asctime()

n_step = int(1e5)
n_hyper = 4 # hyper_c1, hyper_c0, hyper_sigc1, hyper_sigc0

'''
hyper0 =  np.array([ 2.*hyper_c1, 2.*hyper_c0, hyper_sigc1, hyper_sigc0]) 
local0 =  2.*np.hstack((local_c1,local_c0))
hyper_stepsize = np.array([3e-2, 3e-2, 3e-2, 3e-2])
local_stepsize = 3e-2 * np.ones(2*n_group)

hyper_chain, local_chain, loglikelihood_chain, repeat_chain, stop_step= \
hbm_joint(hyper0, hyper_stepsize, local0, local_stepsize, n_step, \
local_given_hyper, data_given_local, model, data=[x_real,y_data,y_err], \
hyper_domain=[[2,0,np.inf],[3,0,np.inf]], \
trial_upbound = 1e5, random_seed = seed)
'''


hyper_prob0 = np.array([0.3, 0.3, 0.8, 0.8])
local_prob0 = 0.3 * np.ones(2*n_group)
hyper_stepsize = 3e-3 * np.ones(n_hyper)
local_stepsize = 3e-3 * np.ones(2*n_group)

hyper_prob_chain, hyper_chain, local_prob_chain, local_chain, \
loglikelihood_chain, repeat_chain, stop_step = \
hbm_joint_cdf(hyper_prob0, hyper_stepsize, local_prob0, local_stepsize, n_step,\
inverse_hyper, inverse_local, \
data_given_local, model, data=[x_real, y_data, y_err], \
trial_upbound = 1e6, random_seed = seed)



print 'end:', time.asctime()


### plot
row = 2
col = 4

f,((a00,a01,a02,a03),(a10,a11,a12,a13))=plt.subplots(row,col,figsize=(col*5,row*5))
ax = ((a00,a01,a02,a03),(a10,a11,a12,a13))

for i_group in range(n_group):
	ax[0][0].errorbar(x_real[:,i_group],y_data[:,i_group],yerr = y_err[:,i_group],fmt='.')
	ax[0][0].plot(x_real[:,i_group],model(x_real[:,i_group],(local_c0[i_group], local_c1[i_group])),'b-')
ax[0][0].legend(['c0 %.1f' %hyper_c0, 'c1 %.1f' %hyper_c1, 'sig_c0 %.1f' %hyper_sigc0, 'sig_c1 %.1f' %hyper_sigc1],\
loc=0)

ax[0][1].plot(repeat_chain[:stop_step],'b-')
ax[0][1].set_xlabel('repeat times')

delta_log = loglikelihood_chain[1:] - loglikelihood_chain[:-1]
ratio  = np.exp(delta_log)
ratio[np.where(ratio>1)[0]] = 1
ax[0][2].plot(ratio[:stop_step-1], 'b-')
ax[0][2].set_xlabel('ratio')

ax[0][3].plot(loglikelihood_chain[:stop_step],'b-')
ax[0][3].set_xlabel('loglikelihood')


for j in range(col):
	ax[1][j].plot(hyper_chain[:stop_step, j],'b-')


ax[1][0].set_xlabel('hyper_c1')
ax[1][1].set_xlabel('hyper_c0')
ax[1][2].set_xlabel('hyper_sigc1')
ax[1][3].set_xlabel('hyper_sigc0')


plt.savefig('Figure/test_multiline_hbmjoint'+str(int(time.time()))+'.png')
