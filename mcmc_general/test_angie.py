'''
see if I can reproduce Angie's result with my code
'''


### import 
import numpy as np
import matplotlib.pyplot as plt
from mcmc import hbm_joint_cdf
from scipy.stats import norm, uniform


### data
data = np.loadtxt('/Users/jingjing/Work/Data/2015-Wolfgang/1504-07557v1/table1_data.txt', skiprows=1, usecols=(2,3,4,5))
mass_ob = data[:,0]
mass_err = data[:,1]
rad_ob = data[:,2]
rad_err = data[:,3]

n_group = len(mass_ob)


### prior
def inverse_hyper(hyper_prob):
	pr_gamma, pr_lnc, pr_logsm = hyper_prob

	gamma = norm.ppf(pr_gamma, 1., 1.)
	lnc = uniform.ppf(pr_lnc, -3., 6.)
	logsm2 = uniform.ppf(pr_logsm, -4., 6.)

	hyper = np.array([ gamma, np.exp(lnc), (10.**logsm2)**0.5 ])	

	return hyper

def inverse_local(local_prob, hyper):
	n_group = len(local_prob)/2

	# Rad
	rad_th = uniform.ppf(local_prob[:n_group], 0.1, 10.)	

	# Mass
	gamma, c, sm = hyper[0], hyper[1], hyper[2]
	mu = c * rad_th ** gamma
	mass_th = norm.ppf(local_prob[n_group:], mu, sm)

	local = np.hstack((rad_th, mass_th))

	return local

### likelihood
# we actually dont need the model, however its a required parameter
def model(mass_th, rad_th):
	return mass_th, rad_th

def data_given_local(local, model, mass_ob, mass_err, rad_ob, rad_err ):
	n_group = len(local)/2
	
	rad_th = local[:n_group]
	mass_th = local[n_group:]

	total_loglikelihood = 0.
	for i_group in range(n_group):
		loglikelihood = 0. - np.sum( (mass_ob - mass_th)**2./mass_err**2. ) \
						 - np.sum( (rad_ob - rad_th)**2./rad_err**2. )
		total_loglikelihood = total_loglikelihood + loglikelihood

	return total_loglikelihood


### mcmc
import time
print 'start:', time.asctime()

n_step = int(1e5)
n_hyper = 3

hyper_prob0 = np.array([0.5, 0.5, 0.5])
local_prob0 = 0.5 * np.ones(2*n_group)
hyper_stepsize = np.array([1e-2, 1e-2, 1e-2])
local_stepsize = 1e-2 * np.ones(2*n_group)

hyper_prob_chain, hyper_chain, local_prob_chain, local_chain, \
loglikelihood_chain, repeat_chain, stop_step = \
hbm_joint_cdf(hyper_prob0, hyper_stepsize, local_prob0, local_stepsize, n_step,\
inverse_hyper, inverse_local,\
data_given_local, model, data=[mass_ob, mass_err, rad_ob, rad_err],\
trial_upbound = 1e6)

print 'end', time.asctime()
print 'stop_step', stop_step

### plot
row = 2
col = 4

f, ((a00,a01,a02,a03),(a10,a11,a12,a13))=plt.subplots(row,col,figsize=(col*5,row*5))
ax = ((a00,a01,a02,a03),(a10,a11,a12,a13))

'''
ax[0][0].plot(hyper_chain[:stop_step, 1], hyper_chain[:stop_step, 0], '.')
ax[0][0].plot(hyper_chain[stop_step-100:stop_step, 1], hyper_chain[stop_step-100:stop_step, 0], 'r.')
ax[0][0].set_xlabel('C'); ax[0][0].set_ylabel(r'$\gamma$');

ax[0][1].plot(hyper_chain[:stop_step, 1], hyper_chain[:stop_step, 2], '.')
ax[0][1].plot(hyper_chain[stop_step-100:stop_step, 1], hyper_chain[stop_step-100:stop_step, 2], 'r.')
ax[0][1].set_xlabel('C'); ax[0][1].set_ylabel(r'$\sigma_M$')

ax[0][2].plot(hyper_chain[:stop_step, 0], hyper_chain[:stop_step, 2], '.')
ax[0][2].plot(hyper_chain[stop_step-100:stop_step, 0], hyper_chain[stop_step-100:stop_step, 2], 'r.')
ax[0][2].set_xlabel(r'$\gamma$'); ax[0][2].set_ylabel(r'$\sigma_M$')
'''

choice = np.random.choice(np.arange(n_group),5)

for i in range(5):
	ax[0][0].plot(local_chain[:stop_step,choice[i]])
	ax[0][1].plot(local_chain[:stop_step,n_group+choice[i]])
ax[0][0].set_xlabel('rad_th')
ax[0][1].set_xlabel('mass_th')

print rad_ob[choice], rad_err[choice]
print mass_ob[choice], mass_err[choice]


ax[0][3].plot(loglikelihood_chain[:stop_step])
ax[0][3].set_xlabel('loglikelihood')



for i in range(3):
	ax[1][i].plot(hyper_chain[:stop_step,i])
ax[1][0].set_xlabel(r'$\gamma$')
ax[1][1].set_xlabel('C')
ax[1][2].set_xlabel(r'$\sigma_M$')

ax[1][3].plot(repeat_chain[:stop_step])
ax[1][3].set_xlabel('repeat times')
ax[1][3].set_yscale('log')

plt.savefig('Figure/test_angie'+str(int(time.time()))+'.png')
