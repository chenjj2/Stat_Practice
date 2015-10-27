'''
see if I can reproduce Angie's result with my code
'''


### import 
import numpy as np
import matplotlib.pyplot as plt
from mcmc import hbm_joint_cdf, hbm_joint
from scipy.stats import norm, uniform


### data
data = np.loadtxt('/Users/jingjing/Work/Data/2015-Wolfgang/1504-07557v1/table1_data.txt', skiprows=1, usecols=(2,3,4,5))
mass_ob = data[:,0]
mass_err = data[:,1]
rad_ob = data[:,2]
rad_err = data[:,3]

### select rad_ob < 1.6
'''
ind = rad_ob < 1.6
mass_ob = mass_ob[ind]
mass_err = mass_err[ind]
rad_ob = rad_ob[ind]
rad_err = rad_err[ind]
'''

n_group = len(mass_ob)
print 'n_group', n_group


### inverse sampling
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


### inverse likelihood
def loglike_function(hyper, local, mass_ob, mass_err, rad_ob, rad_err):
	sm = hyper[2]

	n_group = len(local)/2
	rad_th = local[:n_group]
	mass_th = local[n_group:]

	L = 0. - np.sum( (rad_th - rad_ob)**2./rad_err**2. ) \
		- np.sum( (mass_th - mass_ob)**2./(mass_err**2. + sm**2.) ) \
		- 0.5 * np.sum(np.log( mass_err**2.+ sm**2. ))

	return L

### likelihood
'''
def data_given_local( local, mass_ob, mass_err, rad_ob, rad_err ):
	n_group = len(local)/2
	
	rad_th = local[:n_group]
	mass_th = local[n_group:]

	total_loglikelihood = 0. - np.sum( (mass_ob - mass_th)**2./mass_err**2. ) \
						 - np.sum( (rad_ob - rad_th)**2./rad_err**2. )
		
	return total_loglikelihood


def local_given_hyper(hyper,local):
	c, gamma, sm = hyper

	n_group = len(local)/2
	rad_th = local[:n_group]
	mass_th = local[n_group:]

	mu = c * rad_th ** gamma

	total_loglikelihood = 0. - n_group * np.log(sm) - np.sum( (mass_th-mu)**2./sm**2. )

	return total_loglikelihood


def hyper_prior(hyper):
	c, gamma, sm = hyper
	
	total_loglikelihood = 0. - np.log(c) - (gamma-1.)**2. - np.log(sm)

	return total_loglikelihood
'''


### domain check
'''
def hyper_domain(hyper_tmp, hyper_old):
	hyper_new = hyper_tmp + 0.
	if hyper_tmp[0] > np.exp(3.) or hyper_tmp[0] < np.exp(-3.):
		hyper_new[0] = hyper_old[0] + 0.
	if hyper_tmp[2] > 10. or hyper_tmp[2] < 10.**-2.:
		hyper_new[2] = hyper_old[2] + 0.

	return hyper_new

def local_domain(local_tmp, local_old):
	local_new = local_tmp + 0.

	n_group = len(local_tmp)/2

	# radius ~ uniform(0.1, 10)
	for i_group in range(n_group):
		if local_tmp[i_group] > 10. or local_tmp[i_group] < 0.1:
			local_new[i_group] = local_old[i_group] + 0.

	return local_new
'''


### mcmc
import time
print 'start:', time.asctime()

n_step = int(5e5)


### include prior likelihood
'''
hyper0 = np.array([1., 1., 0.01,])
hyper_stepsize = np.array([1e-2, 1e-2, 0.])
local0 = np.hstack(( 2.* np.ones(n_group), 5.* np.ones(n_group) ))
local_stepsize = 1e-2 * np.ones(2*n_group)


hyper_chain, local_chain, loglikelihood_chain, repeat_chain, stop_step = \
hbm_joint(hyper0, hyper_stepsize, local0, local_stepsize, n_step, \
hyper_prior, local_given_hyper, data_given_local, data=[mass_ob, mass_err, rad_ob, rad_err], \
hyper_domain=hyper_domain, local_domain=local_domain, \
trial_upbound = n_step*10)
'''



### inverse sampling
hyper_prob0 = np.array([0.5, 0.5, 0.5])
local_prob0 = 0.5 * np.ones(2*n_group)
hyper_stepsize = np.array([1e-3, 1e-3, 1e-3])
local_stepsize = 1e-3 * np.ones(2*n_group)

hyper_prob_chain, hyper_chain, local_prob_chain, local_chain, \
loglikelihood_chain, repeat_chain, stop_step = \
hbm_joint_cdf(hyper_prob0, hyper_stepsize, local_prob0, local_stepsize, n_step,\
inverse_hyper, inverse_local,\
loglike_function, data=[mass_ob, mass_err, rad_ob, rad_err],\
trial_upbound = n_step*10)


print 'end', time.asctime()

### plot
row = 2
col = 4

f, ((a00,a01,a02,a03),(a10,a11,a12,a13))=plt.subplots(row,col,figsize=(col*5,row*5))
ax = ((a00,a01,a02,a03),(a10,a11,a12,a13))


# hyper VS hyper
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

# local chain
choice = np.random.choice(np.arange(n_group),5)

for i in range(5):
	ax[0][0].plot(local_chain[:stop_step,choice[i]])
	ax[0][1].plot(local_chain[:stop_step,n_group+choice[i]])
ax[0][0].set_xlabel('rad_th')
ax[0][1].set_xlabel('mass_th')


# loglikelihood chain
ax[0][3].plot(loglikelihood_chain[:stop_step])
ax[0][3].set_xlabel('loglikelihood')


# hyper chain
for i in range(3):
	ax[1][i].plot(hyper_chain[:stop_step,i])
ax[1][1].set_xlabel('C')
ax[1][0].set_xlabel(r'$\gamma$')
ax[1][2].set_xlabel(r'$\sigma_M$')


# repeat chain
ax[1][3].plot(repeat_chain[:stop_step])
ax[1][3].set_xlabel('repeat times')
ax[1][3].set_yscale('log')

plt.savefig('Figure/test_angie'+str(int(time.time()))+'.png')


np.savetxt('test_angie.out', hyper_chain)
