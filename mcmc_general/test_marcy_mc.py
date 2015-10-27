'''
is this a more natural way to find out the populations and their properties?
simple mcmc on each planet -> a distribution of (alpha, beta) for each planet
combine all the planets' distribution and see if there is some cluster
'''

### import 
import numpy as np
import matplotlib.pyplot as plt
from mcmc import mcmc
### radius as a function of mass, alpha, beta, same as test_marcy.py
from model_fit import radius_lz


### data
data = np.loadtxt('/Users/jingjing/Work/Data/2014-Marcy/2014-Marcy-TestSample.txt', skiprows=2, usecols=(1,2,5,6), delimiter=',')

mass_obs = data[:,0]
mass_err = data[:,1]
rad_obs = data[:,2]
rad_err = data[:,3]

n_planet = len(mass_obs)


### loglikelyhood
def loglikelihood(parameter, mass_obs, mass_err, rad_obs, rad_err):
	alpha, beta, mass_th = parameter
	rad_th = radius_lz(mass_th, alpha, beta)
	
	loglikelihood = 0.- (mass_th-mass_obs)**2./mass_err**2. - (rad_th-rad_obs)**2./rad_err**2.

	return loglikelihood


### parameter domain
def domain(p_tmp, p_old):
	p_new = p_tmp

	alpha, beta, mass = p_tmp
	if (alpha<0.) or (beta<0.) or (alpha+beta>1.):
		p_new[0], p_new[1] = p_old[0], p_old[1]
	if (mass<0.):
		p_new[2] = p_old[2]
	return p_new


### loop mcmc
import time
print 'start', time.asctime()

n_step = int(5e5)
n_para = 3

para_chain = np.zeros((n_step, n_planet*n_para))
loglikelihood_chain = np.zeros((n_step, n_planet))

p0 = np.array([0.3, 0.3, 2. ]) # alpha, beta, mass
p_stepsize = np.array([1e-2, 1e-2, 0.2])

for i in range(n_planet):
	para_chain[:, (i*n_para):((i+1)*n_para)], loglikelihood_chain[:, i] = mcmc( p0, p_stepsize, n_step, loglikelihood, \
									data=[mass_obs[i], mass_err[i], rad_obs[i], rad_err[i]], domain=domain )


print 'end', time.asctime()


### print
np.savetxt('test_marcy_cluster_parameter.out',para_chain)
np.savetxt('test_marcy_cluster_loglike.out',loglikelihood_chain)


### plot
row = 2
col = 4

f, ((a00,a01,a02,a03),(a10,a11,a12,a13))=plt.subplots(row,col,figsize=(col*5,row*5))
ax = ((a00,a01,a02,a03),(a10,a11,a12,a13))

for i in range(n_planet):
	ax[0][0].plot(para_chain[:,i*n_para], alpha=1./(i+1))
	ax[0][1].plot(para_chain[:,i*n_para+1], alpha=1./(i+1))
	ax[0][2].plot(para_chain[:,i*n_para+2], alpha=1./(i+1))
	ax[0][3].plot(loglikelihood_chain, alpha=1./(i+1))

	ax[1][0].plot(para_chain[n_step/2:n_step, i*n_para], para_chain[n_step/2:n_step, i*n_para+1], 'b.', alpha=0.005)

ax[0][0].set_xlabel(r'$\alpha$'); ax[0][0].set_ylim([0.,1.])
ax[0][1].set_xlabel(r'$\beta$'); ax[0][1].set_ylim([0.,1.])
ax[0][2].set_xlabel('mass')
ax[0][3].set_xlabel('loglikelihood')

ax[1][0].set_xlabel(r'$\alpha$'); ax[1][0].set_ylabel(r'$\beta$')
ax[1][0].set_xlim([0.,1.]); ax[1][0].set_ylim([0.,1.])

plt.savefig('Figure/test_marcy_cluster.png')






