'''
This script does the specific task of fitting a line with randomly generated data using MCMC.
'''


### import package
import numpy as np
import matplotlib.pyplot as plt
import emcee


### input parameter
np.random.seed(123)
n_step = 5000
a_0 = 0.; b_0 = 0.; p_0 = (a_0, b_0)


### generate line
def model(a,b,x):
	return a*x+b

a_real = 3.
b_real = 5.

x_real = np.arange(10)
y_real = model(a_real, b_real, x_real)

err_size = 3.
y_err = err_size * np.random.random(len(x_real))
y_shift = np.random.normal(loc=0., scale=y_err, size=len(x_real))
y_data = y_real + y_shift


### likelihood
def log_likely(p,x_real,y_data,y_err):
	a,b=p
	#print 0.-sum( (model(a,b,x_real)-y_data)**2./(2.*y_err**2.) )
	return 0.-sum( (model(a,b,x_real)-y_data)**2./(2.*y_err**2.) )


### MCMC from scratch
def mcmc(x_real, y_data, y_err, a_0=a_0, b_0=b_0, a_step=0.3, b_step=0.3, n_step=n_step):
	a_seq = np.zeros(n_step); b_seq = np.zeros(n_step)	
	a_seq[0] = a_0; b_seq[0] = b_0
	
	for i_step in range(1,n_step):
		a_old = a_seq[i_step-1]; b_old = b_seq[i_step-1]
		a_new = a_old + a_step * np.random.uniform(-1,1)
		b_new = b_old + b_step * np.random.uniform(-1,1)
		
		delta_log = log_likely((a_new, b_new), x_real, y_data, y_err) - \
				log_likely((a_old, b_old), x_real, y_data, y_err)
				
		ratio = np.exp(delta_log)
		if (np.random.uniform(0,1) < ratio):
			a_seq[i_step] = a_new; b_seq[i_step] = b_new
		else:
			a_seq[i_step] = a_old; b_seq[i_step] = b_old
			
	return a_seq, b_seq
	

### run
a_seq, b_seq = mcmc(x_real, y_data, y_err)

burn = n_step/2
a_est, b_est = np.mean(a_seq[burn:]), np.mean(b_seq[burn:])


### MCMC with emcee package
ndim, nwalker = 2, 10
walker_scatter = [1., 1.]

p0 = np.array([p_0 + walker_scatter * np.random.normal(size=ndim) for i in xrange(nwalker)])
sampler = emcee.EnsembleSampler(nwalker, ndim, log_likely, args=[x_real,y_data,y_err])
sampler.run_mcmc(p0, n_step/nwalker)


### plot
plt.figure(figsize=(20,10))
# MCMC from scatch
ax00 = plt.subplot2grid((2,4),(0,0))
ax01 = plt.subplot2grid((2,4),(0,1))
ax02 = plt.subplot2grid((2,4),(0,2))
ax03 = plt.subplot2grid((2,4),(0,3))

ax00.plot(a_seq,'-')

ax01.plot(b_seq,'-')

ax02.plot(a_seq,b_seq,'.')
ax02.plot(a_seq[burn:],b_seq[burn:],'r.')
ax02.set_xlim([min(a_seq[burn:]),max(a_seq[burn:])])
ax02.set_ylim([min(b_seq[burn:]),max(b_seq[burn:])])

ax03.errorbar(x_real, y_data, yerr=y_err, fmt='x')
ax03.plot(x_real,y_real,'b-')
ax03.plot(x_real,model(a_est,b_est,x_real),'r--')

# MCMC with emcee
ax10 = plt.subplot2grid((2,4),(1,0))
ax11 = plt.subplot2grid((2,4),(1,1))
ax12 = plt.subplot2grid((2,4),(1,2))
ax13 = plt.subplot2grid((2,4),(1,3))

ax10.plot(sampler.flatchain[:,0],'-')

ax11.plot(sampler.flatchain[:,1],'-')

ax12.plot(sampler.chain[:,burn/nwalker:,0],sampler.chain[:,burn/nwalker:,1],'r.')

ax13.errorbar(x_real, y_data, yerr=y_err, fmt='x')
ax13.plot(x_real,y_real,'b-')
ax13.plot(x_real,model(np.mean(sampler.chain[:,burn/nwalker:,0]), np.mean(sampler.chain[:,burn/nwalker:,1]), x_real),'r--')

plt.savefig('plot_mcmc.png')
