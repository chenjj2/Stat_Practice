'''
Does HBM with a bunch of data break the degeneracy?
-----------
Problem: With Mass and Radius measurements from some exoplanets, can we use HBM to infer the fraction of iron, silicate, and water, from a 3-layer model.
-----------
Test:
Given A,B,C
Draw (a,b,c)s from A,B,C
Calculate y1=8a+3b+c and y2=a+b+c
Now, given (y1,y2)s
Can we find A,B,C?
'''

### import 
import numpy as np
import matplotlib.pyplot as plt
from mcmc import hbm_joint_cdf
from scipy.stats import norm, uniform

# random seed
seed = 2357
np.random.seed(seed)

# hyper A,B,C
hyper_a = 3.
hyper_b = 2.
hyper_c = 7.

# local a,b,c
n_group = 20
scatter = 0.5
local_a = np.random.normal( loc = hyper_a, scale = scatter, size = n_group)
local_b = np.random.normal( loc = hyper_b, scale = scatter, size = n_group)
local_c = np.random.normal( loc = hyper_c, scale = scatter, size = n_group)

# observed
y1 = 8.*local_a + 3.*local_b + 1.*local_c
y2 = 1.*local_a + 1.*local_b + 1.*local_c

print 'a', local_a
print 'b', local_b
print 'c', local_c
print 'y1', y1
print 'y2', y2


### inverse prior
def inverse_hyper(hyper_prob):
	pr_a, pr_b, pr_c = hyper_prob
	a, b, c = uniform.ppf([pr_a, pr_b, pr_c], 0., 10.)
	
	hyper = np.array([a, b, c])
	return hyper

def inverse_local(local_prob, hyper):
	n_group  = len(local_prob) /3
	a = norm.ppf(local_prob[0:n_group], hyper[0], scatter)
	b = norm.ppf(local_prob[n_group:2*n_group], hyper[1], scatter)
	c = norm.ppf(local_prob[2*n_group:], hyper[2], scatter)
	local = np.hstack((a,b,c))
	return local



### likelihoods
def model(a,b,c):
	y1 = 8.*a + 3.*b + 1.*c
	y2 = 1.*a + 1.*b + 1.*c
	return y1, y2

def data_given_local(local, model, y1, y2):
	n_group = len(local)/3

	local_a = local[0:n_group]
	local_b = local[n_group:2*n_group]
	local_c = local[2*n_group:]

	y1_model, y2_model = model(local_a, local_b, local_c)
	total_loglikelihood = - np.sum(y1-y1_model)**2. - np.sum(y2-y2_model)**2.
	return total_loglikelihood



### mcmc
import time
print 'start:', time.asctime()

n_step = int(1e4)
n_hyper = 3

hyper_prob0 = np.array([0.5, 0.5, 0.5])
local_prob0 = 0.5 * np.ones(3*n_group)
hyper_stepsize = 1e-2 * np.ones(n_hyper)
local_stepsize = 1e-2 * np.ones(3*n_group)

hyper_prob_chain, hyper_chain, local_prob_chain, local_chain, \
loglikelihood_chain, repeat_china, stop_step = \
hbm_joint_cdf( hyper_prob0, hyper_stepsize, local_prob0, local_stepsize, n_step, \
inverse_hyper, inverse_local, \
data_given_local, model, data = [y1, y2], \
trial_upbound = 1e5, random_seed = seed )

print 'end:', time.asctime()


### plot
row = 1
col = 3

f, (a0,a1,a2) = plt.subplots(row, col, figsize=(col*5, row*5))
ax = (a0,a1,a2)

for j in range(3):
	ax[j].plot(hyper_chain[:stop_step, j], 'b-')
	ax[j].set_ylim([0.,10.])

ax[0].set_xlabel('a=%.2f' %hyper_a)
ax[1].set_xlabel('b=%.2f' %hyper_b)
ax[2].set_xlabel('c=%.2f' %hyper_c)

plt.savefig('Figure/test_degen_abc.png')