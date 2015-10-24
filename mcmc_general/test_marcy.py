'''
use 5 (mass, radius) from Marcy 2014 to test the composition project
alpha/beta are the fraction of iron/silicate
'''


### import 
import numpy as np
import matplotlib.pyplot as plt
from mcmc import hbm_joint_cdf
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

### prior
def 



### loglikelihood


