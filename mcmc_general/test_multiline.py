'''
first HBM based on test_line
fit multiple lines this time
'''



### import package
import numpy as np
import matplotlib.pyplot as plt
from model_fit import model_poly1 as model
from mcmc import hbm


### input for mcmc
n_step = 50


### generate line
seed = 2357
np.random.seed(seed)

hyper_c1 = 3.
hyper_c0 = 5.
n_group = 3
n_point = 50
hyper_sigc1 = 1e-1
hyper_sigc0 = 1e-1

local_c1 = np.random.normal( loc=hyper_c1, scale = hyper_sigc1, size=n_group )
local_c0 = np.random.normal( loc=hyper_c0, scale = hyper_sigc0, size=n_group )

x_real = np.zeros((n_point,n_group))
y_data = np.zeros((n_point,n_group))
y_err = np.zeros((n_point,n_group))
err_size = 3.

for i in range(n_group):
	x_real[:,i] = np.sort(np.random.uniform(-5,5,n_point))
	y_real = model(x_real[:,i],(local_c0[i], local_c1[i]))
	y_err[:,i] = err_size * np.random.random(n_point)
	y_data[:,i] = y_real + np.random.normal(loc=0., scale=y_err[:,i], size=n_point)



'''
log likelihood depends on the assumption of the data
assuming y_data ~ N(y_real, y_err)
'''
def single_log_likely(para, model_func, x_real, y_data, y_err):
	y_model = model_func(x_real,para)
	return 0.-np.sum( (y_model - y_data)**2./(2.* y_err**2.) )



''' draw local from hyper '''
def draw_local_func(hyper):
	hyper_c1, hyper_c0, hyper_sigc1, hyper_sigc0 = hyper
	local = np.random.normal(hyper_c0, hyper_sigc0), \
			np.random.normal(hyper_c1, hyper_sigc1)
	return local


### mcmc
n_hyper = 4 # hyper_c1, hyper_c0, hyper_sigc1, hyper_sigc0
hyper0 = np.array([hyper_c1,hyper_c0,hyper_sigc1,hyper_sigc0]) 
hyper_step = np.array([1e-2,1e-2,1e-2,1e-2]) # randomly selected step size

hyper_seq, loglike_seq, repeat_seq = \
hbm(hyper0, hyper_step, n_step, draw_local_func, n_group, single_log_likely, model, \
data=[x_real,y_data,y_err], seed=2357, domain=[[2,0,np.inf],[3,0,np.inf]], \
draw_times=100, single_jump = True, trial_upbound = 10000 )

print hyper_seq[0,:]


### plot
row = 2
col = 4

f,((a00,a01,a02,a03),(a10,a11,a12,a13))=plt.subplots(row,col,figsize=(col*5,row*5))
ax = ((a00,a01,a02,a03),(a10,a11,a12,a13))

#for j in range(col):
	#ax[0][j].plot(np.repeat(hyper_seq[j,:],repeat),'b-')


for i_group in range(n_group):
	ax[0][0].errorbar(x_real[:,i_group],y_data[:,i_group],yerr = y_err[:,i_group],fmt='r.')
	ax[0][0].plot(x_real[:,i_group],model(x_real[:,i_group],(local_c0[i], local_c1[i])),'b-')
ax[0][0].legend(['c0 %.1f' %hyper_c0, 'c1 %.1f' %hyper_c1, 'sig_c0 %.1f' %hyper_sigc0, 'sig_c1 %.1f' %hyper_sigc1],\
loc=0)

ax[0][1].plot(repeat_seq,'b-')
ax[0][1].set_ylabel('repeat times')

delta_log = loglike_seq[1:] - loglike_seq[:-1]
ratio  = np.exp(delta_log)
ratio[np.where(ratio>1)[0]] = 1
ax[0][2].plot(ratio, 'b-')
ax[0][2].set_ylabel('ratio')


for j in range(col):
	ax[1][j].plot(hyper_seq[j,:],'b-')


ax[1][0].set_xlabel('hyper_c1')
ax[1][1].set_xlabel('hyper_c0')
ax[1][2].set_xlabel('hyper_sigc1')
ax[1][3].set_xlabel('hyper_sigc0')


plt.savefig('plt_test_multiline.png')
