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
n_step = 5000


### generate line
seed = 2357
np.random.seed(seed)

A_real = 3.
B_real = 5.
n_group = 100
n_point = 50
A_scatter = 1e0
B_scatter = 1e0

a_real = np.random.normal( loc=A_real, scale = A_scatter, size=n_group )
b_real = np.random.normal( loc=B_real, scale = B_scatter, size=n_group )

x_real = np.zeros((n_point,n_group))
y_data = np.zeros((n_point,n_group))
y_err = np.zeros((n_point,n_group))
err_size = 1.

for i in range(n_group):
	x_real[:,i] = np.sort(np.random.uniform(-5,5,n_point))
	y_real = model(x_real[:,i],(b_real[i], a_real[i]))
	y_err[:,i] = err_size * np.random.random(n_point)
	y_data[:,i] = y_real + np.random.normal(loc=0., scale=y_err[:,i], size=n_point)



'''
log likelihood depends on the assumption of the data
assuming y_data ~ N(y_real, y_err)
'''
def single_log_likely(para, model_func, x_real, y_data, y_err):
	y_model = model_func(x_real,para)
	return 0.-np.sum( (y_model - y_data)**2./(2.* y_err**2.) )


def log_likely_func(local_para, model, x_real, y_data, y_err):
	n_point,n_group = np.shape(x_real)

	# local_para[n_local,n_group]
	n1,n2 = np.shape(local_para)
	if n1 == n_group: local_para = np.transpose(local_para)

	# assign each set of para to a set of data, sum the log likelihood
	sum_loglike = 0.
	for i in range(n_group):
		sum_loglike = sum_loglike + \
					single_log_likely(local_para[:,i], model, x_real[:,i], y_data[:,i], y_err[:,i])

	return sum_loglike


''' draw local from hyper '''
def draw_local_func(hyper):
	A_real, B_real, A_scatter, B_scatter = hyper
	local = np.random.normal(B_real, B_scatter), \
			np.random.normal(A_real, A_scatter)
	return local


### mcmc
n_hyper = 4 # A_real, B_real, A_scatter, B_scatter
hyper0 = np.array([A_real,B_real,A_scatter,B_scatter]) 
hyper_step = np.array([0.5,0.5,0.,0.]) # randomly selected step size

hyper_seq, local_seq, loglike_seq, repeat, ratio_seq= \
hbm(hyper0, hyper_step, n_step, draw_local_func, n_group, log_likely_func, model, \
data=[x_real,y_data,y_err], seed=2357, domain=[[2,0,np.inf],[3,0,np.inf]] )

#print hyper_seq[0,:]
#print repeat
#print loglike_seq

### plot
row = 2
col = 4

f,((a00,a01,a02,a03),(a10,a11,a12,a13))=plt.subplots(row,col,figsize=(col*5,row*5))
ax = ((a00,a01,a02,a03),(a10,a11,a12,a13))

#for j in range(col):
	#ax[0][j].plot(np.repeat(hyper_seq[j,:],repeat),'b-')

ax[0][0].plot(repeat,'b-')
ax[0][0].set_ylabel('(repeat times)')

for j in range(col):
	ax[1][j].plot(hyper_seq[j,:],'b-')


ax[1][0].set_xlabel('A_real')
ax[1][1].set_xlabel('B_real')
ax[1][2].set_xlabel('A_scatter')
ax[1][3].set_xlabel('B_scatter')

for i_group in range(n_group):
	ax[0][1].plot(x_real[:,i_group],model(x_real[:,i_group],(b_real[i], a_real[i])),'b-')
	ax[0][1].errorbar(x_real[:,i_group],y_data[:,i_group],yerr = y_err[:,i_group],fmt='ro')

ax[0][2].plot(ratio_seq,'b-')
 

plt.savefig('plt_test_multiline.png')
