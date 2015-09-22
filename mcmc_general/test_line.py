'''
fitting a simulated line with error in the y axis.
test with the right model and overfitting models.
'''



### import package
import numpy as np
import matplotlib.pyplot as plt
from model_fit import model_poly1, model_poly2, model_poly3
from mcmc import mcmc, auto_burn


### input for mcmc
n_step = 5000


### generate line
seed = 2357
np.random.seed(seed)

a_real = 3.
b_real = 5.

x_real = np.arange(10)
y_real = model_poly1(x_real,(b_real, a_real))

err_size = 3.
y_err = err_size * np.random.random(len(x_real))
y_shift = np.random.normal(loc=0., scale=y_err, size=len(x_real))
y_data = y_real + y_shift


'''
log likelihood depends on the assumption of the data
assuming y_data ~ N(y_real, y_err)
'''
def log_likely(para, model_func, x_real, y_data, y_err):
	y_model = model_func(x_real,para)
	return 0.-sum( (y_model - y_data)**2./(2.* y_err**2.) )


### test with 1st/2nd/3rd order polynomial
p0 = np.zeros(2)
p_step = np.ones(2) * 0.3

poly1_seq = mcmc(p0, p_step, n_step, log_likely, \
			[model_poly1,x_real,y_data,y_err],seed)


p0 = np.zeros(3)
p_step = np.ones(3) * 0.3

poly2_seq = mcmc(p0, p_step, n_step, log_likely, \
			log_args=[model_poly2,x_real,y_data,y_err])


p0 = np.zeros(4)
p_step = np.ones(4) * 0.3

poly3_seq = mcmc(p0, p_step, n_step, log_likely, \
			log_args=[model_poly3,x_real,y_data,y_err])


### result
burn_step = auto_burn(poly1_seq)
#print burn_step


poly1_est = np.median(poly1_seq[:, burn_step:],axis=1)
#print poly1_est

poly2_est = np.median(poly2_seq[:, burn_step:],axis=1)
#print poly2_est

poly3_est = np.median(poly3_seq[:, burn_step:],axis=1)
#print poly3_est



### plot
plt.figure(figsize=(15,5))
ax1,ax2,ax3 = plt.subplot2grid((1,3),(0,0)),\
			plt.subplot2grid((1,3),(0,1)),\
			plt.subplot2grid((1,3),(0,2))

ax1.errorbar(x_real,y_data,yerr=y_err,fmt='x')
ax1.plot(x_real,y_real,'b-')
ax1.plot(x_real,model_poly1(x_real,poly1_est),'r--')

ax2.errorbar(x_real,y_data,yerr=y_err,fmt='x')
ax2.plot(x_real,y_real,'b-')
ax2.plot(x_real,model_poly2(x_real,poly2_est),'r--')

ax3.errorbar(x_real,y_data,yerr=y_err,fmt='x')
ax3.plot(x_real,y_real,'b-')
ax3.plot(x_real,model_poly3(x_real,poly3_est),'r--')

plt.savefig('fit_poly123.png')
