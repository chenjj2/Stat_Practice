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
In this case log_likelihood is the same thing as chi2
and we can use chi2 to check if the model is good
'''
def log_likely(para, model_func, x_real, y_data, y_err):
	y_model = model_func(x_real,para)
	return 0.-sum( (y_model - y_data)**2./(2.* y_err**2.) )

def chi2(para, model_func, x_real, y_data, y_err):
	y_model = model_func(x_real,para)
	return sum( (y_model - y_data)**2./(2.* y_err**2.) )


### test with 1st/2nd/3rd order polynomial
p0 = np.zeros(2)
p_step = np.ones(2) * 0.3

poly1_seq, chi1_seq = mcmc(p0, p_step, n_step, log_likely, \
			[model_poly1,x_real,y_data,y_err], seed, chi2)


p0 = np.zeros(3)
p_step = np.ones(3) * 0.3

poly2_seq, chi2_seq = mcmc(p0, p_step, n_step, log_likely, \
			[model_poly2,x_real,y_data,y_err], seed, chi2)


p0 = np.zeros(4)
p_step = np.ones(4) * 0.3

poly3_seq, chi3_seq = mcmc(p0, p_step, n_step, log_likely, \
			[model_poly3,x_real,y_data,y_err], seed, chi2)


### result
burn_step = auto_burn(poly1_seq)
print burn_step

poly1_best = poly1_seq[:,np.argmin(chi1_seq)]
poly2_best = poly2_seq[:,np.argmin(chi2_seq)]
poly3_best = poly3_seq[:,np.argmin(chi3_seq)]
   
print np.min(chi1_seq)/(len(x_real)-2), poly1_best
print np.min(chi2_seq)/(len(x_real)-3), poly2_best
print np.min(chi3_seq)/(len(x_real)-4), poly3_best



### plot
plt.figure(figsize=(15,5))
ax1,ax2,ax3 = plt.subplot2grid((1,3),(0,0)),\
			plt.subplot2grid((1,3),(0,1)),\
			plt.subplot2grid((1,3),(0,2))

ax1.errorbar(x_real,y_data,yerr=y_err,fmt='x')
ax1.plot(x_real,y_real,'b-')
ax1.plot(x_real,model_poly1(x_real,poly1_best),'r--')

ax2.errorbar(x_real,y_data,yerr=y_err,fmt='x')
ax2.plot(x_real,y_real,'b-')
ax2.plot(x_real,model_poly2(x_real,poly2_best),'r--')

ax3.errorbar(x_real,y_data,yerr=y_err,fmt='x')
ax3.plot(x_real,y_real,'b-')
ax3.plot(x_real,model_poly3(x_real,poly3_best),'r--')

plt.savefig('fit_poly123.png')
