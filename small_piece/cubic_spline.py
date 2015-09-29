'''
write an algorithm to do cubic spline
refer to cubicsplines.pdf
'''

import numpy as np
import matplotlib.pyplot as plt

###
seed = 2357
np.random.seed(seed)

### generate x,y
x = np.arange(10)

### define your own function
def real_func(x):
	return np.sin(x)

y = real_func(x)


### input x_test, the place where you want to know y from interpolation
x_test = np.linspace(0.1,8.9,150)


### cubic spline
# interpolation function
def inter_one(x,t1,t2,y1,y2,z1,z2):
	h = t2-t1
	y = (z2/6./h) * (x-t1)**3. + (z1/6./h) * (t2-x)**3. +\
		(y2/h - z2/6.*h) * (x-t1) + (y1/h - h/6.*z1) * (t2-x)
	return y

# solve coefficients
def cubic_spline(x,y):
	n = len(x)
	h = x[1:] - x[:-1]
	b = (y[1:] - y[:-1])/h
	v = 2.* (h[1:] + h[:-1])
	u = 6.* (b[1:] - b[:-1])
	z = np.zeros(n); z[0] = 0.; z[n-1] = 0.

	C = np.diag(v,0)+np.diag(h[1:-1],1)+np.diag(h[1:-1],-1)

	z[1:-1] = np.linalg.solve(C,u)
	print z
	
	def interpolate(x_test):
		y_test = np.zeros_like(x_test)
		for i,ax in enumerate(x_test):
			if (ax>=x[-1]) or (ax<=x[0]):
				print ax,'is out of fitting bound'
				return 0.
			j = np.where(ax<x)[0][0]
			y_test[i] = inter_one(ax,x[j-1],x[j],y[j-1],y[j],z[j-1],z[j])  
		return y_test

	return interpolate
	

###
inter = cubic_spline(x,y)
y_test = inter(x_test)


### plot
plt.figure(figsize=(8,6))
data, = plt.plot(x,y,'bo')
fit, = plt.plot(x_test,y_test,'r--')

all_x = np.sort(np.concatenate((x,x_test)))
real, = plt.plot(all_x,real_func(all_x),'k--')

plt.legend([data,fit,real],['data','fit','real'], loc=0)

plt.savefig('plt_cubic_spline.png')



