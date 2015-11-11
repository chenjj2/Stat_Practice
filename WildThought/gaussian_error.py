import numpy as np
import matplotlib.pyplot as plt

n_sample = 10000

x = 10.
x_err = x/5.
x_sample = np.random.normal(x,x_err,n_sample)

logx = np.log10(x)
logx_sample = np.log10(x_sample)

plt.hist(x_sample, bins=50)
plt.show()

plt.clf()

plt.hist(logx_sample, bins=50)
plt.show()