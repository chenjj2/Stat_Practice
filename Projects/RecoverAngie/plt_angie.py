import numpy as np
#import matplotlib.pyplot as plt
import corner

datafile = '/Users/jingjing/Work/DK_project/Stat_Practice/Projects/RecoverAngie/test_angie.out'

data = np.loadtxt(datafile)
n_data = int(5e5)
data_burnout = data[n_data/3*2:,:]
data_thin = data_burnout[0::20,:]
data_rearange = np.transpose(np.vstack((data_thin[:,1],data_thin[:,0],data_thin[:,2])))
#c, gamma, sm = data[:,0], data[:,1], data[:,2]

figure = corner.corner(data_rearange, labels=['C',r'$\gamma$',r'$\sigma_m$'],\
						truths=[0.0,0.0,0.0], quantiles=[0.025,0.16,0.5,0.84,0.975]
					)
figure.savefig("plt_angie.png")
