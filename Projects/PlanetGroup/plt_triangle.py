import numpy as np
#import matplotlib.pyplot as plt
import corner

datafile = '/Users/jingjing/Work/DK_project/Output/OutFile/PlanetGroup/straight/str_hyper.out'

data = np.loadtxt(datafile)[:,-3:]
n_data = int(5e5)
data_burnout = data[n_data/2:,:]
data_thin = data_burnout[0::25,:]

figure = corner.corner(data_thin, labels=[r'transition 1',r'transition 2',r'transition 3'],\
						truths=[0.0,0.0,0.0], quantiles=[0.025,0.16,0.5,0.84,0.975]
					)
figure.savefig("lin_plot/tri_trans.png")