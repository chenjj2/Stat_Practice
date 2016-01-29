import numpy as np
from scipy.stats import percentileofscore


### best hyper
'''
maxlogs = np.ones(9)
for i in range(9):
	log = np.loadtxt('st'+str(i+1)+'/loglike_top.out')
	maxlogs[i] = log[-1]

maxL = np.max(maxlogs)
print 'max L', maxL
argmaxL = np.argmax(maxlogs)
print 'file', argmaxL

best_hyper = np.loadtxt('st'+str(argmaxL)+'/hyper_top.out')[-1,:]
#np.savetxt('best_hyper.txt', best_hyper)
'''


### thin data
'''
all_hyper = np.loadtxt('h2e0/hyper.out')
for i in range(9):
	hyper = np.loadtxt('h2e'+str(i+1)+'/hyper.out')
	all_hyper = np.vstack((all_hyper, hyper))

### thin all the data for output
thin_hyper = all_hyper[::50,:]
np.savetxt('h2_thin_hyper.out',thin_hyper)
'''

### use all the data to calculate distribution for +/- 34% of best

all_hyper = np.loadtxt('h4_thin_hyper.out')
#best_hyper = np.loadtxt('best_hyper.txt')
best_hyper = np.loadtxt('h4_spatial_median.txt', delimiter=',')

up_hyper = np.zeros(12)
down_hyper = np.zeros(12)

for i in range(12):
	quantile_mid = percentileofscore(all_hyper[:,i], best_hyper[i])
	print i, 'best quantile', repr(quantile_mid)

	up_hyper[i] = np.percentile(all_hyper[:,i], np.min([quantile_mid + 34., 100.]), interpolation='nearest')
	down_hyper[i] = np.percentile(all_hyper[:,i], np.max([quantile_mid - 34., 0.]), interpolation='nearest')

print '------------hyper corresponding to max L and its +/- 34%'
print 'hyper best', repr(best_hyper)
print 'hyper up', repr(up_hyper-best_hyper)
print 'hyper down', repr(best_hyper-down_hyper)

### trans 
#best_trans = best_hyper[-3:]
#up_trans = up_hyper[-3:]
#down_trans = down_hyper[-3:]
#print 'trans best', repr(best_trans)
#print 'trans up/down', repr(up_trans), repr(down_trans)

### median and +/- 34%

median_hyper = np.median(all_hyper,axis=0)
medup_hyper = np.percentile(all_hyper, 84., interpolation='nearest', axis=0)
meddown_hyper = np.percentile(all_hyper, 16., interpolation='nearest', axis=0)

print '------------simply 16%, 50%, 84%'
print 'median', repr(median_hyper)
print '16%', repr(meddown_hyper)
print '84%', repr(medup_hyper)
