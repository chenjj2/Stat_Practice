'''
'''


import numpy as np
import matplotlib.pyplot as plt


### unit
msun2mjup = 1047.92612
rsun2rjup = 9.948


### data
dir = '/Users/jingjing/Work/Data/'

# M (Msun), Merr, R (Rsun), Rerr
st = np.loadtxt(dir+'2010-Torres/binarystar.tsv', skiprows=48, usecols=(4,5,6,7), delimiter=';')
# M (Msun), M+, M-, R (Rsun), R+, R-
st2 = np.loadtxt(dir+'2015-Hatzes/otherstars.csv', skiprows=1, usecols=(1,2,3,4,5,6), delimiter=',')
# M (Mjup), M+, M-, R (Rjup), R+, R-
bd = np.loadtxt(dir+'2015-Hatzes/browndwarfs.csv', skiprows=1,usecols=(1,2,3,4,5,6),delimiter=',')
# M (Mjup), M+, M-, R (Rjup), R+, R-
pl = np.loadtxt(dir+'TEPCat/allplanets.csv', skiprows=1, usecols=(26,27,28,29,30,31), delimiter=',')


### M VS R plot
plt.plot(st[:,0]*msun2mjup, st[:,2]*rsun2rjup,'.')
plt.plot(st2[:,0]*msun2mjup, st2[:,3]*rsun2rjup, '.')
plt.plot(bd[:,0], bd[:,3], '.')
plt.plot(pl[:,0], pl[:,3], '.')

plt.xscale('log'); plt.yscale('log')
plt.xlim([1e-2,1e4]); plt.ylim([1e-2,1e3])

plt.savefig('test_mr.png')
