'''
fit a piecewise linear line on log(M) VS log(R)
'''


import numpy as np
import matplotlib.pyplot as plt


### unit
msun2mjup = 1047.92612
rsun2rjup = 9.948


### data
dir = '/Users/jingjing/Work/Composition_project/Data/'
outdir = '/Users/jingjing/Work/Composition_project/Output/'

# M (Msun), Merr, R (Rsun), Rerr
st = np.loadtxt(dir+'2010-Torres/binarystar.tsv', skiprows=48, usecols=(4,5,6,7), delimiter=';')
# M (Msun), M+, M-, R (Rsun), R+, R-
st2 = np.loadtxt(dir+'2015-Hatzes/otherstars.csv', skiprows=1, usecols=(1,2,3,4,5,6), delimiter=',')
rowind = (st2[:,1]==st2[:,2]) & (st2[:,4]==st2[:,5])
st2 = st2[rowind,:]
# M (Mjup), M+, M-, R (Rjup), R+, R-
bd = np.loadtxt(dir+'2015-Hatzes/browndwarfs.csv', skiprows=1,usecols=(1,2,3,4,5,6),delimiter=',')
rowind = (bd[:,1]==bd[:,2]) & (bd[:,4]==bd[:,5])
bd = bd[rowind,:]
# M (Mjup), M+, M-, R (Rjup), R+, R-
pl = np.loadtxt(dir+'TEPCat/allplanets.csv', skiprows=1, usecols=(26,27,28,29,30,31), delimiter=',')
rowind = (pl[:,1]==pl[:,2]) & (pl[:,4]==pl[:,5])
pl = pl[rowind,:]

### M VS R plot
plt.errorbar(st[:,0]*msun2mjup, st[:,2]*rsun2rjup, xerr=st[:,1], yerr=st[:,3], fmt='.', alpha=0.5)
plt.errorbar(st2[:,0]*msun2mjup, st2[:,3]*rsun2rjup, xerr=st2[:,1], yerr=st2[:,4], fmt='.', alpha=0.5)
plt.errorbar(bd[:,0], bd[:,3], xerr=bd[:,1], yerr=bd[:,4], fmt='.', alpha=0.5)
plt.errorbar(pl[:,0], pl[:,3], xerr=pl[:,1], yerr=pl[:,4], fmt='.', alpha=0.5)

plt.xscale('log'); plt.yscale('log')
plt.xlim([1e-2,1e4]); plt.ylim([1e-2,1e3])
plt.xlabel('M [Jupiter]'); plt.ylabel('R [Jupiter]')

plt.savefig(outdir+'Figure/test_mr.png')

'''
### linear fit
mass = np.hstack(( st[:,0]*msun2mjup, st2[:,0]*msun2mjup, bd[:,0], pl[:,0] ))
merr = np.hstack(( st[:,1]*msun2mjup, st2[:,1]*msun2mjup, bd[:,1], pl[:,1] ))
radius = np.hstack(( st[:,2]*rsun2rjup, st2[:,3]*rsun2rjup, bd[:,3], pl[:,3] ))
raderr= np.hstack(( st[:,3]*rsun2rjup, st2[:,4]*rsun2rjup, bd[:,4], pl[:,4] ))

# range cut
ind = (radius<1e3) & (radius>1e-2) & (mass>1e-2) & (mass<1e4)
mass = mass[ind]
merr = merr[ind]
radius = radius[ind]
raderr = raderr[ind]

logm = np.log10(mass)
logmerr = merr/mass
logr = np.log10(radius)
logrerr = raderr/radius


### mcmc
from mcmc import mcmc

# fix number of piece
n_piece = 3

# piecewise model
def piece_model(p,x):
	ai = p[:n_piece]
	b1 = p[n_piece]
	xi = p[n_piece+1:]

	bi = np.zeros_like(ai)
	bi[0] = b1
	for i in range(1,n_piece):
		bi[i] = (ai[i-1]-ai[i])*xi[i-1]+bi[i-1] 

	y = np.zeros_like(x)
	
	ind = (x<xi[0])
	y[ind] = ai[0]*x[ind]+bi[0]

	for i in range(1,n_piece-1):
		ind = (x>=xi[i-1])&(x<xi[i])
		y[ind] = ai[i]*x[ind]+bi[i]

	ind = ( x>=xi[-1] )
	y[ind] = ai[-1]*x[ind]+bi[-1]

	return y


# loglike
def loglikefunc(p, x, y, yerr):
	ai = p[:n_piece]
	b1 = p[n_piece]
	xi = p[n_piece+1:]

	bi = np.zeros_like(ai)
	bi[0] = b1
	for i in range(1,n_piece):
		bi[i] = (ai[i-1]-ai[i])*xi[i-1]+bi[i-1]
	
	y_fit = piece_model(p, x)
	loglike = 0.- np.sum( (y_fit-y)**2. / yerr**2. )
	return loglike


# domain
xmin, xmax = np.min(logr), np.max(logr)
def domain(p_tmp, p_old):
	p_new = p_tmp
	if (p_tmp[n_piece+1] < xmin):
		p_new[n_piece+1] = p_old[n_piece+1]
	if (p_tmp[-1] > xmax):
		p_new[-1] = p_old[-1]	
	return p_new


# initial
ai = np.ones(n_piece)
b1 = 0.
xi = (xmax-xmin)/n_piece * np.arange(1,n_piece) + xmin
p0 = np.hstack(( ai, b1, xi ))
astep = 0.03*np.ones_like(ai)
bstep = 0.01
xstep = 0.01*np.ones_like(xi)
p_stepsize = np.hstack(( astep, bstep, xstep ))


n_step = int(1e4)


# run
p_chain, loglike_chain = mcmc(p0, p_stepsize, n_step, loglikefunc, data=[logr, logm, logmerr], domain=domain)

for i in range(2*n_piece):
	plt.plot(p_chain[:,i])
plt.show()

print p_chain[-1,:]

plt.errorbar(logr, logm, yerr=logmerr, fmt='.')
x = logr
y = piece_model(p_chain[-1,:], x)
order = np.argsort(x)
plt.plot(x[order], y[order])
plt.savefig('test.png')
'''