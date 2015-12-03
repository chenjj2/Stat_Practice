'''
exam the chain
1. mixing by checking if the effective length is long enough (> 1000)
2. convergence by checking where is the burnout step
'''


import numpy as np
import sys
sys.path.append('../..')
from NaiveMC.efflength import dk_acf
from NaiveMC.burnout import median_burn


def shortchain(hyper,loglike,limit,short=10):
	nstep = len(loglike)
	nshort = int(nstep/short)
	
	hyper = hyper[:nshort, :]
	loglike = loglike[:nshort]
	
	n_burn = median_burn(loglike)
        eff = dk_acf(hyper, n_burn, limit)

	return n_burn, eff

def thinchain(hyper,loglike,limit,thin=10):
	loglike = loglike[::thin]
	hyper = hyper[::thin,:]

	n_burn = median_burn(loglike)
        eff = dk_acf(hyper, n_burn, limit)	

	return n_burn, eff

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("dir")
	parser.add_argument("file")
	parser.add_argument("limit", type=float)
	args = parser.parse_args()

	dir = args.dir
	file = args.file

	hyper_chain = np.loadtxt(dir+file+'_hyper.out')
	loglike = np.loadtxt(dir+file+'_loglike.out')

	#n_burn = median_burn(loglike)
	#eff = dk_acf(hyper_chain, n_burn, args.limit)

	#print 'burn step', n_burn
	#print 'effective chain length', eff

	#print 'shorter chain', shortchain(hyper_chain, loglike, args.limit)
	#print 'thin=5', thinchain(hyper_chain, loglike, args.limit, thin=5) 
	#print 'thin=10', thinchain(hyper_chain, loglike, args.limit)
	#print 'thin=20', thinchain(hyper_chain, loglike, args.limit, thin=20)
	print 'thin=50', thinchain(hyper_chain, loglike, args.limit, thin=50)
	print 'thin=100', thinchain(hyper_chain, loglike, args.limit, thin=100)	

	

