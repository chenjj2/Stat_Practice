'''
translate from DK's fortran script to calculate effective lenght of mcmc chain
'''

import numpy as np
# http://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
# calculate the statistical correlation for a lag of t:
def autocorr(x, t=1):
	return np.corrcoef(np.array([x[:-t], x[t:]]))[0,1]


def dk_acf(chain, nburn, limit=0.9e3):
	nlength, npara = np.shape(chain)
	max_lag = int( (nlength-nburn)/limit )

	corrlen = max_lag * np.ones(npara)

	# go throught each parameter
	for ipara in range(npara):
		for lag in range(1, max_lag):
			acf = autocorr(chain[nburn:,ipara],lag)
			if abs(acf) < 0.5:
				corrlen[ipara] = lag
				break
	
	eff = (nlength-nburn)/corrlen

	return eff


'''http://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introbayes_sect008.htm'''
'''effective sample size '''
'''
def ess(chain, nburn, limit=0.01):
	nlength, npara = np.shape(chain)
	

	return eff
'''
# dk_acf is definitely giving a longer eff than ess
	
