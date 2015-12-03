''' auto_burn '''
### pick the first step that p == median(p_chain)

'''
def almost_equal(array,value,tolerance=0.):
	diff = np.abs(array - value)
	return diff<=tolerance

def auto_burn(p_seq, tolerance=[], fix_ratio=0.5):
	n_para, n_step = np.shape(p_seq)

	p_median = np.median(p_seq, axis=1)

	if len(tolerance)!=0:
		t = tolerance
	else:
		p_stddev = np.std(p_seq, axis=1)
		t = p_stddev / n_step

	burn = (n_step-1) * np.ones(n_para)
	for i_para in range(n_para):
		burn_ind = np.where( almost_equal(p_seq[i_para,:],p_median[i_para],t[i_para]) )[0]
		if len(burn_ind)==0: burn[i_para] = n_step-1
		else: burn[i_para] = burn_ind[0]

	burn_step = np.max(burn)

	# if the NO burn in, just use the last half of the chain
	if burn_step==n_step-1: burn_step= int(n_step * fix_ratio)
	
	return burn_step
'''

import numpy as np

def median_burn(loglike):
	median = np.median(loglike)
	burn_step = np.where(loglike>median)[0][0]
	return burn_step