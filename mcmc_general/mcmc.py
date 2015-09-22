'''
function:
1. mcmc
2. auto_burn

'''

# import
import numpy as np


''' mcmc'''
def mcmc(p0, p_step, n_step, log_likely_func, log_args=[], *seed):
	# random seed
	if seed: np.random.seed(seed[0])

	# setup parameter chain
	n_para = len(p0)
	p_seq = np.zeros((n_para, n_step))
	p_seq[:,0] = p0

	# advance with p_step and check if accept p_new
	for i_step in range(1,n_step):

		p_old = p_seq[:, i_step-1]
		p_new = p_old + p_step * np.random.uniform(-1,1,n_para)

		delta_log = log_likely_func(p_new, *log_args) - \
					log_likely_func(p_old, *log_args)

		ratio = np.exp(delta_log)
		if (np.random.uniform(0,1) < ratio):
			p_seq[:,i_step] = p_new
		else:
			p_seq[:,i_step] = p_old

	return p_seq



''' auto_burn'''
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

	if burn_step==n_step-1: burn_step= int(n_step * fix_ratio)
	
	return burn_step