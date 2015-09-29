'''
function:
1. mcmc
2. auto_burn

'''

# import
import numpy as np


''' mcmc '''
def mcmc(p0, p_step, n_step, log_likely_func, data=[], seed=2357, *check_func):
	# set random seed
	np.random.seed(seed)

	# setup parameter chain
	n_para = len(p0)
	p_seq = np.zeros((n_para, n_step))
	p_seq[:,0] = p0

	# setup check function (eg. chi2), otherwise return log likelihood
	if check_func:
		check = check_func[0]
	else:
		check = log_likely_func

	check_seq = np.zeros(n_step)
	check_seq[0] = check(p_seq[:,0],*data)

	# advance with p_step and check if accept p_new
	for i_step in range(1,n_step):

		p_old = p_seq[:, i_step-1]
		p_new = p_old + p_step * np.random.uniform(-1,1,n_para)

		delta_log = log_likely_func(p_new, *data) - \
					log_likely_func(p_old, *data)

		ratio = np.exp(delta_log)
		if (np.random.uniform(0,1) < ratio):
			p_seq[:,i_step] = p_new
		else:
			p_seq[:,i_step] = p_old

		check_seq[i_step] = check(p_seq[:,i_step],*data)

	return p_seq, check_seq


''' hbm (hierarchical bayesian model) '''
def hbm(hyper0, hyper_step, n_step, draw_local_func, n_group, log_likely_func, model, data=[], seed=2357):

	np.random.seed(seed)

	# initial setup: hyper, local, loglike
	n_hyper = len(hyper0)
	hyper = np.zeros((n_hyper,n_step))
	hyper[:,0] = hyper0

	n_local = len(draw_local_func(hyper[:,0]))
	local = np.zeros((n_local,n_group,n_step))
	for i_group in range(n_group):
		local[:,i_group,0] = draw_local_func(hyper[:,0])

	loglike = np.zeros(n_step)
	loglike[0] = log_likely_func(local[:,:,0], model, *data)

	# run mcmc: hyper walks, generates local, calculate delta_log, accept/reject
	for i_step in range(1,n_step):
		hyper_old = hyper[:, i_step-1]
		hyper_new = hyper_old + hyper_step * np.random.uniform(-1,1,n_hyper)

		local_old = local[:,:,i_step-1]
		local_new = np.zeros((n_local,n_group))
		for i_group in range(n_group):
			local_new[:,i_group] = draw_local_func(hyper_new)

		delta_log = log_likely_func(local_new, model, *data) - \
					log_likely_func(local_old, model, *data)

		ratio = np.exp(delta_log)
		if (np.random.uniform(0,1) < ratio):
			hyper[:,i_step] = hyper_new
			local[:,:,i_step] = local_new
		else:
			hyper[:,i_step] = hyper_old
			local[:,:,i_step] = local_old

		loglike[i_step] = log_likely_func(local[:,:,i_step], model, *data)

	return hyper, local, loglike



''' auto_burn '''
### pick the first step that p == median(p_chain)

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


