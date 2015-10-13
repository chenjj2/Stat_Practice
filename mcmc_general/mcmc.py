'''
function:
* domain_pass
* jump
* mcmc
* hbm
	- hbm_initial
	- hbm_likelihood
	- hbm
* auto_burn

'''

# import
import numpy as np


''' domain_pass ''' 
# incase the parameter jumps outside the domain
def domain_pass(para, domain):

	n_require = len(domain)
	for i in range(n_require):
		require = domain[i]
		target = para[require[0]]
		if not (target>require[1]) & (target<require[2]): return False
	
	return True


''' mcmc jump '''
def jump(para_old, para_stepsize, single_jump):
	n_para = len(para_old)
	if single_jump:
		ind_para = np.random.choice(n_para,1)
		para_new = para_old + 0.
		para_new[ind_para] = para_new[ind_para] + para_stepsize[ind_para] * np.random.normal(0.,1.)
	else:
		para_new = para_old + para_stepsize * np.random.normal(0.,1.,n_para)
	return para_new


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

		p_old = p_seq[:, i_step-1]+0.
		p_new = p_old + p_step * np.random.uniform(-1,1,n_para)

		delta_log = log_likely_func(p_new, *data) - \
					log_likely_func(p_old, *data)
		# ?? local prior likelihood
		# log(data|local)+log(local|local prior)

		ratio = np.exp(delta_log)
		if (np.random.uniform(0,1) < ratio):
			p_seq[:,i_step] = p_new
		else:
			p_seq[:,i_step] = p_old

		check_seq[i_step] = check(p_seq[:,i_step],*data)

	return p_seq, check_seq


''' hbm initial (hierarchical bayesian model) '''
def hbm_initial(hyper0, n_step):

	n_hyper = len(hyper0)
	hyper = np.zeros((n_hyper,n_step))
	hyper[:,0] = hyper0

	repeat = np.zeros((n_step,), dtype=np.int)
	repeat[n_step-1] = 1

	loglike = np.repeat(-np.inf, n_step)

	return hyper, loglike, repeat


''' hbm p(data|hyper)'''
# sum of log average likelihood
# DONT log{mean[exp(loglikelihood)]}, causing overflow and making comparison impossible
def hbm_likelihood(hyper,draw_local_func,draw_times,n_group,log_likely_func,model,data=[]):
	n_local = len(draw_local_func(hyper))
	loglike_each = np.zeros((draw_times, n_group))
	for i_group in range(n_group):
		for i_draw in range(draw_times):
			local = draw_local_func(hyper)
			loglike_each[i_draw,i_group] = log_likely_func(local, model, *data)

	if draw_times==1:
		total_loglike = np.sum(loglike_each)
	else:
		maximum = np.max(loglike_each,axis=0)
		delta = loglike_each-maximum
		mean_ratio_of_maximum = np.mean(np.exp(delta),axis=0)
	
		loglikelihood = maximum + np.log(mean_ratio_of_maximum)
		total_loglike = np.sum(loglikelihood)

	return total_loglike


''' hbm with mcmc '''
def hbm(hyper0, hyper_stepsize, n_step, draw_local_func, n_group, log_likely_func, model, \
data=[], seed=2357, domain=[], draw_times=1, single_jump = False, trial_upbound = 1e5):

	np.random.seed(seed)

	### initial setup: hyper, local, loglike
	hyper_chain, loglike, repeat = hbm_initial(hyper0, n_step)
	loglike0 = hbm_likelihood(hyper0,draw_local_func,draw_times,n_group,log_likely_func,model,data)
	loglike[0] = loglike0	

	loglike_old = loglike0
	hyper_old = hyper0


	### run mcmc: hyper walks, generates local, calculate delta_log, accept/reject
	for i_step in range(0, n_step-1):
				
		if (np.sum(repeat)>=trial_upbound):
			break

		step_stay = True
		while step_stay & (np.sum(repeat)<trial_upbound):
			repeat[i_step] = repeat[i_step]+1

			hyper_new = jump(hyper_old, hyper_stepsize, single_jump)

			# reject new step if out of domain
			if len(domain) != 0:
				if not domain_pass(hyper_new,domain): 
					continue
			
			loglike_new = hbm_likelihood(hyper_new,draw_local_func,draw_times,n_group,log_likely_func,model,data)
			
			ratio_tmp = np.exp(loglike_new - loglike_old)
	
			# accept new step
			if (np.random.uniform(0,1) < ratio_tmp):
				hyper_chain[:,i_step+1] = hyper_new
				hyper_old = hyper_new + 0.
				loglike[i_step+1] = loglike_new
				loglike_old = loglike_new + 0.
				step_stay = False

			# reject new step
			else: pass

	return hyper_chain, loglike, repeat, i_step



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


