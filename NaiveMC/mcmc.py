'''
function:
* domain_pass
* jump
* mcmc
* hbm
	- hbm_initial
	- hbm_likelihood
	- hbm
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
def jump(para_old, para_stepsize, single_jump=False):
	n_para = len(para_old)
	if single_jump:
		ind_para = np.random.choice(n_para,1)
		para_new = para_old + 0.
		para_new[ind_para] = para_new[ind_para] + para_stepsize[ind_para] * np.random.normal(0.,1.)
	else:
		para_new = para_old + para_stepsize * np.random.normal(0.,1.,n_para)
	return para_new


''' mcmc '''
def mcmc(p0, p_step, n_step, log_likely_func, data=[], domain = None, seed=2357, *check_func):
	# set random seed
	np.random.seed(seed)

	# setup parameter chain
	n_para = len(p0)
	p_seq = np.zeros((n_step, n_para))
	p_seq[0,:] = p0

	# setup check function (eg. chi2), otherwise return log likelihood
	if check_func:
		check = check_func[0]
	else:
		check = log_likely_func

	check_seq = np.zeros(n_step)
	check_seq[0] = check(p_seq[0,:],*data)

	# advance with p_step and check if accept p_new
	for i_step in range(1,n_step):

		p_old = p_seq[i_step-1, :] + 0.
		p_new = p_old + p_step * np.random.uniform(-1,1,n_para)
		if domain != None:
			p_new = domain(p_new, p_old)
		else: pass

		delta_log = log_likely_func(p_new, *data) - \
					log_likely_func(p_old, *data)
		# ?? local prior likelihood
		# log(data|local)+log(local|local prior)

		ratio = np.exp(delta_log)
		if (np.random.uniform(0,1) < ratio):
			p_seq[i_step, :] = p_new
		else:
			p_seq[i_step, :] = p_old

		check_seq[i_step] = check(p_seq[i_step, :],*data)

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
			loglike_each[i_draw,i_group] = log_likely_func(local, model, i_group, *data)

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


''' hbm_jump '''
# copy from jump, I am no more using func_mcmc or func_hbm anyway
def hbm_jump(para_old, para_stepsize):
	n_para = len(para_old)
	
	para_new = para_old + para_stepsize * np.random.normal(0.,1.,n_para)

	return para_new


''' hbm_joint '''
def hbm_joint(hyper0, hyper_stepsize, local0, local_stepsize, n_step, \
			hyper_prior, local_given_hyper, data_given_local, data=[], \
			hyper_domain=None, local_domain=None,\
			trial_upbound = 1e5, random_seed = 2357):

	np.random.seed(random_seed)

	### initial setup
	n_hyper = len(hyper0)
	hyper_chain = np.zeros((n_step,n_hyper))
	hyper_chain[0,:] = hyper0

	n_local = len(local0)
	local_chain = np.zeros((n_step,n_local))
	local_chain[0,:] = local0

	loglikelihood_chain = np.repeat(-np.inf,n_step)
	loglikelihood0 = hyper_prior(hyper0) + local_given_hyper(hyper0, local0) + data_given_local(local0, *data)
	loglikelihood_chain[0] = loglikelihood0

	repeat_chain = np.zeros((n_step,), dtype=np.int)
	repeat_chain[n_step-1] = 1

	### first step
	hyper_old = hyper0
	local_old = local0
	loglikelihood_old = loglikelihood0

	### mcmc
	for i_step in range(0, n_step-1):
		if (np.sum(repeat_chain)>=trial_upbound):
			break

		step_stay = True
		while step_stay & (np.sum(repeat_chain)<trial_upbound):
			repeat_chain[i_step] = repeat_chain[i_step]+1

			### jump
			hyper_new = jump(hyper_old, hyper_stepsize)
			local_new = jump(local_old, local_stepsize)

			if hyper_domain == None: pass
			else:
				hyper_new = hyper_domain(hyper_new, hyper_old)
			if local_domain == None: pass
			else:
				local_new = local_domain(local_new, local_old)


			### calculate loglikelihood
			loglikelihood_new = hyper_prior(hyper_new) + local_given_hyper(hyper_new,local_new) + data_given_local(local_new, *data)

			### accept/reject
			ratio = np.exp(loglikelihood_new - loglikelihood_old)
	
			if (np.random.uniform(0,1) < ratio):
				hyper_chain[i_step+1,:] = hyper_new
				hyper_old = hyper_new + 0.

				local_chain[i_step+1,:] = local_new
				local_old = local_new + 0.

				loglikelihood_chain[i_step+1] = loglikelihood_new
				loglikelihood_old = loglikelihood_new + 0.

				step_stay = False

			else: pass

	return hyper_chain, local_chain, loglikelihood_chain, repeat_chain, i_step


''' jump in the probability space '''
def jump_prob(para, stepsize, continuous=False):
	
	n_para = len(para)
	para_new = para + stepsize * np.random.normal(0.,1., n_para)

	if continuous:
		para_new = para_new - np.floor(para_new)		
	else:
		out = (para_new>1.) | (para_new<0.)
		para_new[out] = para[out] + 0.

	return para_new 


''' hbm_joint with jump in the probability space '''
def hbm_joint_cdf(hyper_prob0, hyper_stepsize, local_prob0, local_stepsize, n_step,\
				inverse_hyper, inverse_local, \
				loglike_func, data, \
				trial_upbound = 1e5):

	### initial setup
	n_hyper = len(hyper_prob0)
	hyper_prob_chain = np.zeros((n_step, n_hyper))
	hyper_chain = np.zeros((n_step, n_hyper))
	hyper0 = inverse_hyper(hyper_prob0)
	hyper_prob_chain[0] = hyper_prob0
	hyper_chain[0] = hyper0

	n_local = len(local_prob0)
	local_prob_chain = np.zeros((n_step, n_local))
	local_chain = np.zeros((n_step, n_local))
	local0 = inverse_local(local_prob0, hyper0)
	local_prob_chain[0] = local_prob0
	local_chain[0] = local0

	loglikelihood_chain = np.repeat(-np.inf, n_step)
	#loglikelihood0 = data_given_local(local0, model, *data) + np.sum(np.log(local_prob0))
	loglikelihood0 = loglike_func(hyper0, local0, *data)
	loglikelihood_chain[0] = loglikelihood0

	repeat_chain = np.zeros((n_step,), dtype=np.int)
	repeat_chain[n_step-1] = 1

	
	### first step
	hyper_prob_old, local_prob_old = hyper_prob0, local_prob0
	hyper_old, local_old = hyper0, local0
	loglikelihood_old = loglikelihood0

	
	### mcmc
	for i_step in range(0, n_step-1):
		if (np.sum(repeat_chain)>=trial_upbound):
			break
		
		step_stay = True
		while step_stay & (np.sum(repeat_chain)<trial_upbound):
			repeat_chain[i_step] = repeat_chain[i_step]+1

			## jump in prob-space, inverse cdf
			hyper_prob_new = jump_prob(hyper_prob_old, hyper_stepsize)
			local_prob_new = jump_prob(local_prob_old, local_stepsize)
			hyper_new = inverse_hyper(hyper_prob_new)
			local_new = inverse_local(local_prob_new, hyper_new)

			#loglikelihood_new = data_given_local(local_new, model, *data) + np.sum(np.log(local_prob_new))
			loglikelihood_new = loglike_func(hyper_new, local_new, *data)
	
			### accept/reject		
			ratio = np.exp(loglikelihood_new - loglikelihood_old)

			if (np.random.uniform(0,1)<ratio):
				hyper_prob_chain[i_step+1,:] = hyper_prob_new
				hyper_chain[i_step+1,:] = hyper_new
				hyper_prob_old = hyper_prob_new + 0.
				hyper_old = hyper_new + 0.

				local_prob_chain[i_step+1, :] = local_prob_new
				local_chain[i_step+1,:] = local_new
				local_prob_old = local_prob_new + 0.
				local_old =local_new +0.

				loglikelihood_chain[i_step+1] = loglikelihood_new
				loglikelihood_old = loglikelihood_new + 0.

				step_stay = False

			else: pass
			
	return hyper_prob_chain, hyper_chain, local_prob_chain, local_chain, \
			loglikelihood_chain, repeat_chain, i_step



