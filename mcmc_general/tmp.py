
def jump_prob(para, stepsize, continuous=False):

	para_new = para + np.random.normal(0.,1., stepsize)

	if continuous:
		para_new = para_new - np.floor(para_new)		
	else:
		out = (para_new>1.) | (para_new<0.)
		para_new[out] = para[out] + 0.

	return para_new 



def hbm_joint_cdf(hyper_prob0, hyper_stepsize, local_prob0, local_stepsize, n_step,\
				inverse_hyper, inverse_local, \
				data_given_local, model, data=[], \
				trial_upbound = 1e5, random_seed = 2357):
	
	np.random.seed(random_seed)

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
	loglikelihood0 = data_given_local(local0, model, *data)
	loglikelihood_chain[0] = loglikelihood0

	repeat_chain = np.zeros((n_step,), dtype=np.int)
	repeat_chain[n_step-1] = 1

	
	### first step
	hyper_prob_old, local_prob_old = hyper_prob0, local_prob0
	hyper_old, local_old = hyper0, local0
	lolikelihood_old = loglikelihood0

	
	### mcmc
	for i_step in range(0, n_step-1):
		if (np.sum(repeat_chain)>=trial_upbound):
			break
		
		step_stay = True
		while step_step:
			repeat_chain[i_step] = repeat_chain[i_step]+1

		## jump in prob-space, inverse cdf
		hyper_prob_new = jump_prob(hyper_prob_old, hyper_stepsize)
		local_prob_new = jump_prob(local_prob_old, local_stepsize)
		hyper_new = inverse_hyper(hyper_prob_new)
		local_new = inverse_local(local_prob_new, hyper_new)

		loglikelihood_new = data_given_local(local_new, model, *data)
		
		### accept/reject		
		ratio = np.exp(loglikelihood_new - loglikelihood_old)

		if (np.random.uniform(0,1)<ratio):
			hyper_prob_chain[i_step+1,:] = hyper_prob_new
			hyper_chain[i_step+1,:] = hyper_new
			hyper_prob_old = hyper_prob_new + 0.
			hyper_old = hyper_new + 0.

			local_prob_chain[i_step+1, :] = local_prob_new
			local_chain[i_step+1,:] = local_new
			local_prob_old = hyper_local_new + 0.
			local_old =local_new +0.

			loglikelihood_chain[i_step+1] = loglikelihood_new
			loglikelihood_old = loglikelihood_new + 0.

			step_stay = False

		else: pass
			
	return hyper_prob_chain, hyper_chain, local_prob_chain, local_chain, \
			loglikelihood_chain, repeat_chain, i_step