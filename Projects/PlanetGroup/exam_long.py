import numpy as np
import sys
sys.path.append('../..')
from NaiveMC.efflength import dk_acf
from NaiveMC.burnout import median_burn



''' read iter*.out, thinned, and output for further analysis '''
def read_iter(dir, n_iter=5, thin=10, start_ind=0):

	### save
	hyper_list = []
	loglike_list = []
	total_length = 0

	### read iteratively and thinned
	for i in range(1, n_iter+1):
		hyper_file = dir + 'iter'+str(i)+'_hyper.out'
		log_file = dir + 'iter'+str(i)+'_loglike.out'

		hyper = np.loadtxt(hyper_file)
		loglike = np.loadtxt(log_file)

		last_ind = len(loglike)-1
		total_length = total_length + last_ind

		loglike_thin = loglike[start_ind:last_ind:thin]
		hyper_thin = hyper[start_ind:last_ind:thin, :]

		hyper_list.append(hyper_thin)
		loglike_list.append(loglike_thin)

		# change as the start_ind for the next file
		start_ind = thin - (last_ind - start_ind) % thin

	hypers = np.vstack(( hyper_list[0], hyper_list[1] ))
	loglikes = np.concatenate(( loglike_list[0], loglike_list[1] ))

	for i in range(2, n_iter):
		hypers = np.vstack(( hypers, hyper_list[i] ))	
		loglikes = np.concatenate(( loglikes, loglike_list[i] ))

	print 'total length', total_length+1
	print 'thinned length', len(loglikes)

	return hypers, loglikes

if __name__ == '__main__':

	# input parameters
	dir = 'lin1/'
	n_iter = 3
	thin = 20
	start_ind = 0
	limit = 50

	hypers, loglikes = read_iter(dir, n_iter, thin, start_ind)

	n_burn = median_burn(loglikes)
	eff = dk_acf(hypers, n_burn, limit)

	print 'burn step', n_burn
	print 'effective chain length', eff


	

