import os

total_iter = 20
seed = 2357

hyper_prob_input, local_prob_input, output = ['']*total_iter, ['']*total_iter, ['']*total_iter

dir = 'lin1/'
hyper_prob_input[0] = dir + 't1_s2_hyper_prob_best.out'
local_prob_input[0] = dir + 't1_s2_local_prob_best.out'

for i in range(total_iter):
	output[i] = dir + 'iter' + str(i+1)

for i in range(1,total_iter):
	hyper_prob_input[i] = dir + 'iter' + str(i) + '_hyper_prob_last.out'
	local_prob_input[i] = dir + 'iter' + str(i) + '_local_prob_last.out'

for i in range(total_iter):
	os.system("python straight_long.py "+ str(seed) + " " 
			+ hyper_prob_input[i] + " " 
			+ local_prob_input[i] + " "
			+ output[i] )