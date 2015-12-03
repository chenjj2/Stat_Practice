import numpy as np
import matplotlib.pyplot as plt
#from readdata import data
from func import indicate, convert_data,\
				split_hyper, split_hyper_linear, \
				piece_power, piece_power_frac, piece_linear, piece_linear_complex
from fnmatch import fnmatch
import os

from matplotlib.backends.backend_pdf import PdfPages


### color order: blue, green, red, cyan

def plt_power():
	### data
	hyper = np.loadtxt(dir+'hyper.out')
	loglike = np.loadtxt(dir+'loglike.out')
	repeat = np.loadtxt(dir+'repeat.out')

	### split data
	c0 = hyper[:,0]
	power= hyper[:,1:5]
	sigma = hyper[:,5:9]
	trans = hyper[:,9:12]

	### plot
	plt.clf()

	row = 2
	col = 4

	f, ((a00,a01,a02,a03),(a10,a11,a12,a13))=plt.subplots(row,col,figsize=(col*5,row*5))
	ax = ((a00,a01,a02,a03),(a10,a11,a12,a13))

	# repeat
	ax[0][0].plot(repeat)
	ax[0][0].set_yscale('log')
	ax[0][0].set_xlabel('repeat')

	# loglike
	ax[0][1].plot(loglike)
	ax[0][1].set_xlabel('L')

	# over plot
	dat = data()
	ax[0][2].errorbar(dat[:,0], dat[:,2], xerr=dat[:,1], yerr=dat[:,3], fmt='.')

	hyper_last = hyper[-1,:]
	trans_last = hyper_last[-3:]
	m_sample = np.logspace(np.log10(np.min(dat[:,0])), np.log10(np.max(dat[:,0])), 1000)
	r_sample = piece_power_frac(hyper_last, m_sample, prob_R = 0.5 * np.ones_like(m_sample))
	r_upper = piece_power_frac(hyper_last, m_sample, prob_R = 0.84 * np.ones_like(m_sample))
	r_lower = piece_power_frac(hyper_last, m_sample, prob_R = 0.16 * np.ones_like(m_sample))
	ax[0][2].plot(m_sample,r_sample,'r-')
	ax[0][2].fill_between(m_sample, r_lower, r_upper, color='grey', alpha=0.2)

	r_trans = piece_power_frac(hyper_last, trans_last, prob_R = 0.5 * np.ones_like(trans_last))
	ax[0][2].plot(trans_last, r_trans, 'rx')
	
	ax[0][2].set_xscale('log')
	ax[0][2].set_yscale('log')
	ax[0][2].set_xlabel(r'M [M$_\oplus$]')
	ax[0][2].set_ylabel(r'R [R$_\oplus$]')

	# C
	ax[1][0].plot(c0)
	ax[1][0].set_yscale('log')
	ax[1][0].set_xlabel('c0')
	ax[1][0].set_ylim([1e-3,1e3])

	# power
	for i in range(4):
		ax[1][1].plot(power[:,i])
	ax[1][1].set_xlabel('power')

	# sigma
	for i in range(4):
		ax[1][2].plot(sigma[:,i])
	ax[1][2].set_yscale('log')
	ax[1][2].set_xlabel('sigma')
	ax[1][2].set_ylim([1e-3,1e2])

	# transition
	for i in range(3):
		ax[1][3].plot(trans[:,i])
	ax[1][3].set_yscale('log')
	ax[1][3].set_xlabel('transition')
	ax[1][3].set_ylim([1e-4,1e6])

	plt.savefig('plt_power.png')

	return None

def plt_linear(infile, outfile, datadir=''):

	pp = PdfPages(outfile)

	### data
	hyper = np.loadtxt(infile+'hyper.out')
	loglike = np.loadtxt(infile+'loglike.out')
	repeat = np.loadtxt(infile+'repeat.out')

	### split data
	c0 = hyper[:,0]
	slope= hyper[:,1:5]
	sigma = hyper[:,5:9]
	trans = hyper[:,9:12]

	### plot
	plt.clf()

	row = 2
	col = 4

	f, ((a00,a01,a02,a03),(a10,a11,a12,a13))=plt.subplots(row,col,figsize=(col*5,row*5))
	ax = ((a00,a01,a02,a03),(a10,a11,a12,a13))

	# repeat
	ax[0][0].plot(repeat)
	ax[0][0].set_yscale('log')
	ax[0][0].set_xlabel('repeat')

	# loglike
	ax[0][1].plot(loglike)
	ax[0][1].set_xlabel('L')

	# loglike
	ax[0][2].plot(range(len(loglike)/2, len(loglike)), loglike[len(loglike)/2:])
	ax[0][1].set_xlabel('L')

	# over plot
	dat = convert_data(np.loadtxt(datadir+'PlanetGroup.txt'))
	ax[0][3].errorbar(dat[:,0], dat[:,2], xerr=dat[:,1], yerr=dat[:,3], fmt='.')

	best_ind = np.argmax(loglike)
	hyper_best = hyper[best_ind,:]
	trans_best = hyper_best[-3:]; print 10.**trans_best
	m_sample = np.linspace(np.min(dat[:,0]), np.max(dat[:,0]), 1000)
	r_sample = piece_linear(hyper_best, m_sample, prob_R = 0.5 * np.ones_like(m_sample))
	#r_upper = piece_linear(hyper_best, m_sample, prob_R = 0.84 * np.ones_like(m_sample))
	#r_lower = piece_linear(hyper_best, m_sample, prob_R = 0.16 * np.ones_like(m_sample))
	r_upper = piece_linear_complex(hyper_best, m_sample, prob_R = 0.84 * np.ones_like(m_sample))
	r_lower = piece_linear_complex(hyper_best, m_sample, prob_R = 0.16 * np.ones_like(m_sample))	

	ax[0][3].plot(m_sample,r_sample,'r-')
	ax[0][3].fill_between(m_sample, r_lower, r_upper, color='grey', alpha=0.2)

	r_trans = piece_linear(hyper_best, trans_best, prob_R = 0.5 * np.ones_like(trans_best))
	ax[0][3].plot(trans_best, r_trans, 'rx')
	
	ax[0][3].set_xlabel(r'log10(M [M$_\oplus$])')
	ax[0][3].set_ylabel(r'log10(R [R$_\oplus$])')

	# C
	ax[1][0].plot(c0)
	ax[1][0].set_xlabel('c0')
	ax[1][0].set_ylim([-1,1])

	# slope
	for i in range(4):
		ax[1][1].plot(slope[:,i])
	ax[1][1].set_xlabel('slope')

	# sigma
	for i in range(4):
		ax[1][2].plot(sigma[:,i])
	ax[1][2].set_yscale('log')
	ax[1][2].set_xlabel('sigma')
	ax[1][2].set_ylim([1e-3,1e0])

	# transition
	for i in range(3):
		ax[1][3].plot(trans[:,i])
	ax[1][3].set_xlabel('transition')
	ax[1][3].set_ylim([-4,6])

	pp.savefig()
	pp.close()

	return None

if __name__ == '__main__':
	#plt_power()
	infile = '/Users/jingjing/Work/DK_project/Output/OutFile/PlanetGroup/cf2/'
	outfile = '/Users/jingjing/Work/DK_project/Stat_Practice/Projects/PlanetGroup/linplot/lin_scatter/cf2.pdf'
	datadir = '/Users/jingjing/Work/DK_project/Data/Mine/'
	plt_linear(infile, outfile, datadir)
