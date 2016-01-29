'''
make a plot for Mstat2R & Rstat2M
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from func import Mstat2R, Rstat2M, Mpost2R, Rpost2M

### set plot 
fig = plt.figure()
gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, :-1])
ax2 = plt.subplot(gs[1:,:-1])
ax3 = plt.subplot(gs[1:, -1])


### m2r
nsample = 5e2
m2r_m = np.log10(np.random.normal(2,0.5,size=nsample))
m2r_r = np.log10(Mpost2R(10.**m2r_m))
m2r_m2 = np.log10(Rpost2M(10.**m2r_r))

### r2m
r2m_r = np.log10(np.random.normal(10,0.5,size=nsample))
r2m_m = np.log10(Rpost2M(10.**r2m_r))
r2m_r2 = np.log10(Mpost2R(10.**r2m_m))

### mass histogram
nbin=10
ax1.hist(m2r_m2, fc='r', ec='r', alpha=0.3, bins=nbin)
ax1.hist(m2r_m,fc='b', bins=nbin, label=['M -> R -> M'])
ax1.hist(r2m_m,fc='g', bins=nbin, label=['R -> M'])
ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

### model
from func import piece_linear, piece_linear_complex
best_hyper = np.loadtxt('spatial_median.txt', delimiter=',')

m_sample = np.linspace( -3.9, 5.5, 1000 )
r_sample = piece_linear(best_hyper, m_sample, prob_R = 0.5*np.ones_like(m_sample))
r_upper = piece_linear_complex(best_hyper, m_sample, prob_R = 0.84 * np.ones_like(m_sample))
r_lower = piece_linear_complex(best_hyper, m_sample, prob_R = 0.16 * np.ones_like(m_sample))
r_2upper = piece_linear_complex(best_hyper, m_sample, prob_R = 0.975 * np.ones_like(m_sample))
r_2lower = piece_linear_complex(best_hyper, m_sample, prob_R = 0.025 * np.ones_like(m_sample))

ax2.plot(m_sample, r_sample, 'r-')
ax2.fill_between(m_sample, r_lower, r_upper, color='grey', alpha=0.6)
ax2.fill_between(m_sample, r_2lower, r_2upper, color='grey', alpha=0.4)

ax2.set_xlabel(r'$\rm log_{10}\ M/M_\oplus$')
ax2.set_ylabel(r'$\rm log_{10}\ R/R_\oplus$')

### radius histogram
ax3.hist(m2r_r,fc='b', orientation='horizontal', bins=nbin)
ax3.hist(r2m_r,fc='g', orientation='horizontal', bins=nbin)
#ax3.hist(r2m_r2, fc='r', ec='r', alpha=0.3, bins=nbin)

### save
ax1.set_xlim([-4.,6.])
ax2.set_xlim([-4.,6.])
ax2.set_ylim([-1.3,2.1])
ax3.set_ylim([-1.3,2.1])

fig.savefig('demo_mr.pdf')