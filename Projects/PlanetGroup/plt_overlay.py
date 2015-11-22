import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import *
from func import convert_data, piece_linear

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('plt_overlay.pdf')

### log-log data
data_dir = '/Users/jingjing/Work/DK_project/Data/Mine/'
dwarfplanet, planet, browndwarf, star = \
np.loadtxt(data_dir+'dpl.txt'), \
np.loadtxt(data_dir+'pl.txt'), \
np.loadtxt(data_dir+'bd.txt'), \
np.loadtxt(data_dir+'st.txt')

data = [dwarfplanet, planet, browndwarf, star]

### mcmc result
mcmc_dir = '/Users/jingjing/Work/DK_project/Output/OutFile/PlanetGroup/straight/'
hyper = np.loadtxt(mcmc_dir+'str_hyper.out')
loglike = np.loadtxt(mcmc_dir+'str_loglike.out')


# best fit
best_ind = np.argmax(loglike)
best_hyper = hyper[best_ind]
print 'best_hyper', best_hyper


# thin
def thin(origin, fraction=0.5, jump=25):
	n_step = np.shape(origin)[0]
	n_start = int(n_step*fraction)
	h = origin[n_start:n_step:jump, ]
	return h

thin_hyper = thin(hyper)

### split data
trans = thin_hyper[:,9:12]


# 68, 95 percentage of translocation
def quantile(para):
	one_up = np.percentile(para, 84., axis=0, interpolation='nearest')
	one_down = np.percentile(para, 16., axis=0, interpolation='nearest')
	two_up = np.percentile(para, 97.5, axis=0, interpolation='nearest')
	two_down = np.percentile(para, 2.5, axis=0, interpolation='nearest')

	return one_up, one_down, two_up, two_down

# trans stat
trans_med = np.percentile(trans, 50., axis=0, interpolation='nearest')
trans_1up, trans_1down, trans_2up, trans_2down = quantile(trans)


### plot
plt.clf()
figscale=1.5
fig = plt.figure(figsize=(4*figscale,3*figscale))
ax = plt.gca()


# fit line
m_max = np.log10(np.max(star[:,0]))
m_min = np.log10(np.min(dwarfplanet[:,0]))
m_sample = np.linspace( m_min, m_max, 1000 )
r_sample = piece_linear(best_hyper, m_sample, prob_R = 0.5*np.ones_like(m_sample))
r_upper = piece_linear(best_hyper, m_sample, prob_R = 0.84 * np.ones_like(m_sample))
r_lower = piece_linear(best_hyper, m_sample, prob_R = 0.16 * np.ones_like(m_sample))
r_2upper = piece_linear(best_hyper, m_sample, prob_R = 0.975 * np.ones_like(m_sample))
r_2lower = piece_linear(best_hyper, m_sample, prob_R = 0.025 * np.ones_like(m_sample))


# plt fit
plt.plot(m_sample, r_sample, 'r-')
plt.fill_between(m_sample, r_lower, r_upper, color='grey', alpha=0.6)
plt.fill_between(m_sample, r_2lower, r_2upper, color='grey', alpha=0.4)

# trans location
for i in range(len(trans_med)):
	plt.axvline(trans_med[i], linestyle = 'dashed' , color='r')
	plt.axvline(trans_1up[i], linestyle = 'dotted', color='m',alpha=0.7)
	plt.axvline(trans_1down[i], linestyle = 'dotted', color='m',alpha=0.7)


# class bg color
r=Rectangle((-4.,-2.), trans_med[0]+4, 5., alpha=0.45, color='yellow')
ax.add_patch(r)
r=Rectangle((trans_med[0],-2.), trans_med[1]-trans_med[0], 5., alpha=0.3,color='yellow')
ax.add_patch(r)
r=Rectangle((trans_med[1],-2.), trans_med[2]-trans_med[1], 5., alpha=0.15, color='yellow')
ax.add_patch(r)
r=Rectangle((trans_med[2],-2.), 6-trans_med[2], 5., alpha=0.05, color='yellow')
ax.add_patch(r)



# data
alpha = [0.9,0.9,0.9,0.9]
colors = ['k','b','c','g']
fmts = ['o','o','s','h']
sizes = [2,3,3,5]
legends = []
for i,datum in enumerate(data):
	dat = convert_data(datum)
	alegend, = plt.plot(dat[:,0], dat[:,2], linestyle='None',
					color=colors[i], marker=fmts[i], markersize=sizes[i], alpha = alpha[i])
	plt.errorbar(dat[:,0], dat[:,2], xerr=dat[:,1], yerr=dat[:,3], 
				ecolor=colors[i], fmt='none',
				capsize=0, capthick=1, elinewidth=1)
	legends.append(alegend)

plt.legend(legends,[r'$\rm moons/dwarf\ planets$',r'$\rm planets$',r'$\rm brown\ dwarfs$',r'$\rm stars$'],
			loc=(0.04,0.7),prop={'size':10}, numpoints=1)


# set log scale ticks
# x axis
tick = np.zeros(9*(6+4)+1)
for i in range(6+4):
	tick[i*9:(i+1)*9] = np.linspace(10.**(i-4), 9.*10.**(i-4), 9)
tick[-1] = 10.**6

xticks = np.log10(tick)

labelnumber = [[r'$10^{-4}$'],[r'$10^{-3}$'],[r'$10^{-2}$'],[r'$10^{-1}$'],[r'$10^{0}$'],\
			[r'$10^{1}$'],[r'$10^{2}$'],[r'$10^{3}$'],[r'$10^{4}$'],[r'$10^{5}$'],[r'$10^{6}$']]
labelrest = ['']*8

xticklabel = []
for i in range(6+4):
	xticklabel += labelnumber[i] + labelrest	
xticklabel += labelnumber[-1]

plt.xticks( xticks, xticklabel )

# yaxis
tick = np.zeros(9*(2+1)+1)
for i in range(2+1):
	tick[i*9:(i+1)*9] = np.linspace(10.**(i-1), 9.*10.**(i-1), 9)
tick[-1] = 10.**2

yticks = np.log10(tick)

labelnumber = [[r'$10^{-1}$'],[r'$10^{0}$'],[r'$10^{1}$'],[r'$10^{2}$']]
labelrest = ['']*8

yticklabel = []
for i in range(2+1):
	yticklabel += labelnumber[i] + labelrest	
yticklabel += labelnumber[-1]


plt.yticks( yticks, yticklabel )

# xy label

plt.xlabel(r'$\rm M/M_\oplus$', fontsize=12)
plt.ylabel(r'$\rm R/R_\oplus$', fontsize=12, rotation=0)
ax.xaxis.set_label_coords(0.5,-0.06)
ax.yaxis.set_label_coords(-0.08,0.47)



pp.savefig()
pp.close()


