import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import *
from func import convert_data, piece_linear
from scipy.stats import percentileofscore

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('plt_overlay.pdf')

### log-log data
data_dir = '/Users/jingjing/Work/DK_project/Data/Mine/'
dwarfplanet, exoplanet, browndwarf, star = \
np.loadtxt(data_dir+'dpl.txt'), \
np.loadtxt(data_dir+'epl.txt'), \
np.loadtxt(data_dir+'bd.txt'), \
np.loadtxt(data_dir+'st.txt')

data = [dwarfplanet, exoplanet, browndwarf, star]

### solar planets
# Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
solarplanet = convert_data( np.loadtxt(data_dir+'spl.txt') )


### mcmc result, FIXME
mcmc_dir = '/Users/jingjing/Work/DK_project/Output/OutFile/PlanetGroup/straight/'
hyper = np.loadtxt(mcmc_dir+'str_hyper.out')
loglike = np.loadtxt(mcmc_dir+'str_loglike.out')

### burn out
nburn = np.shape(hyper)[0]/2
hyper = hyper[nburn:,:]
loglike = loglike[nburn:]

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

#thin_hyper = thin(hyper)

### split data
slope_best = best_hyper[1:5]
trans = hyper[:,9:12]


# 68, 95 percentage of translocation
def quantile(para, mid_point):
	one_up = np.zeros(3)
	one_down = np.zeros(3)
	for i in range(3):	
		quantile_mid = percentileofscore(para[:,i], mid_point[i])
		print quantile_mid
		one_up[i] = np.percentile(para[:,i], np.min([quantile_mid + 34.,100.]), interpolation='nearest')
		one_down[i] = np.percentile(para[:,i], np.max([quantile_mid - 34., 0.]), interpolation='nearest')

	return one_up, one_down

# trans stat
trans_best = best_hyper[-3:]
trans_1up, trans_1down = quantile(trans, trans_best)

# convert trans best
mearth2mjup = 317.828
mearth2msun = 333060.4

m_trans = 10.**trans_best
m1 = m_trans[0]
m2 = m_trans[1] / mearth2mjup
m3 = m_trans[2] / mearth2msun


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
for i in range(len(trans_best)):
	plt.axvline(trans_best[i], linestyle = 'dashed' , color='k', alpha=0.9, linewidth=0.5)
	plt.axvline(trans_1up[i], linestyle = 'dotted', color='k',alpha=0.8, linewidth=0.5)
	plt.axvline(trans_1down[i], linestyle = 'dotted', color='k',alpha=0.8, linewidth=0.5)

# trans footnote
footnote_fontsize = 9
plt.text(0.1,-0.8,r'$\rm %.1f M_\oplus$' % m1 , fontsize=footnote_fontsize, rotation=90)
plt.text(1.5,-0.8,r'$\rm %.2f M_J$' % m2 , fontsize=footnote_fontsize, rotation=90)
plt.text(4.,-0.7,r'$\rm %.3f M_\odot$' % m3 , fontsize=footnote_fontsize, rotation=90)

# trans topnote
plt.text(0.1,2.02, r'$\rm volatile$', fontsize=footnote_fontsize, rotation=90)
plt.text(0.8,2.05, r'$\rm accretion$', fontsize=footnote_fontsize, rotation=90)

plt.text(1.5,2., r'$\rm grav.$', fontsize=footnote_fontsize, rotation=90)
plt.text(2.1,2.05, r'$\rm compression$', fontsize=footnote_fontsize, rotation=90)

plt.text(4.05,2.05, r'$\rm hydrogen$', fontsize=footnote_fontsize, rotation=90)
plt.text(4.55,2.02, r'$\rm burning$', fontsize=footnote_fontsize, rotation=90)


# class footnote

plt.text(-1.5,0.2,r'$\rm M \sim R^{%.2f}$' % slope_best[0] , fontsize=footnote_fontsize)
plt.text(-1.47,0.1,r'$\rm rocky$' , fontsize=footnote_fontsize)
plt.text(-1.5,0.,r'$\rm worlds$' , fontsize=footnote_fontsize)

plt.text(0.75,-0.2,r'$\rm M \sim R^{%.2f}$' % slope_best[1] , fontsize=footnote_fontsize)
plt.text(0.75,-0.3,r'$\rm gaseous$' , fontsize=footnote_fontsize)
plt.text(0.8,-0.4,r'$\rm worlds$' , fontsize=footnote_fontsize)

plt.text(2.6,0.5,r'$\rm M \sim R^{%.2f}$' % slope_best[2] , fontsize=footnote_fontsize)
plt.text(2.4,0.4,r'$\rm degenerate$' , fontsize=footnote_fontsize)
plt.text(2.7,0.3,r'$\rm worlds$' , fontsize=footnote_fontsize)

plt.text(4.7,0.8,r'$\rm M \sim R^{%.2f}$' % slope_best[3] , fontsize=footnote_fontsize)
plt.text(4.7,0.7,r'$\rm stellar$' , fontsize=footnote_fontsize)
plt.text(4.7,0.6,r'$\rm worlds$' , fontsize=footnote_fontsize)

# class bg color
r=Rectangle((-4.,-2.), trans_best[0]+4, 5., alpha=0.35, color='b')
ax.add_patch(r)
r=Rectangle((trans_best[0],-2.), trans_best[1]-trans_best[0], 5., alpha=0.25,color='b')
ax.add_patch(r)
r=Rectangle((trans_best[1],-2.), trans_best[2]-trans_best[1], 5., alpha=0.15, color='b')
ax.add_patch(r)
r=Rectangle((trans_best[2],-2.), 6-trans_best[2], 5., alpha=0.05, color='b')
ax.add_patch(r)



# data
fmts = ['^','o','s','h']
legends = []
for i,datum in enumerate(data):
	dat = convert_data(datum)
	alegend, = plt.plot(dat[:,0], dat[:,2], linestyle='None',
					marker=fmts[i], markersize=4, 
					markerfacecolor = 'None', markeredgecolor='k', markeredgewidth=0.5
				)
	'''plt.errorbar(dat[:,0], dat[:,2], xerr=dat[:,1], yerr=dat[:,3], 
				ecolor='k', fmt='none',
				capsize=0, capthick=1, elinewidth=0.5,
				zorder = 50)'''
	legends.append(alegend)

plt.legend(legends,[r'$\rm moons/dwarf\ planets$',r'$\rm exoplanets$',r'$\rm brown\ dwarfs$',r'$\rm stars$'],
			loc=(0.04,0.7),prop={'size':9}, numpoints=1)


### plot solar planets symbol
plt.plot(solarplanet[:,0],solarplanet[:,2], 'm.', markersize = 0.5)
# earth
plt.plot(solarplanet[2,0],solarplanet[2,2], '+', markerfacecolor = 'None', markeredgecolor='#71EEB8', markersize = 4, markeredgewidth=0.5)
plt.plot(solarplanet[2,0],solarplanet[2,2], 'o', markerfacecolor = 'None', markeredgecolor='#71EEB8', markersize = 4, markeredgewidth=0.5)


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
plt.ylim([-1.2, 2.2])
tick = np.zeros(3+9*(2+1)+1)
tick[0:3] = np.arange(7,10) * 10.**-2
for i in range(2+1):
	tick[3+i*9:3+(i+1)*9] = np.linspace(10.**(i-1), 9.*10.**(i-1), 9)
tick[-1] = 10.**2

yticks = np.log10(tick)

labelnumber = [[r'$10^{-1}$'],[r'$10^{0}$'],[r'$10^{1}$'],[r'$10^{2}$']]
labelrest = ['']*8

yticklabel = ['']*3
for i in range(2+1):
	yticklabel += labelnumber[i] + labelrest	
yticklabel += labelnumber[-1]


plt.yticks( yticks, yticklabel )

# xy label
plt.xlabel(r'$\rm M/M_\oplus$', fontsize=12)
plt.ylabel(r'$\rm R/R_\oplus$', fontsize=12) # may inverse the letter by specify: rotation=0
ax.xaxis.set_label_coords(0.5,-0.06)
ax.yaxis.set_label_coords(-0.08,0.5)
#ax.yaxis.set_label_coords(-0.08,0.47)



pp.savefig()
pp.close()


