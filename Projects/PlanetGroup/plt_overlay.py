import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import *
from func import convert_data, piece_linear, piece_linear_complex
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


### mcmc result, spatial_median and trans_1up/down

best_hyper = np.loadtxt('h4_spatial_median.txt', delimiter=',')
trans_best = best_hyper[-3:]
slope_best = best_hyper[1:5]
# find_best.py with spatial_median.txt and find trans_1up/down
trans_1up = trans_best + np.array([0.1245372 , 0.0534476 ,  0.04035559])
trans_1down = trans_best - np.array([0.14317994, 0.06079727,  0.03514506 ])


# convert trans best
mearth2mjup = 317.828
mearth2msun = 333060.4

m_trans = 10.**trans_best
m1 = m_trans[0]
m2 = m_trans[1] / mearth2mjup
m3 = m_trans[2] / mearth2msun

# find the corresponding R at transition points
rad_trans = 10.** piece_linear( best_hyper, trans_best, prob_R = 0.5*np.ones(3) )
print 'radius at transition (RE)/(RJ)', rad_trans, rad_trans/11.21


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
	plt.plot([trans_best[i]]*2, [-1.2,1.9], linestyle = 'dashed' , color='k', alpha=0.9, linewidth=0.5)
	plt.plot([trans_1up[i]]*2, [-1.2,1.9], linestyle = 'dotted' , color='k', alpha=0.8, linewidth=0.5)
	plt.plot([trans_1down[i]]*2, [-1.2,1.9], linestyle = 'dotted' , color='k', alpha=0.8, linewidth=0.5)

# trans footnote
footnote_fontsize = 9
plt.text(trans_1down[0]-0.26,-0.8,r'$\rm %.1f M_\oplus$' % m1 , fontsize=footnote_fontsize, rotation=90)
plt.text(trans_1down[1]-0.26,-0.79,r'$\rm %.2f M_J$' % m2 , fontsize=footnote_fontsize, rotation=90)
plt.text(trans_1down[2]-0.26,-0.72,r'$\rm %.3f M_\odot$' % m3 , fontsize=footnote_fontsize, rotation=90)

# trans topnote
#plt.text(trans_1down[0]-0.26,2.02, r'$\rm volatile$', fontsize=footnote_fontsize, rotation=90)
#plt.text(trans_1up[0]+0.05,2.05, r'$\rm accretion$', fontsize=footnote_fontsize, rotation=90)
plt.text(trans_best[0]-0.45,2.02, r'$\rm volatile$', fontsize=footnote_fontsize)
plt.text(trans_best[0]-0.5,1.92, r'$\rm accretion$', fontsize=footnote_fontsize)

#plt.text(trans_1down[1]-0.26,1.93, r'$\rm grav.$', fontsize=footnote_fontsize, rotation=90)
#plt.text(trans_1up[1]+0.05,2.05, r'$\rm compression$', fontsize=footnote_fontsize, rotation=90)
plt.text(trans_best[1]-0.25,2.02, r'$\rm grav.$', fontsize=footnote_fontsize)
plt.text(trans_best[1]-0.6,1.92, r'$\rm compression$', fontsize=footnote_fontsize)


#plt.text(trans_1down[2]-0.26,2.05, r'$\rm hydrogen$', fontsize=footnote_fontsize, rotation=90)
#plt.text(trans_1up[2]+0.05,2.02, r'$\rm burning$', fontsize=footnote_fontsize, rotation=90)
plt.text(trans_best[2]-0.45,2.02, r'$\rm hydrogen$', fontsize=footnote_fontsize)
plt.text(trans_best[2]-0.4,1.92, r'$\rm burning$', fontsize=footnote_fontsize)


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
fmts = ['^','o','s','*']
size = [4,4,4,6]
legends = []
for i,datum in enumerate(data):
	dat = convert_data(datum)
	alegend, = plt.plot(dat[:,0], dat[:,2], linestyle='None',
					marker=fmts[i], markersize=size[i], 
					markerfacecolor = 'None', markeredgecolor='k', markeredgewidth=0.5
				)
	legends.append(alegend)

plt.legend(legends,[r'$\rm moons/dwarf\ planets$',r'$\rm exoplanets$',r'$\rm brown\ dwarfs$',r'$\rm stars$'],
			loc=(0.04,0.68),prop={'size':9}, numpoints=1)


### plot solar planets symbol
plt.plot(solarplanet[:,0],solarplanet[:,2], 'm.', markersize = 0.5)
# earth
plt.plot(solarplanet[2,0],solarplanet[2,2], '+', markerfacecolor = 'None', markeredgecolor='#ffa500', markersize = 4, markeredgewidth=0.5)
plt.plot(solarplanet[2,0],solarplanet[2,2], 'o', markerfacecolor = 'None', markeredgecolor='#ffa500', markersize = 4, markeredgewidth=0.5)


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


