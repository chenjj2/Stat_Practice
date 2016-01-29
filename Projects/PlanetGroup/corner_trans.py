import numpy as np
import corner

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('corner_trans.pdf')

dir = '/Users/jingjing/Work/DK_project/Stat_Practice/Projects/PlanetGroup/'
all_trans = np.loadtxt(dir+'h4_thin_hyper.out')[:,-3:]

### thin
def thin(all, factor=1):
	thinned = all[::factor,:]
	return thinned

trans_me = 10.**( thin(all_trans,1) )

### convert to proper unit
mearth2mjup = 317.828
mearth2msun = 333060.4

trans_plot = np.vstack(( np.vstack((trans_me[:,0], trans_me[:,1]/mearth2mjup)), 
						trans_me[:,2]/mearth2msun )).transpose()


### corner plot
figure = corner.corner(trans_plot, alpha=0.1,
labels = [r'$\rm volatile\ accretion\ [M_\oplus]$',
		r'$\rm grav.\ compression\ [M_J]$',
		r'$\rm hydrogen\ burning\ [M_\odot]$'],
#truths = [0., 0., 0.],
#quantiles = [0.16, 0.5, 0.84],
#show_titles=True, title_args={'fontsize':12}
)
#figure.savefig('corner_trans.png')

pp.savefig()
pp.close()


