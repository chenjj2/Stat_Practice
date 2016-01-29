import sys
sys.path.append('../PlanetGroup/')
from func import Rpost2M

import numpy as np

for i in range(4):
	file = 'radii.0'+str(i+1)+'.dat'
	radius = np.loadtxt(file)
	mass = Rpost2M(radius)
	np.savetxt('mass.0%i.dat' %(i+1),mass)
