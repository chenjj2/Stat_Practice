### import 
import numpy as np
import matplotlib.pyplot as plt

### file paths
dir = '/Users/jingjing/Work/DK_project/Data/'

file = [
'2010-Torres/binarystar.csv', #0
'2011-Carter/2010-Kraus.csv', #1
#'2011-Carter/2009-Vida.csv', #2
'2011-Carter/2010-Cakirli.csv', #3
'2015-Hatzes/browndwarfs.csv', #4
'2015-Hatzes/otherstars.csv', #5
'DKipping/moons.dat', #6
'TEPCat/allplanets.csv', #7
'Wiki/dwarfplanet.csv', #8
'NASA/solarplanets.csv' #9
]


### unit convert
# http://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html
msun2mearth = 333000.
rsun2rearth = 109.2
# http://nssdc.gsfc.nasa.gov/planetary/factsheet/jupiterfact.html
mjup2mearth = 317.83
rjup2rearth = 10.973
# http://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
mearth2kg = 5.9726e24
rearth2km = 6371.0


### read 
# format: M (ME), Merr, R (RE), Rerr
i=0

d0 = np.loadtxt(dir+file[i], skiprows=48, usecols=(4,5,6,7), delimiter=';')
d0_m = d0[:,0:2]
d0_r = d0[:,2:4]
d0 = np.hstack(( d0_m * msun2mearth, d0_r * rsun2rearth ))
i +=1

d1 = np.loadtxt(dir+file[i], skiprows=1, usecols=(1,2,3,4,5), delimiter=',')
d1_m = d1[:,0:2]
d1_r = (d1[:,2]).reshape(len(d1[:,2]),1)
d1_rerr = ( (d1[:,3]**2.+ d1[:,4]**2.)**0.5 ).reshape(np.shape(d1_r))
d1 = np.hstack(( d1_m * msun2mearth, d1_r * rsun2rearth, d1_rerr * rsun2rearth ))
i +=1

'''
d2 = np.loadtxt(dir+file[2], skiprows=1, delimiter=',')
d2_m = d2[:,0:2]
d2_r = d2[:,2:4]
d2 = np.hstack(( d2_m * msun2mearth, d2_r * rsun2rearth ))
'''

d3 = np.loadtxt(dir+file[i], skiprows=1, delimiter=',')
d3_m = d3[:, 0:2]
d3_r = d3[:, 2:4]
d3 = np.hstack(( d3_m * msun2mearth, d3_r * rsun2rearth ))
i +=1

print dir+file[i]
d4 = np.loadtxt(dir+file[i], skiprows=1, usecols=(1,2,3,4,5,6),delimiter=',')
rowind = (d4[:,1]==d4[:,2]) & (d4[:,4]==d4[:,5]) # symmetric error
d4_m = d4[rowind, 0:2]
d4_r = d4[rowind, 3:5]
d4 = np.hstack(( d4_m * mjup2mearth, d4_r * rjup2rearth ))
i +=1

d5 = np.loadtxt(dir+file[i], skiprows=1, usecols=(1,2,3,4,5,6), delimiter=',')
rowind = (d5[:,1]==d5[:,2]) & (d5[:,4]==d5[:,5]) # symmetric error
d5_m = d5[rowind, 0:2]
d5_r = d5[rowind, 3:5]
d5 = np.hstack(( d5_m * msun2mearth, d5_r * rsun2rearth ))
i +=1

d6 = np.loadtxt(dir+file[i], skiprows=1, usecols=(1,2))
n_line,n_col = np.shape(d6)
d6_m = d6[:,0].reshape(n_line,1)
d6_r = d6[:,1].reshape(n_line,1)
err = np.zeros_like(d6_m)
d6 = np.hstack(( d6_m, err, d6_r, err ))
i +=1

d7 = np.loadtxt(dir+file[i], skiprows=1, usecols=(26,27,28,29,30,31),delimiter=',')
rowind = (d7[:,1]==d7[:,2]) & (d7[:,4]==d7[:,5]) # symmetric error
d7_m = d7[rowind, 0:2]
d7_r = d7[rowind, 3:5]
d7 = np.hstack(( d7_m * mjup2mearth, d7_r * rjup2rearth ))
i +=1

d8 = np.loadtxt(dir+file[i], skiprows=3, usecols=(1,2,3,4,5), delimiter=',')
rowind = (d8[:,3] == d8[:,4])
d8_m = d8[rowind, 0:2]
d8_d = d8[rowind, 2:4]
d8 = np.hstack(( d8_m*1e21/mearth2kg, d8_d/2./rearth2km ))
i +=1

d9 = np.loadtxt(dir+file[i], skiprows=5, usecols=(1,2),delimiter=',')
n_line,n_col = np.shape(d9)
d9_m = d9[:,0].reshape(n_line,1)
d9_r = d9[:,1].reshape(n_line,1)
err = np.zeros_like(d9_m)
d9 = np.hstack(( d9_m, err, d9_r, err ))


### combine data
#dat_list = [d0,d1,d2,d3,d4,d5,d6,d7,d8,d9]
dat = np.vstack(( d0,d1,d3,d4,d5,d6,d7,d8,d9 ))


### select
# valid data
ind = (dat[:,0]>0.) & (dat[:,1]>=0.) & (dat[:,2]>0.) & (dat[:,3]>=0.)
# 3 sigma cut
ind = ind & (dat[:,0]/dat[:,1] > 3.) & (dat[:,2]/dat[:,3] > 3.)
# mass range
max_mass = 2.9e5 # less than universe age
min_mass = 5.7e-5 # wiki dwarf planet, the transition to hydrostatic equilibrium
ind = ind & (dat[:,0] > min_mass) & (dat[:,0] < max_mass)

dat = dat[ind,:]

### pass data
def data():
	return dat


### examine data
def plot():
	plt.errorbar(dat[:,0], dat[:,2], xerr=dat[:,1], yerr=dat[:,3], fmt='.')
	plt.xscale('log'); plt.yscale('log')
	plt.xlabel(r'M [M$_\oplus$]'); plt.ylabel(r'R [R$_\oplus$]')
	plt.savefig('MR_data.png')
	return 0


### save data
def save():
	np.savetxt(dir+'Mine/PlanetGroup.txt',dat)
	return 0


### main
if __name__ == '__main__':
	plot()
	save()


