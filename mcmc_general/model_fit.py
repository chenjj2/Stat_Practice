'''
polynomial (1st order/2nd/3rd) & etc model that fits the data
'''


### 1st polynomial
def model_poly1(x,p):
        c0,c1 = p
        return c0 + c1*x
        
### 2nd polynomial
def model_poly2(x,p):
        c0,c1,c2 = p
        return c0 + c1*x + c2*x**2.
        
### 3rd polynomial      
def model_poly3(x,p):
        c0,c1,c2,c3 = p
        return c0 + c1*x + c2*x**2. + c3*x**3.


### radius as a function of mass, alpha(frac_iron), beta(frac_silicate)
''' using Li Zeng's table output, nearest neighbor interpolation '''
import numpy as np

radius_table = np.loadtxt('/Users/jingjing/Work/Model/Composition_LZeng/Radius.out', delimiter=';', unpack=True)

def radius_lz(mass,alpha,beta):
        index = np.round([ mass*4., alpha*100., beta*100. ]).astype(int)
        mass_ind, iron_ind, sili_ind = index[0]-2, index[1], index[2]
        row_ind = mass_ind * 100 + iron_ind
        col_ind = sili_ind
        rad = radius_table[row_ind, col_ind]
        
        return rad


