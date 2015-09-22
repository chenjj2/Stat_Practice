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
