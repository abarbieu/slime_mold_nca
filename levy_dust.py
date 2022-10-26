import numpy as np 
from scipy.stats import uniform
from scipy.stats import levy_stable

# put x in range -w/2 to w/2
def norm_center(x,w):
    x -= x.min()
    x *= w/x.max()
    return x

def get_levy_dust(shape, points, alpha, beta):
    # uniformly distributed angles
    angle = uniform.rvs(size=(points,), loc=.0, scale=2.*np.pi )

    # Levy distributed step length
    r = abs(levy_stable.rvs(alpha, beta, size=points))

    x = norm_center(np.cumsum(r * np.cos(angle)), shape[0]-1)
    y = norm_center(np.cumsum(r * np.sin(angle)), shape[1]-1)
    
    return np.array([x,y])