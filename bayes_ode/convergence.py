import numpy as np

from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d

# functions to estimate slope of loss over time
def lin_fit(x, a, b):
    return a + b * x

def check_convergence(f):

    # smooth function evaluations
    f = uniform_filter1d(f, size=10, mode='nearest')

    # fit linear fit to estimate slope of convergence 
    p, cov = curve_fit(lin_fit, xdata=np.arange(len(f)), ydata=f / np.max(np.abs(f)), p0=[1., 0.])
    a, b, = p

    # return value of slope
    return b