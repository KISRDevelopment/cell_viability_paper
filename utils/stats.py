import numpy as np 
import scipy.stats 

def compute_stars(pvalue, alpha, return_level=False):
    """ compute the stars to visualize a p-value """
    PVALUE_STARS = [
        1.,
        1/5.,
        1/50.,
        1/500.
    ]

    for i in range(len(PVALUE_STARS)-1, -1, -1):
        level = PVALUE_STARS[i] * alpha
        if pvalue < level:
            if return_level:
                return i+1, level 
            else:
                return i+1

    return 0
