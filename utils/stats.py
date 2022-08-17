import scipy
import numpy as np

def calc_jsd(p,q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """
    # from here https://medium.com/@sourcedexter/how-to-find-the-similarity-between-two-probability-distributions-using-python-a7546e90a08d
    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)
    # calculate m
    m = (p + q) / 2
    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
    # If two distribution differ only at level of machine precision,
    # set divergence to zero (to avoid negative numbers)
    if divergence<0:
        divergence = 0
    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)
    return distance