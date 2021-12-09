"""
Wavelet for source time function

@date: 2021-12-09
@author: zchen
"""

import numpy as np


def ricker(t, f0):
    """
    Ricker wavelet, 
    see https://pysit.readthedocs.io/en/latest/exercises/part_1.html#seismic-sources

    Parameters:
    -----------
    t: float or torch.Tensor
        Time of source time function

    f0: float
        Dominant frequency of the source
        


    Returns:
    --------
    source: the same as `t`
        Source time function value at time `t` 
    """

    sigma = 1 / (np.pi * f0 * np.sqrt(2))
    t0 = 6 * sigma # time shift
    tmp = np.pi**2 * f0**2 * (t-t0)**2 
    w = (1 - 2*tmp) * np.exp(-tmp)
    return w

