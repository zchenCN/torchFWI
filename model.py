"""
Underground physical model for Full Waveform Inversion

@data: 2021-12-09
@author: chazen
"""

import os
import segyio 
import numpy as np 
import matplotlib.pyplot as plt

__all__ = ['marmousi', 'bp2004']

currentdir = os.path.dirname(os.path.abspath(__file__))


def marmousi():
    """Marmousi model for 2d scalar wave equation

    Returns:
    --------
    v: np.array of shape (101, 101)
        Velocity in a square region
    """
    filepath = currentdir + '/data/marmousi_vel.segy'
    with segyio.open(filepath, mode='r', strict=False) as f:
        trace = f.trace
        num_traces = len(trace)
        v = tuple()
        for i in range(num_traces):
            vi = trace[i]
            v = v + (vi,)
        v = np.vstack(v)
    
    # Select a subregion
    n_row = np.arange(101) * 6
    n_col = 1000 + np.arange(101) * 10
    v = v[n_row]
    v = v[:, n_col]
    return v


def bp2004():
    """
    BP 2004 model 

        Returns:
    --------
    v: np.array of shape (101, 101)
        Velocity in a square region
    """
    filepath = currentdir + '/data/bp2004_vel.segy'
    with segyio.open(filepath, strict=False) as f:
        trace = f.trace
        num_traces = len(trace)
        v = tuple()
        for i in range(num_traces):
            vi = trace[i]
            v = v + (vi,)
        v = np.column_stack(v)

    # Select a subregion
    n_row = np.arange(101) * 15
    n_col = 2000 + np.arange(101) * 20
    v = v[n_row]
    v = v[:, n_col]

    return v


if __name__ == "__main__":

    fig = plt.figure(figsize=(9, 5))
    # Marmousi
    v_ma = marmousi()
    ax1 = fig.add_subplot(121)
    m1 = ax1.imshow(v_ma, aspect='equal')
    ax1.xaxis.set_ticks_position('top')
    ax1.set_title("Marmousi model", y=-0.1)
    # Bp 2004
    v_bp = bp2004()
    ax2 = fig.add_subplot(122)
    m1 = ax2.imshow(v_bp, aspect='equal')
    ax2.xaxis.set_ticks_position('top')
    ax2.set_title("BP2004 model", y=-0.1)
    plt.savefig(currentdir + "/figure/model.jpg")


    