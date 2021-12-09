"""2D scalar wave equation forward modeling implemented using PyTorch

author: zchen
date: 2021-09-16
"""

__all__ = ['forward2d', 'gen_data']

import numpy as np
import torch 
import torch.nn as nn


class TimeStepCell(nn.Module):
    """One forward modeling step of scalar wave equation with PML

    Attributes:
    -----------
    model_padded2_dt: torch.Tensor 
        A 2D tensor containing the padded, squared model times time step size

    dt: float
        Time step size 

    h: float
        Size of square grid cells

    sigmax: torch.Tensor of shape (nptz_padded, nptx_padded)
        Dampling parameters in x-direction on grid points

    sigmax_half_grid: torch.Tensor of shape (nptz_padded-2, nptx_padded-1)
        Dampling parameters in x-direction on grid points

    sigmaz: torch.Tensor of shape (nptz_padded, nptx_padded)
        Dampling parameters in z-direction on grid points

    sigmaz_half_grid: torch.Tensor of shape (nptz_padded-1, nptx_padded-2)
        Dampling parameters in z-direction on grid points
    """
    def __init__(self, model_padded2_dt, dt, h, sigmaz, sigmaz_half_grid,
                    sigmax, sigmax_half_grid):
        super().__init__()
        self.model_padded2_dt = model_padded2_dt
        device = model_padded2_dt.device
        self.dt = dt 
        self.h = h
        self.sigmaz = sigmaz.to(device)
        self.sigmax = sigmax.to(device)
        self.sigmaz_half_grid = sigmaz_half_grid.to(device)
        self.sigmax_half_grid = sigmax_half_grid.to(device)

    def forward(self, state, input):
        """Propagate the wavefield forward one time step and inject sources

        Parameters:
        -----------
        state: tuple
            A tupe containing [Px, Pz, Ax, Az] at last time step

        input: torch.Tensor of shape (ns, nptz_padded, nptx_padded)
            3D tensor containing the sources to be injected into wavefile in
            this time step size

        Returns:
        --------
        state: tuple
            A tupe containing [Px, Pz, Ax, Az] at current time step

        output: torch.Tensor of shape (ns, nptz_padded, nptx_padded)
            The current wavefield
        """
        Px, Pz, Ax, Az = state 

        #--------------------------Main evolution-----------------------------------------
        # First derivatives
        Ax_x = (Ax[:, :, 1:] - Ax[:, :, :-1]) / self.h
        Ax_x = nn.functional.pad(Ax_x, (1, 1, 1, 1), 'constant', 0.)
        Az_z = (Az[:, 1:, :] - Az[:, :-1, :]) / self.h 
        Az_z = nn.functional.pad(Az_z, (1, 1, 1, 1), 'constant', 0.)


        Px = (self.model_padded2_dt * Ax_x 
            + (1 - self.sigmax * self.dt / 2) * Px) / (1 + self.sigmax * self.dt / 2)
        Pz = (self.model_padded2_dt * Az_z 
            + (1 - self.sigmaz * self.dt / 2) * Pz) / (1 + self.sigmaz * self.dt / 2) 
        del Ax_x
        del Az_z
        # Inject the sources
        P = Px + Pz + input * self.dt 
        # First derivatives
        P_x = (P[:, 1:-1, 1:]- P[:, 1:-1, :-1]) / self.h 
        P_z = (P[:, 1:, 1:-1] - P[:, :-1, 1:-1]) / self.h 

        Ax = ((1 - self.sigmax_half_grid * self.dt / 2) * Ax
            + self.dt * P_x ) / (1 + self.sigmax_half_grid * self.dt / 2)
        Az = ((1 - self.sigmaz_half_grid * self.dt / 2) * Az
            + self.dt * P_z ) / (1 + self.sigmaz_half_grid * self.dt / 2)
        #--------------------------Main evolution-----------------------------------------
        del P_x
        del P_z
        statef = (Px, Pz, Ax, Az)
        return (statef, P)


def acoustic2d(model, source_time, sources_x, h, dt, pml_width=10):
    """Forward modeling of 2D scalar wave equation

    Parameters:
    -----------
    model: torch.Tensor of shape (nptz, nptx)
        Velocity model in pysical domain

    source_time: torch.Tensor of shape (nt)
        Amplitude of source time function

    sources_x: torch.Tensor of shape (ns, 2)
        2D tensor containing the indices of source positions, sources_x[i, 0] 
        is the index of source i in z-direction, sources_x[i, 1] is the index 
        of source i in x-direction

    h: float
        Size of square grid cells

    dt: float
        Time step size 

    pml_width: int
        Interger specifying the width of each PML region

    Returns:
    --------
    P_all_time: torch.Tensor of shape (nt, ns, nptz, nptx)
        4D tensor containing wavefield of all time steps
    """
    nptz = model.shape[0]
    nptx = model.shape[1]
    nt = len(source_time)
    assert sources_x.dim() == 2
    ns = len(sources_x)
    device = model.device

    # CFL check
    cfl = model.max().item() * dt / h
    if cfl > 0.8:
        raise ValueError(f"CFL number={cfl} is too large, try to adjust h and dt")

    nptz_padded, nptx_padded  = _set_pad(model, pml_width)
    model_padded2_dt = _set_model(model, pml_width, dt)
    sigmaz, sigmaz_half_grid, sigmax, sigmax_half_grid \
                                = _set_pml(pml_width, nptz_padded, nptx_padded, h)
    sources = _set_source(source_time, sources_x, pml_width, nptz_padded, nptx_padded)
    sources = sources.to(device)
    cell = TimeStepCell(model_padded2_dt, dt, h, sigmaz, sigmaz_half_grid, 
                            sigmax, sigmax_half_grid)

    # Time extrapolate
    P_all_time = torch.zeros(nt, ns, nptz, nptx, device=device)
    Px = torch.zeros(ns, nptz_padded, nptx_padded, device=device)
    Pz = torch.zeros(ns, nptz_padded, nptx_padded, device=device)
    Ax = torch.zeros(ns, nptz_padded-2, nptx_padded-1, device=device)
    Az = torch.zeros(ns, nptz_padded-1, nptx_padded-2, device=device)
    state = (Px, Pz, Ax, Az)
    for t in range(nt):
        input = sources[t]
        state, Pt = cell(state, input)
        P_all_time[t] = Pt[:, pml_width:-pml_width, pml_width:-pml_width]
    
    return P_all_time
    

# def gen_data(model, propagator, source_time, sources_x, h, dt, 
#                                         receivers_x):
#     """Generate sythetic seismic data for full waveform inversion

#     Parameters:
#     -----------
#     model: torch.Tensor
#         Underground velocity model used to generate sythetic seismic data

#     propagator: callable function
#         Forward propagate function 

#     source_time: torch.Tensor of shape (nt)
#         Amplitude of source time function

#     sources_x: torch.Tensor of shape (ns, 2)
#         2D tensor containing the indices of source positions, sources_x[i, 0] 
#         is the index of source i in z-direction, sources_x[i, 1] is the index 
#         of source i in x-direction

#     h: float
#         Size of square grid cells

#     dt: float
#         Time step size 

#     receivers_x: torch.Tensor of shape (ns, 2)
#         2D tensor containing the indices of receiver positions, reveivers_x[i, 0] 
#         is the index of receivers i in z-direction, receivers_x[i, 1] is the index 
#         of receiver i in x-direction

#     Returns:
#     --------
#     signal: torch.Tensor of shape (nt, ns, nr)
#         Seismic signal at receivers
#     """
#     P_all_time = propagator(model, source_time, sources_x, h, dt) 
#     # signal of shape (nt, ns, nr)
#     signal = P_all_time[:, :, receivers_x[:, 0], receivers_x[:, 1]]
#     assert not torch.any(torch.isnan(signal))
#     return signal     

def gen_data(model, source_time, sources_x, h, dt, 
                                        receivers_x):
    """Generate sythetic seismic data for full waveform inversion

    Parameters:
    -----------
    model: torch.Tensor
        Underground velocity model used to generate sythetic seismic data

    source_time: torch.Tensor of shape (nt)
        Amplitude of source time function

    sources_x: torch.Tensor of shape (ns, 2)
        2D tensor containing the indices of source positions, sources_x[i, 0] 
        is the index of source i in z-direction, sources_x[i, 1] is the index 
        of source i in x-direction

    h: float
        Size of square grid cells

    dt: float
        Time step size 

    receivers_x: torch.Tensor of shape (ns, 2)
        2D tensor containing the indices of receiver positions, reveivers_x[i, 0] 
        is the index of receivers i in z-direction, receivers_x[i, 1] is the index 
        of receiver i in x-direction

    Returns:
    --------
    signal: torch.Tensor of shape (nt, ns, nr)
        Seismic signal at receivers
    """
    P_all_time = acoustic2d(model, source_time, sources_x, h, dt) 
    # signal of shape (nt, ns, nr)
    signal = P_all_time[:, :, receivers_x[:, 0], receivers_x[:, 1]]
    assert not torch.any(torch.isnan(signal))
    return signal 

def _set_pad(model, total_pad):
    """Calculate the size of the model after padding has been added

    Parameters:
    -----------
    model: torch.Tensor of shape (nptz, nptx)
        Underlying velocity model for forward modeling, nptz and nptz are
        the number of grid points in z-direction and x-direction respectively 

    Returns:
    --------
    nptz_padded: int
        Interger specifying number of grid points in padded model in z-direction

    nptx_padded: int
        Interger specifying number of grid points in padded model in x-direction
    """
    nptz = int(model.shape[0])
    nptx = int(model.shape[1])
    nptz_padded = nptz + 2*total_pad
    nptx_padded = nptx + 2*total_pad
    return nptz_padded, nptx_padded


def _set_model(model, total_pad, dt):
    """Add padding to the model(extending edge values) and compute v^2 * dt

    Parameters:
    -----------
    model: torch.Tensor of shape (nptz, nptx)
        Underlying velocity model for forward modeling, nptz and nptz are
        the number of grid points in z-direction and x-direction respectively 
    
    total_pad: int
        Interger specifying number of cells to padding to each edge

    dt: float
        Time step size of finite difference method

    Returns:
    --------
        A 2D tensor containing the padded, squared model times time step size
    """
    pad_tensor = nn.ReplicationPad2d(total_pad)
    model = torch.unsqueeze(torch.unsqueeze(model, 0), 0)
    model_padded = pad_tensor(model).squeeze() 
    return torch.square(model_padded) * dt


def _set_profile(pml_width, h):
    """Create profile for the PML

    Parameters:
    -----------
    pml_width: int
        Interger specifying the width of PML

    h: float
        Float specifying spacing between grid points

    Returns:
    --------
    profile: torch.Tensor of shape (pml_width)
        Tensor containing PML profile on grid points

    profile_half_grid: torch.Tensor of shape (pml_width)
        Tensor containing PML profile on half grid points
    """
    vmax = 5000. # Approximate maximum wave velocity
    R = 1e-5 # Reflection coefficient
    if pml_width == 20:
        R = 1e-7

    profile = ((torch.arange(1, pml_width+1)/pml_width)**2
                * 3 * vmax * np.log(1/R) 
                / (2 * h * pml_width))

    profile_half_grid = (((torch.arange(pml_width) + 0.5)/pml_width)**2
                * 3 * vmax * np.log(1/R) 
                / (2 * h * pml_width))

    return profile, profile_half_grid


def _set_pml(pml_width, nptz_padded, nptx_padded, h):
    """Calculate dampling parameters in PML

    Paramters:
    ----------
    pml_width: int
        Interger specifying the width of PML

    nptz_padded: int
        Interger specifying number of grid points in padded model in z-direction

    nptx_padded: int
        Interger specifying number of grid points in padded model in x-direction

    h: float
        Float specifying spacing between grid points

    Returns:
    ----------
    sigmax: torch.Tensor of shape (nptz_padded, nptx_padded)
        Dampling parameters in x-direction on grid points

    sigmax_half_grid: torch.Tensor of shape (nptz_padded-2, nptx_padded-1)
        Dampling parameters in x-direction on grid points

    sigmaz: torch.Tensor of shape (nptz_padded, nptx_padded)
        Dampling parameters in z-direction on grid points

    sigmaz_half_grid: torch.Tensor of shape (nptz_padded-1, nptx_padded-2)
        Dampling parameters in z-direction on grid points
    """
    def sigma_1d(pml_width, npt_padded, profile, profile_half_grid):
        """Create 1D sigma array"""
        sigma = np.zeros(npt_padded, np.float32)
        sigma[pml_width-1::-1] = profile 
        sigma[-pml_width:] = profile 
        sigma_half_grid = np.zeros(npt_padded-1, np.float32)
        sigma_half_grid[pml_width-1::-1] = profile_half_grid
        sigma_half_grid[-pml_width:] = profile_half_grid
        return sigma, sigma_half_grid

    profile, profile_half_grid = _set_profile(pml_width, h)

    sigmaz, sigmaz_half_grid = sigma_1d(pml_width, \
                            nptz_padded, profile, profile_half_grid)
    sigmaz = sigmaz.reshape((-1, 1))
    sigmaz = np.tile(sigmaz, (1, nptx_padded))
    sigmaz_half_grid = sigmaz_half_grid.reshape((-1, 1))
    sigmaz_half_grid = np.tile(sigmaz_half_grid, (1, nptx_padded-2))

    sigmax, sigmax_half_grid = sigma_1d(pml_width, \
                            nptx_padded, profile, profile_half_grid)
    sigmax = sigmax.reshape((1, -1))
    sigmax = np.tile(sigmax, (nptz_padded, 1))
    sigmax_half_grid = sigmax_half_grid.reshape((1, -1))
    sigmax_half_grid = np.tile(sigmax_half_grid, (nptz_padded-2, 1))

    return torch.from_numpy(sigmaz), torch.from_numpy(sigmaz_half_grid), \
            torch.from_numpy(sigmax), torch.from_numpy(sigmax_half_grid)

    
def _set_source(source_time, sources_x, total_pad, nptz_padded, nptx_padded):
    """Set the source amplitudes and the source positions

    Parameters:
    -----------
    source_time: torch.Tensor of shape (nt)
        Amplitue of source time function

    sources_x: torch.Tensor of shape (ns, 2)
        2D tensor containing the indices of source positions, sources_x[i, 0] 
        is the index of source i in z-direction, sources_x[i, 1] is the index 
        of source i in x-direction
    
    total_pad: int 
        Integer specifying the width of padding region

    nptz_padded: int
        Interger specifying number of grid points in padded model in z-direction

    nptx_padded: int
        Interger specifying number of grid points in padded model in x-direction

    Returns:
    --------
    sources: torch.Tensor of shape (nt, ns, nptz_padded, nptx_padded)
        Tensor containing amplitude of source at source positions for each time step
    """
    nt = len(source_time)
    ns = len(sources_x)
    sources = torch.zeros(nt, ns, nptz_padded-2*total_pad, nptx_padded-2*total_pad)
    for t, st in enumerate(source_time):
        for i, sx in enumerate(sources_x):
            sources[t, i, sx[0], sx[1]] = st 
    sources = nn.functional.pad(sources, (total_pad, total_pad, total_pad, total_pad)
                , mode='constant', value=0)
    return sources

