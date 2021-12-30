"""
Full waveform inverion for 2d scalar wave equation

@date: 2021-12-09
@author: chazen
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from forward import gen_data, acoustic2d
from source import ricker


class TVLoss(nn.Module):
    """Total variation loss for tensors
    """
    def __init__(self, weight):
        super.__init__()
        self.weight = weight

    def forward(self, t):
        tv_h = torch.square(t[1:, :] - t[:-1, :]).sum()
        tv_w = torch.square(t[:, 1:] - t[:, :-1]).sum()
        h, w = t.size()
        return self.weight * (tv_h + tv_w) / (h * w)


class SeisData(Dataset):
    """ Synthetic seismic dataset

    Attributes:
    -----------
    signal: torch.Tensor of shape (nt, ns, nr)
        Seismic signal at receivers

    sources_x: torch.Tensor of shape (ns, 2)
        2D tensor containing the indices of source positions, sources_x[i, 0] 
        is the index of source i in z-direction, sources_x[i, 1] is the index 
        of source i in x-direction

    receivers_x: torch.Tensor of shape (ns, 2)
        2D tensor containing the indices of receiver positions, reveivers_x[i, 0] 
        is the index of receivers i in z-direction, receivers_x[i, 1] is the index 
        of receiver i in x-direction
    """ 
    def __init__(self, signal, sources_x, receivers_x, noise_level=0.):
        super().__init__()
        # Add noise here ?????
        noise = torch.zeros_like(signal)
        self.signal = signal + noise_level * noise # signal of shape (nt, ns, nr)

        self.sources_x = sources_x 
        self.receivers_x = receivers_x 

    def __len__(self):
        return self.signal.shape[1]

    def __getitem__(self, id):
        return (self.sources_x[id], self.signal[:, id, :])


class FWI:
    """Full waveform inversion
    """
    def __init__(self, true_model, init_model, sources_x, receivers_x, 
                h=10., nt=1000, dt=0.001, f0=10.):
        super.__init__()
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.true_model = torch.tensor(true_model).float().to(self.device)
        # self.init_model = torch.tensor(init_model, requires_grad=True).float().to(self.device)
        self.sources_x = sources_x
        self.receivers_x = receivers_x
        self.h = h 
        self.nt = nt
        self.dt = dt
        self.ts = torch.arange(nt) * dt
        self.source_time = ricker(self.ts, f0)
        self.create_dataset()
        

    def create_dataset(self):
        signal = gen_data(self.true_model, self.source_time, 
                self.sources_x, self.h, self.dt, self.receivers_x)
        
        self.dataset = SeisData(signal, self.sources_x, self.receivers_x)

    def compute_loss(self, syn, obs, f0):
        obs = obs.permute(1, 0, 2)
        obs = obs.flatten(start_dim=1)
        obs = obs.T
        syn = syn.flatten(start_dim=1)
        syn = syn.T 
        # Low pass filter
        F_obs = torch.fft.rfft(obs, dim=1)
        freq = torch.fft.rfftfreq(self.nt, self.dt)
        F_obs = torch.where(freq>f0, 0, F_obs)
        obs = torch.fft.irfft(F_obs)
        F_syn = torch.fft.rfft(syn, dim=1)
        F_syn = torch.where(freq>f0, 0, F_syn)
        syn = torch.fft.irfft(F_syn)
        loss = torch.mean(torch.square(obs - syn))
        return loss 


    def train(self, init_model, lr, f0, batch_size=2, 
                max_niters=50, print_interval=10):
        loss_list = list()
        niters = 0
        time_begin = time.time()
        print("Training begin")
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        updater = torch.optim.Adam(init_model, lr=lr)
        while niters < max_niters:
            for i, data in enumerate(dataloader):
                sx, obs = data 
                updater.zero_grad()
                syn =  gen_data(self.init_model, acoustic2d, self.source_time, 
                        sx, self.h, self.dt, self.receivers_x)
                loss = self.compute_loss(syn, obs, f0)
                loss.backward()
                updater.step()            
                
            with torch.no_grad():
                syn = gen_data(self.init_model, self.source_time, 
                    self.sources_x, self.h, self.dt, self.receivers_x)
                loss = torch.mean(torch.square(syn - self.dataset.signal)).item()
                loss_list.append(loss)

                if (niters + 1) % print_interval == 0:
                    print(f'epoch: {niters+1}, data loss: {loss:g}')

            niters += 1
        time_end = time.time()
        print(f"Training finished, total time{time_begin-time_end:.1f}s")

        model_inverted = self.init_model.detach().cpu().numpy()
        return loss_list, model_inverted


def smoothing(model, h=10, w=10, mode='gaussian', sigma=0.01):
    """
    Smoothing the standard model to create initial model
    using average pooling

    Parameters:
    -----------
    h: int
        Hight of box for average pooling
    w: int
        Width of box for average pooling

    Returns:
    --------
    torch.Tensor with the same shape as `model`
        Model used as initial guess in FWi
    """
    # pading with edge values
    padding_left = (w - 1) // 2
    padding_right = w - 1 - padding_left
    padding_top = (h - 1) // 2
    padding_bottom = h - 1 - padding_top
    m = torch.nn.ReplicationPad2d((padding_left, padding_right, 
                                    padding_top, padding_bottom))
    init = model.unsqueeze(0).unsqueeze(0)
    init = m(init)
    
    # kernel
    if mode == 'mean':
        kernel = torch.ones(h, w) / (h*w)
    elif mode == 'gaussian':
        x = torch.arange(w)
        y = torch.arange(h)
        x = -(x - (w-1)/2)**2 / sigma**2
        y = -(y - (h-1)/2)**2 / sigma**2
        kernel = torch.exp(x.reshape(1, -1) + y.reshape(-1, 1))
    else:
        raise NotImplementedError("Not implemented till now!")
        
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    init = torch.nn.functional.conv2d(init, kernel).squeeze()
    assert init.shape == model.shape 
    return init